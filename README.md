# ImageInt

Bilderzeugung als eigener Dienst für den LLMInt-Chat. ImageInt stellt
**Qwen-Image-2.1** hinter einer kleinen HTTP-API bereit, die das jeweils laufende
Text-LLM in LLMInt als Tool `generate_image` aufruft.

ImageInt ist nach dem Vorbild von [SpeechInt](https://github.com/dareinelt/SpeechInt)
gebaut und **ersetzt die Integrationen von ComfyUI und AUTOMATIC1111**. Die
Tool-Bezeichnung `generate_image` wird von diesen übernommen, damit die
vorhandenen Prompts und Tool-Definitionen in LLMInt unverändert weiterlaufen.

```
LLMInt (Text-LLM)
   │  Tool-Call generate_image { prompt }
   ▼
ImageInt gateway  ──►  Prompt-Enhancer (Qwen-Image-2.1-PE-T2I, transformers, CPU)
   │                └►  Bildmodell    (Qwen-Image-2.1, diffusers + transformers, CPU)
   ▼
fertiges Bild + Job-URL  ──►  Chat  (und optional per E-Mail benachrichtigt)
```

---

## Inhalt

- [Anforderungen](#anforderungen)
- [Schnellstart](#schnellstart)
- [Architektur](#architektur)
  - [Warum `diffusers` und nicht vLLM-Omni](#warum-diffusers-und-nicht-vllm-omni)
  - [Warum der Enhancer ebenfalls nicht auf vLLM läuft](#warum-der-enhancer-ebenfalls-nicht-auf-vllm-läuft)
  - [Prompt-Enhancer](#prompt-enhancer)
- [Speicherbudget und Profile](#speicherbudget-und-profile)
- [CPU-Modus](#cpu-modus)
- [CUDA (vorbereitet)](#cuda-vorbereitet)
- [Ladefortschritt](#ladefortschritt)
- [API](#api)
- [LLMInt-Integration](#llmint-integration)
- [E-Mail-Benachrichtigung](#e-mail-benachrichtigung)
- [Konfiguration](#konfiguration)
- [Tests](#tests)
- [Projektstruktur](#projektstruktur)
- [Lizenz](#lizenz)

---

## Anforderungen

Dimensioniert und geprüft ist ImageInt für:

| | Anforderung | Verhalten bei Verletzung |
|---|---|---|
| CPU-Kerne | **mindestens 8** | Warnung im Log, `host.cpu_cores_ok = false` |
| CPU-Befehlssatz | **AVX2** (x86-64) | **harter Blocker** auf Bare Metal |
| RAM | **mindestens 32 GB** | Warnung im Log, `host.memory_ok = false` |
| AVX-512 | optional | **nur informativ**, kein Blocker |
| GPU | optional | CPU-only ist der Standard; CUDA ist vorbereitet |

Der Gateway prüft das beim Start und meldet das Ergebnis unter `GET /v1/config`
im Feld `host`. Fehlt AVX2 auf einem Bare-Metal-Host, startet der Prozess nicht;
auf einer **virtuellen Maschine oder in einem Container** wird ein fehlendes
AVX2 nur gewarnt. Grund: Hypervisoren maskieren CPUID-Flags, und Docker Desktop
reicht AVX2 auch auf einer fähigen CPU nicht durch — der Check darf auf einer VM
also nicht fehlschlagen.

```bash
# Flags der eigenen CPU ansehen
lscpu | grep -o 'avx2\|avx512[a-z]*' | sort -u

# Check-Modus: auto (Standard) | strict | off
IMAGEINT_HOST_CHECK=auto docker compose up -d
```

| Modus | Bedeutung |
|---|---|
| `auto` | AVX2 blockiert nur auf Bare Metal; auf VM/Container Warnung |
| `strict` | AVX2 blockiert immer — nur für Hosts mit vertrauenswürdigen Flags |
| `off` | blockiert nie, protokolliert nur |

---

## Schnellstart

```bash
git clone https://github.com/dareinelt/ImageInt.git
cd ImageInt
cp .env.example .env          # optional, jeder Wert hat einen Standard
docker compose build          # baut gateway, enhancer und image lokal
docker compose up -d
docker compose logs -f gateway
```

Alle drei Images entstehen lokal aus diesem Repository; es wird kein
Fremd-Image gezogen. Der erste Build lädt Torch und die Python-Abhängigkeiten
(ca. 10 Minuten), der erste Start danach den Enhancer mit ca. 19 GB und das
Bildmodell mit ca. 33 GB nach `hf_cache`. Auf einer CPU dauert das je nach
Leitung und Platte 10–40 Minuten; in dieser Zeit antwortet `GET /v1/ready` mit
**503** und `Retry-After`, nicht mit einem Fehler.

Danach:

```bash
# Bereitschaft
curl -s localhost:8080/v1/ready | python3 -m json.tool

# Ein Bild erzeugen (Prompt ist das einzige Pflichtfeld)
curl -s -X POST localhost:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Ein roter Panda im Schnee, fotografisch"}' | python3 -m json.tool
```

### Ohne Modell-Download testen

Für API- und Docker-Tests gibt es in **beiden** Modellservern einen Stub-Engine,
der keine Gewichte lädt: der Bildserver zeichnet ein Platzhalterbild mit Pillow,
der Enhancer antwortet mit einem festen, wohlgeformten JSON-Prompt.

```bash
IMAGEINT_IMAGE_ENGINE=stub IMAGEINT_IMAGE_STUB_DELAY=3 \
IMAGEINT_PE_ENGINE=stub IMAGEINT_PE_STUB_DELAY=2 \
  docker compose up -d
```

Damit lässt sich die komplette Strecke inklusive 202/Polling und
E-Mail-Benachrichtigung ohne GPU oder 50 GB Download durchspielen. Für einen
Lauf gegen einen echten Enhancer einfach `IMAGEINT_PE_ENGINE` weglassen.

---

## Architektur

Drei Container, nur einer davon veröffentlicht:

| Dienst | Inhalt | Port | Rolle |
|---|---|---|---|
| `gateway` | `gateway/` — FastAPI | **8080 → Host** | einzige öffentliche API |
| `enhancer` | `enhancer_server/` — transformers + torch | intern 8000 | Prompt-Enhancer |
| `image` | `image_server/` — diffusers + transformers | intern 8000 | Bildmodell |

Der Gateway hält **keine Gewichte**. Er kennt die Prompts, die
Sampling-Parameter und die Modell-IDs als Konfiguration, sodass ein
Administrator alles in LLMInt einstellen kann und die Modell-Container
austauschbar bleiben.

### Warum `diffusers` und nicht vLLM-Omni

Qwen-Image-2.1 ist ein Diffusers-Checkpoint (`QwenImage21Pipeline`). Die
vLLM-Variante, die ihn bedienen könnte, ist **vLLM-Omni** — und die veröffentlicht
ausschließlich **GPU-Images**. Die Anforderung „vLLM läuft im CPU-only-Modus" ist
damit nicht erfüllbar, deshalb wird das Bildmodell von `image_server/` bedient:

- **`diffusers` ≥ 0.41.0** für die Pipeline (ältere Versionen kennen
  `QwenImage21Pipeline` nicht),
- **`transformers` ≥ 5.17** für Text-Encoder und Tokenizer,
- **`torch`** in der CPU-Variante, ohne CUDA-Bibliotheken im Referenz-Image,
- **`optimum-quanto`** für die int8/int4-Quantisierung auf der CPU.

Der Container spricht trotzdem die **OpenAI Images API**, also blieb auf der
Gateway-Seite beim Wechsel von vLLM-Omni auf diffusers kein Code zu ändern.

### Warum der Enhancer ebenfalls nicht auf vLLM läuft

Der Prompt-Enhancer ist ein reines Textmodell, aber der CPU-Build von vLLM kann
dieses Checkpoint nicht initialisieren. `Qwen-Image-2.1-PE-T2I` ist ein
Qwen3.5-VL 9B, dessen verschachtelte Text-Config keinen expliziten
`architectures`-Eintrag trägt; das CPU-Backend von vLLM lehnt das beim Start der
Engine ab. `enhancer_server/` lädt denselben Checkpoint deshalb über
**transformers**, das den Remote-Code des Checkpoints ausführt.

Damit laufen **beide** Modell-Container auf demselben Stack (`torch`,
`transformers`, CPU-Wheels ohne CUDA), teilen sich dieselbe
Genauigkeitsrichtlinie (bf16) und denselben Health-Vertrag (200 = bereit,
503 = lädt). Der Gateway merkt davon nichts: der Enhancer spricht die
**OpenAI Chat-Completions API** über SSE, inklusive `reasoning_content` für den
Denkblock — genau die Form, die der Gateway schon gelesen hat.

**llama.cpp wurde geprüft und verworfen.** Es wäre technisch möglich, aber es
bräuchte eine GGUF-Konvertierung des 9B-Checkpoints und würde dessen eigenes
Chat-Template samt Thinking-Behandlung verlieren; die Antwort, die der Gateway
parst (`rewritten_prompt`, `negative_prompt`, `wh_ratio`), hängt aber genau an
diesem Template. Da `transformers` im Bild-Container ohnehin installiert ist,
wäre llama.cpp ein zusätzlicher Stack ohne Gegenwert.

### Prompt-Enhancer

Der in der Qwen-Image-2.1-Modellkarte dokumentierte **PE-T2I-Enhancer**
(`Qwen/Qwen-Image-2.1-PE-T2I`) ist verpflichtender Bestandteil der Strecke. Er
ist ein feinabgestimmtes Qwen3.5-VL 9B, das den kurzen Nutzerwunsch in den langen
strukturierten Prompt umschreibt, auf den das Bildmodell trainiert wurde, und
dabei zusätzlich das Seitenverhältnis wählt:

```json
{"rewritten_prompt": "…", "wh_ratio": 1.7777778}
```

Sein System-Prompt ist Teil des Checkpoints und liegt **wörtlich** in
[`gateway/app/prompts/pe_t2i_system_prompt.txt`](gateway/app/prompts/pe_t2i_system_prompt.txt).
Diese Datei darf nicht verändert werden — sie ist kein Stilmittel, sondern die
Eingabe, mit der das Modell trainiert wurde. Die Sampling-Parameter
(Temperatur 1.0, top-p 0.95, top-k 20, presence penalty 1.5) stehen in
`.env.example` und sind ebenfalls die dokumentierten Werte.

Wird `wh_ratio` nicht geparst, fällt der Gateway auf das native 2K-Format
zurück, statt die Anfrage abzulehnen; das Feld `parse_ok` im Job zeigt es an.

---

## Speicherbudget und Profile

bf16 ist das Format der Modellkarte und damit der Standard:

| Komponente | bf16 | int8 |
|---|---|---|
| Text-Encoder | 17,5 GB | ~8,8 GB |
| DiT (Denoiser) | 14 GB | ~7 GB |
| VAE | 1,35 GB | 1,35 GB |
| **Bildmodell gesamt** | **~33 GB** | **~17 GB** |
| Enhancer (PE-T2I 9B) | ~19 GB + ~2 GB KV | ~10 GB |
| Gateway | ~0,4 GB | ~0,4 GB |
| **beide Modelle resident** | **~57 GB** | **~29 GB** |

Das passt **nicht** auf die Referenzmaschine mit 32 GB. Deshalb sind mehrere
Profile dokumentiert:

```bash
# ── Host mit >= 64 GB RAM ────────────────────────────────────────────────────
docker compose up -d

# ── Host mit >= 32 GB RAM (Referenzmaschine) ─────────────────────────────────
IMAGEINT_IMAGE_QUANT=int8 docker compose up -d          # Bildmodell ~17 GB

# Enhancer ebenfalls verkleinern -> beide Modelle bleiben nutzbar
IMAGEINT_IMAGE_QUANT=int8 IMAGEINT_PE_QUANT=int8 docker compose up -d

# Enhancer erst bei Bedarf laden statt beim Start (spart im Leerlauf alles)
IMAGEINT_IMAGE_QUANT=int8 IMAGEINT_PE_PRELOAD=false docker compose up -d

# Enhancer ganz weglassen -> Prompts werden wörtlich verwendet
IMAGEINT_IMAGE_QUANT=int8 IMAGEINT_PE_ENABLED=false docker compose up -d
```

`IMAGEINT_PE_PRELOAD=false` ist der interessanteste der Kompromisse: der
Enhancer bleibt konfiguriert und liefert weiterhin die dokumentierte Qualität,
belegt aber erst dann 19 GB, wenn wirklich ein Bild angefordert wird. Der
Gateway stößt das Laden über `POST /v1/warmup` an, sobald er den Enhancer zum
ersten Mal braucht (siehe `docs/api.md`).

Beide Container haben harte Obergrenzen (`mem_limit 40g` für das Bildmodell,
`24g` für den Enhancer). Läuft der Host in ein Limit, bricht der Kernel den
Prozess ab — das ist gewollt, weil ein zu groß dimensionierter Dienst sonst den
ganzen Host in den Swap zwingt. Die Werte stehen in `.env.example` und lassen
sich dort anpassen.

> **Hinweis zu int4.** `IMAGEINT_IMAGE_QUANT=int4` halbiert den Bedarf erneut
> (~9 GB) und ist für einen Testhost oder ein sehr kleines Modell gedacht. Der
> Qualitätsverlust ist sichtbar; für den Produktivbetrieb ist int8 die Grenze.
> Für Tests auf einem Entwicklungsrechner ist `IMAGEINT_IMAGE_ENGINE=stub` die
> bessere Wahl, weil es gar keine Gewichte lädt.

---

## CPU-Modus

CPU-only ist der Standard und die einzige vollständig verifizierte
Betriebsart. Wichtig für die Laufzeit:

| Stellschraube | Wirkung |
|---|---|
| `IMAGEINT_IMAGE_STEPS` | 40 ist dokumentiert; 20 halbiert die Zeit bei sichtbarer Einbuße |
| `IMAGEINT_IMAGE_TRUE_CFG_SCALE` | 1.0 ist der Standard (ohne Guidance). > 1.0 verdoppelt die Rechenzeit fast, weil die Pipeline zweimal pro Schritt läuft |
| `IMAGEINT_IMAGE_QUANT` | int8 halbiert Speicher **und** beschleunigt auf einer CPU spürbar |
| `IMAGEINT_IMAGE_CPU_THREADS` | 0 = ein Thread pro Kern, auf der 8-Kern-Referenz richtig |
| `IMAGEINT_IMAGE_KV_CACHE` | Prefix-Cache des DiT, spart viel Wiederholung bei mehreren Bildern |
| `IMAGEINT_IMAGE_MAX_PIXELS` | kleinere Fläche = quadratisch weniger Rechenzeit |
| `IMAGEINT_PE_QUANT` | int8 halbiert die Enhancer-Gewichte auf ~10 GB |
| `IMAGEINT_PE_ENABLE_THINKING=false` | überspringt den Denkblock des Enhancers und spart viel CPU-Zeit, kostet Prompt-Qualität |
| `IMAGEINT_PE_PRELOAD=false` | Enhancer lädt erst beim ersten Bild statt beim Start |
| `IMAGEINT_PE_ENABLED=false` | überspringt den Enhancer komplett (Minuten!). Auch Aufrufe, die `enhance` gar nicht mitschicken, laufen dann ohne Enhancer |

Eine CPU-Renderzeit von mehreren Minuten für 2K ist normal und der Grund für die
202/Polling-Architektur und die E-Mail-Benachrichtigung.

---

## CUDA (vorbereitet)

NVIDIA/CUDA ist **vorbereitet, aber nicht verifiziert**. Die Umschaltung ist ein
Override-File:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d
```

Voraussetzungen: NVIDIA-Treiber und das
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
geprüft mit

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

Das Override baut `image_server` **und** `enhancer_server` mit dem CUDA-Target,
setzt `IMAGEINT_IMAGE_DEVICE=cuda` bzw. `IMAGEINT_PE_DEVICE=cuda` und reicht die
GPUs durch. Auf einem CUDA-Host entfällt der AVX2-Teil der Anforderung;
`GET /v1/config` meldet ihn als `"entfällt"`. Das VRAM-Budget steht in den
Kommentaren des Override-Files
(Enhancer ~19 GB, Bildmodell ~33 GB in bf16, also nicht auf eine 24-GB-Karte
gleichzeitig).

---

## Ladefortschritt

Ein Kaltstart dauert Minuten bis Stunden. Damit ein Client das nicht als Fehler
liest, ist der Zustand explizit modelliert:

| Endpunkt | Antwort | Bedeutung |
|---|---|---|
| `GET /health` | immer 200 | Prozess lebt; fragt **kein** Modell |
| `GET /v1/ready` | 200 | alle benötigten Komponenten fertig |
| `GET /v1/ready` | 503 + `Retry-After` | lädt noch — Anfrage behalten, später erneut |
| `GET /v1/models` | 200 | Zustand beider Modellserver einzeln |

Der Zustand ist `loading`, solange ein Modellserver noch nie geantwortet hat und
die Frist `IMAGEINT_STARTING_GRACE_SECONDS` (Standard 1800 s) nicht abgelaufen
ist. Erst danach gilt er als `error`. So unterscheidet der Dienst „braucht noch"
von „ist kaputt", ohne dass ein Client eine Zeitüberschreitung selbst deuten muss.

Die Docker-Healthchecks akzeptieren deshalb **200 oder 503**: ein Container, der
gerade 33 GB lädt, ist gesund, und Docker soll ihn nicht neu starten.

Mit `IMAGEINT_IMAGE_PRELOAD=false` lädt das Bildmodell erst beim ersten Auftrag.
Damit die dokumentierte Empfehlung „später erneut" dann nicht ins Leere läuft,
ruft der Gateway den Bildserver über `POST /v1/warmup` an, sobald er ihn im
Zustand `loading` sieht. Der Client merkt davon nichts: er bekommt weiterhin
dieselbe **503** mit `Retry-After`, nur findet der Wiederholungsversuch nach 15
Sekunden dann tatsächlich ein warmes Modell vor. Voreinstellung ist deshalb
`true`.

Für den Enhancer gilt dasselbe über `IMAGEINT_PE_PRELOAD=false`. Beide
Modellserver implementieren `POST /v1/warmup`, und der Gateway behandelt sie
gleich — ohne diese Route wäre `PRELOAD=false` eine Sackgasse, weil der Gateway
erst sendet, wenn er 200 gesehen hat.

---

## API

Alle Endpunkte außer `/health` und `/v1/ready` verlangen das gemeinsame Secret
als `X-Auth-Token` oder `Authorization: Bearer`. Vollständige Referenz mit
Beispielen: [`docs/api.md`](docs/api.md).

| Methode | Pfad | Zweck |
|---|---|---|
| `GET` | `/` | Banner mit Version und Endpunktliste |
| `GET` | `/health` | Liveness (Container-Healthcheck) |
| `GET` | `/v1/ready` | Bereitschaft, unauthentifiziert (200/503) |
| `GET` | `/v1/health` | Bereitschaft, authentifiziert |
| `GET` | `/v1/config` | wirksame Konfiguration, Tokens maskiert |
| `GET` | `/v1/models` | beide Modellserver und ihr Zustand |
| `GET` | `/v1/ratios` | Leinwand-Tabelle der Modellkarte |
| `POST` | `/v1/enhance` | nur der Enhancer (Debug/Admin-Test) |
| `POST` | `/v1/images/generations` | **das Bild erzeugen** |
| `GET` | `/v1/jobs` | letzte Aufträge |
| `GET` | `/v1/jobs/{id}` | Zustand eines Auftrags |
| `GET` | `/v1/jobs/{id}/image` | fertiges Bild |
| `GET` | `/v1/images/{id}` | fertiges Bild (URL für Clients) |

### Der Kern: `POST /v1/images/generations`

```bash
curl -s -X POST localhost:8080/v1/images/generations \
  -H 'X-Auth-Token: <token>' -H 'Content-Type: application/json' \
  -d '{
        "prompt": "Ein roter Panda im Schnee",
        "size": "16:9",
        "seed": 1234
      }'
```

`prompt` ist das einzige Pflichtfeld. LLMInt sendet nur diesen — alle anderen
Felder existieren, damit ein Client eine Entscheidung des Enhancers übersteuern
kann.

**Antwort 200** (fertig innerhalb von `IMAGEINT_SYNC_TIMEOUT`):

```json
{
  "ok": true,
  "job_id": "8f3c1a2b4d5e6f70",
  "status": "done",
  "prompt": "Ein roter Panda im Schnee",
  "width": 2400, "height": 1350, "wh_ratio": 1.7778, "seed": 1234,
  "image_url": "http://localhost:8080/v1/images/8f3c1a2b4d5e6f70",
  "status_url": "http://localhost:8080/v1/jobs/8f3c1a2b4d5e6f70",
  "b64_json": "iVBORw0KGgo…",
  "content_type": "image/png"
}
```

**Antwort 202** (dauert länger — auf einer CPU der Normalfall):

```json
{
  "ok": true,
  "job_id": "8f3c1a2b4d5e6f70",
  "status": "running",
  "stage": "enhancing",
  "status_url": "http://localhost:8080/v1/jobs/8f3c1a2b4d5e6f70",
  "image_url": "http://localhost:8080/v1/images/8f3c1a2b4d5e6f70",
  "poll_after_seconds": 15,
  "message": "Das Bild wird noch erzeugt. …"
}
```

Der Job bleibt `IMAGEINT_JOB_RETENTION_SECONDS` (Standard 24 h) abrufbar — das
Fenster muss „Nutzer liest die Mail und klickt den Link" abdecken, denn genau
diese URL steht in der Mail.

`stage` zeigt, in welcher Hälfte gewartet wird (`enhancing` / `rendering`), weil
allein der Enhancer auf einer CPU minutenlang „denken" kann.

---

## LLMInt-Integration

LLMInt trägt ImageInt als Bild-Endpunkt ein. Das Tool `generate_image` wird dem
jeweils laufenden Text-LLM präsentiert; das Modell erkennt die Notwendigkeit
selbst an eindeutigen Aufforderungen („Generiere ein Bild", „Erstelle eine
Grafik", „Zeige mir ein Bild von …"). Details und der vollständige Ablauf:
[`docs/llmint-integration.md`](docs/llmint-integration.md).

---

## E-Mail-Benachrichtigung

Weil eine Generierung Minuten dauern kann, kann der Nutzer per E-Mail über die
Fertigstellung informiert werden — **nur mit Zustimmung**:

> Das Generieren dieser Antwort kann einige Zeit in Anspruch nehmen. Möchten Sie
> per E-Mail über die Fertigstellung benachrichtigt werden?

Die Mail enthält einen Deep-Link, der direkt in den Chat mit dem fertig
generierten Bild springt. **Mail-Text und Zustimmungsfrage sind über Platzhalter
voll individualisierbar** (dasselbe `{placeholder}`-Muster wie bei der
Registrierungsmail in LLMInt):

| Platzhalter | Inhalt |
|---|---|
| `{sitename}` | Name der Installation |
| `{username}` | Anzeigename des Nutzers |
| `{prompt}` | der ursprüngliche Bildwunsch |
| `{chat_url}` | Deep-Link in den Chat mit dem fertigen Bild |
| `{image_url}` | direkte URL des Bildes |
| `{duration}` | Dauer der Generierung |

---

## Konfiguration

Alles läuft über Umgebungsvariablen. Vorlage: [`.env.example`](.env.example) —
jede Variable ist dort kommentiert, und jede hat einen funktionierenden
Standard, ein leeres `.env` startet also.

Die wirksame Konfiguration zeigt `GET /v1/config`; Tokens werden dabei maskiert
ausgegeben. Variablen des Gateways beginnen mit `IMAGEINT_`, die des Bildmodells
zusätzlich mit `IMAGEINT_IMAGE_`, die des Enhancers mit `IMAGEINT_PE_`.
`IMAGEINT_PE_MODEL` und `IMAGEINT_PE_TOKEN` werden von beiden Seiten gelesen und
müssen deshalb nirgends doppelt gepflegt werden.

---

## Tests

```bash
# Gateway (FastAPI, Job-Registry, Enhancer, Hardware-Check)
gateway/.venv/bin/python -m pytest gateway -q

# Bildserver (Config, Engine, HTTP-API, Vertrag gegen den Gateway)
gateway/.venv/bin/python -m pytest image_server -q

# Enhancerserver (Config, Engine, HTTP-API, Vertrag gegen den Gateway)
gateway/.venv/bin/python -m pytest enhancer_server -q
```

Die drei Suiten laufen **einzeln**: jede hat ein eigenes `tests/conftest.py`,
und `gateway/tests/test_vllm.py` importiert daraus direkt. Ein gemeinsamer Lauf
`pytest gateway image_server enhancer_server` löst deshalb das falsche Conftest
auf.

`image_server/tests/test_gateway_contract.py` und
`enhancer_server/tests/test_gateway_contract.py` sind die Wächter gegen stilles
Auseinanderlaufen: sie starten den jeweiligen Modellserver auf einem echten
Socket und schicken ihm den echten Request-Body des Gateways — für das
Bildmodell bis zum PNG, für den Enhancer bis zum geparsten JSON
(`rewritten_prompt`, `negative_prompt`, `wh_ratio`).

Die Tests brauchen **keine Gewichte und keine GPU** — beide Server laufen im
Stub-Engine-Modus, der Enhancer antwortet dann mit einem festen Prompt.

---

## Projektstruktur

```
ImageInt/
├── docker-compose.yml            # CPU-Stack (Standard)
├── docker-compose.cuda.yml       # CUDA-Override (vorbereitet)
├── .env.example                  # kommentierte Konfigurationsvorlage
├── docs/
│   ├── api.md                    # API-Referenz mit Beispielen
│   └── llmint-integration.md     # Tool-Anbindung in LLMInt
├── gateway/                      # öffentliche API (FastAPI)
│   ├── app/
│   │   ├── main.py               # Endpunkte, Auth, Fehlerbehandlung
│   │   ├── config.py             # Settings aus der Umgebung
│   │   ├── host.py               # AVX2/AVX-512/VM-Check
│   │   ├── enhancer.py           # Prompt-Enhancer (PE-T2I)
│   │   ├── prompts/              # System-Prompt, wörtlich vom Checkpoint
│   │   ├── vllm.py               # HTTP-Client zu den Modellservern
│   │   ├── pipeline.py           # ein Auftrag von Anfang bis Ende
│   │   ├── jobs.py               # Job-Registry mit Warteschlange
│   │   ├── storage.py            # Bilder auf der Platte
│   │   ├── health.py             # Ladezustand beider Modelle
│   │   ├── ratios.py             # Leinwände der Modellkarte
│   │   └── errors.py             # Fehlercodes und HTTP-Status
│   └── tests/
├── image_server/                 # Bildmodell (diffusers, CPU)
│   ├── app.py                    # OpenAI-Images-API
│   ├── engine.py                 # DiffusersEngine / StubEngine
│   ├── config.py                 # Settings des Bildmodells
│   ├── Dockerfile                # Targets: base / cpu (Standard) / cuda
│   └── tests/
└── enhancer_server/              # Prompt-Enhancer (transformers, CPU)
    ├── app.py                    # OpenAI-Chat-Completions-API (SSE)
    ├── engine.py                 # TransformersEngine / StubEngine
    ├── config.py                 # Settings des Enhancers
    ├── Dockerfile                # Targets: base / cpu (Standard) / cuda
    └── tests/
```

Die beiden Modellserver sind bewusst gleich aufgebaut: gleiche Zustandsmaschine
(`idle → loading → ready | error`), gleiche Konfiguration über die Umgebung,
gleiche Routen (`/health`, `/v1/models`, `POST /v1/warmup`) und je ein
Vertragstest gegen den echten Gateway-Client.

---

## Lizenz

Für den Quellcode dieses Repositories ist noch keine Lizenzdatei hinterlegt.
Die Modellgewichte unterliegen den Lizenzen der jeweiligen
Hugging-Face-Repositories
([Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
[PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I)); sie werden nicht
mitgeliefert, sondern beim ersten Start in den `hf_cache` geladen.
