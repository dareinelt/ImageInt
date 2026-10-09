# ImageInt API

Referenz der HTTP-API des ImageInt-Gateways. Alle Beispiele nehmen an, dass der
Dienst unter `http://localhost:8080` läuft.

- [Authentifizierung](#authentifizierung)
- [Fehlerformat](#fehlerformat)
- [Zustandsendpunkte](#zustandsendpunkte)
- [Bilder erzeugen](#bilder-erzeugen)
- [Aufträge abfragen](#aufträge-abfragen)
- [Nur den Enhancer](#nur-den-enhancer)
- [Tabellen](#tabellen)
- [Fehlercodes](#fehlercodes)
- [Der Bildserver](#der-bildserver)
- [Der Enhancerserver](#der-enhancerserver)

---

## Authentifizierung

Ist `IMAGEINT_TOKEN` gesetzt, verlangt jeder Endpunkt **außer** `/health` und
`/v1/ready` das Secret in einem dieser beiden Header:

```
X-Auth-Token: <token>
Authorization: Bearer <token>
```

Fehlt es oder ist es falsch, antwortet der Dienst mit **401**:

```json
{"ok": false, "error": "unauthorized", "message": "Ungültiger oder fehlender Token."}
```

Ist `IMAGEINT_TOKEN` leer, ist die API offen. Das ist nur in einem
vertrauenswürdigen Netz vertretbar. `/health` und `/v1/ready` bleiben immer
offen, weil sie die Container-Healthchecks bedienen und keine Daten preisgeben.

---

## Fehlerformat

Jede fehlgeschlagene Anfrage hat dieselbe Hülle:

```json
{"ok": false, "error": "<code>", "message": "<Text für den Administrator>"}
```

`error` ist ein stabiler Bezeichner, auf den ein Client verzweigen kann;
`message` ist für Menschen gedacht und wird in LLMInt im Admin-Bereich wörtlich
angezeigt.

---

## Zustandsendpunkte

### `GET /health`

Liveness. Antwortet **immer 200**, solange der Prozess läuft, und kontaktiert
**kein** Modell. Das ist der Docker-Healthcheck.

```json
{"ok": true, "status": "alive", "service": "imageint",
 "version": "1.0.0", "uptime_seconds": 412.7}
```

### `GET /v1/ready`

Bereitschaft, **unauthentifiziert**. Der Endpunkt, den ein Client pollen soll.

**200** — alles bereit:

```json
{
  "ok": true,
  "service": "imageint",
  "version": "1.0.0",
  "state": "ready",
  "transient": false,
  "host": {"avx2": true, "avx512": true, "cpu_cores": 8, "memory_gb": 32.0, "…": "…"},
  "components": {
    "enhancer": {"state": "ready", "ready": true, "url": "http://enhancer:8000", "…": "…"},
    "image":    {"state": "ready", "ready": true, "url": "http://image:8000", "…": "…"}
  },
  "jobs": {"running": 0, "queued": 0, "done": 3, "…": "…"}
}
```

**503** — lädt noch. Header `Retry-After`, zusätzlich im Body `retry_after`:

```json
{
  "ok": false,
  "state": "loading",
  "transient": true,
  "retry_after": 15,
  "components": {"image": {"state": "loading", "message": "Modell wird geladen …"}},
  "…": "…"
}
```

> `transient: true` heißt: **Anfrage behalten und später erneut senden**, nicht
> dem Nutzer einen Fehler zeigen. Genau diese Unterscheidung braucht der
> Chat, weil ein Kaltstart Minuten bis Stunden dauern kann.

### `GET /v1/health`

Wie `/v1/ready`, aber mit Token. Praktisch für das Admin-Formular in LLMInt,
das den Token ohnehin hat.

### `GET /v1/config`

Die wirksame Konfiguration. Tokens werden maskiert ausgegeben.

```json
{
  "ok": true,
  "service": "imageint",
  "version": "1.0.0",
  "public_url": "",
  "auth_required": true,
  "token_masked": "ab12…ef90",
  "limits": {
    "max_prompt_chars": 4000,
    "sync_timeout_seconds": 120,
    "max_concurrent_jobs": 1,
    "max_queued_jobs": 2,
    "max_jobs": 200,
    "job_retention_seconds": 86400,
    "poll_after_seconds": 15
  },
  "storage_dir": "/data/images",
  "enhancer": {"model": "Qwen/Qwen-Image-2.1-PE-T2I", "enabled": true, "…": "…"},
  "image": {
    "url": "http://image:8000",
    "model": "Qwen/Qwen-Image-2.1",
    "quant": "none",
    "route": "images",
    "route_path": "/v1/images/generations",
    "steps": 40,
    "true_cfg_scale": 1.0,
    "max_pixels": 4300800,
    "timeout": 1800
  },
  "host": {
    "avx2": true,
    "avx512": true,
    "avx512_note": "AVX-512 ist vorhanden und beschleunigt die Denoising-Schritte.",
    "virtualization": null,
    "cpu_cores": 8,
    "memory_gb": 32.0,
    "check_mode": "auto",
    "blocking": [],
    "warnings": []
  }
}
```

Das Feld `host` ist die einzige Stelle, an der die Hardwareanforderung sichtbar
wird. `blocking` ist leer, wenn der Dienst laufen darf. Auf einer virtuellen
Maschine enthält `warnings` einen Hinweis statt eines Blockers, und `avx2`
kann `false` sein, ohne dass der Prozess startet nicht — das ist gewollt.

### `GET /v1/models`

Beide Modellserver einzeln, mit erzwungenem frischem Probe (umgeht den
`IMAGEINT_HEALTH_CACHE_SECONDS`-Cache). Das ist der Endpunkt für „warum dauert
das noch".

```json
{
  "ok": true,
  "ready": true,
  "quant": "none",
  "models": {
    "enhancer": {"url": "http://enhancer:8000", "model": "Qwen/Qwen-Image-2.1-PE-T2I",
                 "state": "ready", "ready": true, "http": 200,
                 "message": "", "role": "prompt-enhancer"},
    "image":    {"url": "http://image:8000", "model": "Qwen/Qwen-Image-2.1",
                 "state": "loading", "ready": false, "http": 503,
                 "message": "Modell wird geladen …", "role": "image-generation"}
  }
}
```

### `GET /v1/ratios`

Die Leinwände der Modellkarte und die Regel für unbekannte Verhältnisse.

```json
{
  "ok": true,
  "documented": {
    "1:1":    {"width": 2048, "height": 2048},
    "16:9":   {"width": 2752, "height": 1536},
    "9:16":   {"width": 1536, "height": 2752},
    "4:3":    {"width": 2400, "height": 1792},
    "3:4":    {"width": 1792, "height": 2400},
    "3:2":    {"width": 2400, "height": 1696},
    "2:3":    {"width": 1696, "height": 2400}
  },
  "default": {"width": 2048.0, "height": 2048.0},
  "max_pixels": 4300800,
  "multiple": 64,
  "note": "Nicht dokumentierte Verhältnisse werden auf dieselbe Fläche wie das native 2K-Bild abgebildet, auf ein Vielfaches von 64 gerundet und auf IMAGEINT_IMAGE_MAX_PIXELS begrenzt."
}
```

---

## Bilder erzeugen

### `POST /v1/images/generations`

Der eigentliche Endpunkt und der einzige, den LLMInt im Normalbetrieb aufruft.

#### Anfrage

| Feld | Typ | Standard | Bedeutung |
|---|---|---|---|
| `prompt` | string | **Pflicht** | Der Bildwunsch. Länger als `IMAGEINT_MAX_PROMPT_CHARS` → 413 |
| `size` | string | `""` | `"1024x1024"`, `"16:9"` oder `"4:3"` |
| `ratio` | string | `""` | Seitenverhältnis, falls kein `size` |
| `width` | int | `0` | Breite; `0` = vom Enhancer bestimmen |
| `height` | int | `0` | Höhe; `0` = vom Enhancer bestimmen |
| `enhance` | bool \| null | `null` | Enhancer verwenden. `null` = Einstellung des Betreibers (`IMAGEINT_PE_ENABLED`) |
| `steps` | int | `0` | Denoising-Schritte; `0` = Standard aus der Konfiguration |
| `seed` | int | `null` | Für reproduzierbare Bilder |
| `negative_prompt` | string | `""` | Überschreibt den des Enhancers |
| `wait` | bool | `true` | Bis `IMAGEINT_SYNC_TIMEOUT` warten, sonst sofort 202 |
| `include_image` | bool | `true` | Das Bild als base64 in die Antwort legen |

Ein Aufruf, wie LLMInt ihn macht — nur `prompt`:

```bash
curl -s -X POST localhost:8080/v1/images/generations \
  -H 'X-Auth-Token: <token>' -H 'Content-Type: application/json' \
  -d '{"prompt":"Zeige mir ein Bild von einem roten Panda im Schnee"}'
```

Weil LLMInt `enhance` nicht mitschickt, gilt für solche Aufrufe die Einstellung
des Betreibers: mit `IMAGEINT_PE_ENABLED=true` läuft der dokumentierte Enhancer,
mit `IMAGEINT_PE_ENABLED=false` wird der Prompt wörtlich verwendet. Ein
ausdrückliches `"enhance": true` gegen einen abgeschalteten Enhancer ist ein
echter Widerspruch und wird mit `503 not_configured` und einer Erklärung
abgelehnt, statt still etwas anderes zu tun.

Ein Aufruf mit voller Kontrolle:

```bash
curl -s -X POST localhost:8080/v1/images/generations \
  -H 'X-Auth-Token: <token>' -H 'Content-Type: application/json' \
  -d '{
        "prompt": "Ein roter Panda im Schnee, fotografisch, weiches Licht",
        "size": "16:9",
        "steps": 30,
        "seed": 1234,
        "negative_prompt": "unscharf, Wasserzeichen"
      }'
```

#### Antwort 200 — fertig

Die Anfrage war innerhalb von `IMAGEINT_SYNC_TIMEOUT` fertig.

```json
{
  "ok": true,
  "job_id": "8f3c1a2b4d5e6f70",
  "status": "done",
  "stage": "done",
  "prompt": "Ein roter Panda im Schnee",
  "created_at": 1738000000.1, "started_at": 1738000000.2, "finished_at": 1738000143.7,
  "width": 2400, "height": 1350, "wh_ratio": 1.7777778, "seed": 1234,
  "model": "Qwen/Qwen-Image-2.1",
  "parse_ok": true,
  "timings": {"enhance": 41.2, "render": 102.3},
  "image_bytes": 1843200,
  "enhanced_prompt": "A photorealistic red panda …",
  "status_url": "http://localhost:8080/v1/jobs/8f3c1a2b4d5e6f70",
  "image_url": "http://localhost:8080/v1/images/8f3c1a2b4d5e6f70",
  "b64_json": "iVBORw0KGgoAAAANSUhEUg…",
  "content_type": "image/png"
}
```

`enhanced_prompt` steht nur dann drin, wenn der Enhancer gelaufen ist.
`timings` trennt Enhancer und Render — die einzige verlässliche Antwort auf
„warum hat es so lange gedauert".

#### Antwort 202 — dauert länger

Auf einer CPU der Normalfall. Header `Retry-After: 15`.

```json
{
  "ok": true,
  "job_id": "8f3c1a2b4d5e6f70",
  "status": "running",
  "stage": "enhancing",
  "status_url": "http://localhost:8080/v1/jobs/8f3c1a2b4d5e6f70",
  "image_url": "http://localhost:8080/v1/images/8f3c1a2b4d5e6f70",
  "poll_after_seconds": 15,
  "message": "Das Bild wird noch erzeugt. Der Auftrag bleibt unter status_url abrufbar; der nächste Abruf sollte nach etwa 15 Sekunden erfolgen."
}
```

`stage` ist `enhancing` oder `rendering`. Der Enhancer ist auf einer CPU oft der
langsamere Teil, und „schreibt noch am Prompt" ist eine hilfreichere Auskunft
als „läuft noch".

`wait: false` erzwingt die 202-Antwort sofort. LLMInt nutzt das, um zu
entscheiden, ob es dem Nutzer die E-Mail-Benachrichtigung anbietet.

#### Antwort bei Fehlschlag

Scheitert der Auftrag **innerhalb** des Sync-Fensters, bekommt der Client den
echten Status statt einer 202 für etwas, das schon vorbei ist:

```json
{"ok": false, "error": "image_timeout",
 "message": "Das Bildmodell hat nicht rechtzeitig geantwortet."}
```

Ist ein Modellserver gar nicht erreichbar, wird die Anfrage **vor** dem
Einreihen abgelehnt — ein Client, dem „lädt, in 15 s erneut" gesagt wird, behält
seine Anfrage; einer, der zwei Minuten auf einen 502 wartet, hat die Zeit des
Nutzers verschwendet.

---

## Aufträge abfragen

### `GET /v1/jobs?limit=50`

Die letzten Aufträge, neueste zuerst.

```json
{"ok": true, "jobs": [{"job_id": "…", "status": "done", "…": "…"}],
 "stats": {"running": 0, "queued": 0, "done": 3, "failed": 0}}
```

### `GET /v1/jobs/{job_id}`

Der Zustand eines Auftrags — der Polling-Endpunkt. Der Body ist derselbe wie in
der 200-Antwort, aber **ohne** `b64_json`. Enthält `status` (`queued`,
`running`, `done`, `error`), `stage`, `timings` und bei Erfolg `image_url`.

Ein unbekannter oder abgelaufener Job (älter als
`IMAGEINT_JOB_RETENTION_SECONDS`) ergibt **404**:

```json
{"ok": false, "error": "not_found", "message": "Unbekannter Auftrag …"}
```

Das ist der Fall, den ein alter E-Mail-Link auslöst. Die Retention muss deshalb
„Nutzer liest die Mail und klickt" abdecken.

### `GET /v1/jobs/{job_id}/image` und `GET /v1/images/{job_id}`

Das fertige PNG, `Content-Type: image/png`. Beide Pfade liefern dasselbe;
`/v1/images/{id}` ist die URL, die der Dienst in `image_url` meldet und die in
der Benachrichtigungsmail steht.

```bash
curl -s localhost:8080/v1/images/8f3c1a2b4d5e6f70 -o panda.png
```

Der Fehlertext ist bewusst anders als bei einem Netzproblem: ein abgelaufener
Job ist **kein** Fehler des Aufrufers, und LLMInt zeigt dafür eine eigene
Meldung.

---

## Nur den Enhancer

### `POST /v1/enhance`

Führt ausschließlich den Prompt-Enhancer aus. Für das Admin-Formular in LLMInt
(„funktioniert der Enhancer?") und zum Debuggen von „ist es der Prompt oder das
Modell".

```bash
curl -s -X POST localhost:8080/v1/enhance \
  -H 'X-Auth-Token: <token>' -H 'Content-Type: application/json' \
  -d '{"prompt":"Ein roter Panda im Schnee"}'
```

```json
{
  "ok": true,
  "prompt": "Ein roter Panda im Schnee",
  "enhanced_prompt": "A photorealistic close-up of a red panda …",
  "wh_ratio": 1.7777778,
  "parse_ok": true,
  "width": 2752, "height": 1536,
  "timings": {"enhance": 38.4}
}
```

---

## Tabellen

### Leinwände

`wh_ratio` aus dem Enhancer wird auf eine der dokumentierten Leinwände
abgebildet; die dokumentierten Größen werden **unverändert** verwendet.

| Verhältnis | Breite | Höhe | Pixel |
|---|---|---|---|
| 1:1 | 2048 | 2048 | 4.194.304 |
| 16:9 | 2752 | 1536 | 4.227.072 |
| 9:16 | 1536 | 2752 | 4.227.072 |
| 4:3 | 2400 | 1792 | 4.300.800 |
| 3:4 | 1792 | 2400 | 4.300.800 |
| 3:2 | 2400 | 1696 | 4.070.400 |
| 2:3 | 1696 | 2400 | 4.070.400 |

Ein **nicht** dokumentiertes Verhältnis wird auf dieselbe Fläche wie das native
2K-Bild abgebildet, auf ein Vielfaches von 64 gerundet und auf
`IMAGEINT_IMAGE_MAX_PIXELS` begrenzt. Die angewandte Größe steht in den
Antwort-Headern `X-ImageInt-Width` / `X-ImageInt-Height` und im Job.

### Fehlercodes

| Code | HTTP | Bedeutung |
|---|---|---|
| `bad_request` | 400 | Anfrage nicht verwertbar (z. B. unlesbare Größe) |
| `empty_prompt` | 400 | Kein Prompt übergeben |
| `payload_too_large` | 413 | Prompt länger als `IMAGEINT_MAX_PROMPT_CHARS` |
| `unauthorized` | 401 | Token fehlt oder ist falsch |
| `not_found` | 404 | Unbekannter oder abgelaufener Auftrag |
| `not_configured` | 503 | Endpunkt nicht konfiguriert |
| `host_unsupported` | 503 | Hardware erfüllt die Anforderung nicht (AVX2) |
| `service_loading` | 503 | Modell lädt noch — **transient**, `Retry-After` beachten |
| `busy` | 503 | Alle Renderplätze belegt — **transient**, `Retry-After` beachten |
| `cancelled` | 503 | Auftrag abgebrochen |
| `enhancer_error` | 502 | Der Enhancer hat fehlerhaft geantwortet |
| `enhancer_unavailable` | 502 | Der Enhancer ist nicht erreichbar |
| `image_error` | 502 | Das Bildmodell hat fehlerhaft geantwortet |
| `image_unavailable` | 502 | Das Bildmodell ist nicht erreichbar |
| `image_empty` | 502 | Antwort ohne Bilddaten |
| `enhancer_timeout` | 504 | Enhancer hat `IMAGEINT_PE_TIMEOUT` überschritten |
| `image_timeout` | 504 | Bildmodell hat `IMAGEINT_IMAGE_TIMEOUT` überschritten |
| `storage_failed` | 500 | Bild konnte nicht geschrieben werden |

`service_loading`, `busy` und `cancelled` sind **transient**: der Client soll die
Anfrage behalten und nach `Retry-After` wiederholen. Alle anderen sind echte
Fehler.

---

## Der Bildserver

Der Container, der das Bildmodell ausführt (`image_server/`), hat eine eigene,
kleinere API. Er ist **nicht** veröffentlicht — nur der Gateway spricht mit ihm —
und implementiert die OpenAI Images API, damit der Gateway unverändert bleibt.

| Methode | Pfad | Zweck |
|---|---|---|
| `GET` | `/` | Banner mit Engine, Modell, dtype, device |
| `GET` | `/health` | 200 bereit / 503 lädt / 500 Fehler |
| `GET` | `/v1/models` | Modell-IDs |
| `POST` | `/v1/warmup` | Laden starten, ohne zu rendern (202) |
| `POST` | `/v1/images/generations` | rendern |

```bash
curl -s -X POST http://image:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen-Image-2.1","prompt":"Ein roter Panda",
       "size":"2400x1792","n":1,"num_inference_steps":40,"true_cfg_scale":1.0}'
```

```json
{
  "created": 1738000143,
  "model": "Qwen/Qwen-Image-2.1",
  "engine": "diffusers",
  "size": "2400x1792",
  "latency_ms": 102300,
  "data": [{"b64_json": "iVBORw0KGgo…", "width": 2400, "height": 1792,
            "model": "Qwen/Qwen-Image-2.1", "seed": 1234}]
}
```

Unterschiede zum Gateway, die man kennen muss:

- **Kein Job-Konzept.** Der Bildserver rendert synchron; die 202/Queue-Logik
  lebt im Gateway.
- **`size` wird auf ein Vielfaches von 32 gerundet** (16×16-DiT-Patchgitter und
  8×8-VAE-Gitter). `1000x1000` wird zu `992x992`, `2400x1792` bleibt unverändert.
  Die angewandte Größe steht in der Antwort.
- **`n > 1` wird abgelehnt**, weil das Modell einen Aufruf pro Bild braucht.
- **`negative_prompt` wird nur weitergegeben, wenn `true_cfg_scale > 1.0`.** Bei
  `1.0` (dem dokumentierten Standard) wird ohne Guidance gesampelt, und ein
  Negativ-Prompt hätte dann keine Wirkung.
- **Kein Streaming.** `stream` in der Anfrage wird ignoriert.

### `POST /v1/warmup`

Startet das Laden der Gewichte und antwortet sofort: `202`, wenn geladen wird,
`200`, wenn schon fertig. Idempotent — läuft bereits ein Ladevorgang, passiert
nichts.

Beide Modellserver implementieren die Route. Für das Bildmodell existiert sie für
genau eine Konfiguration: mit `IMAGEINT_IMAGE_PRELOAD=false` bleibt der Container
im Zustand `idle` und antwortet auf `/health` mit 503, bis ein Render eintrifft.
Ein Client, der erst auf 200 wartet und dann rendert, würde damit **ewig** warten.
Der Gateway ruft `/v1/warmup` deshalb als Nebenwirkung seiner
Bereitschaftsprüfung auf: er gibt weiterhin unverändert 503 mit `Retry-After`
zurück, stößt aber im Hintergrund das Laden an, sodass der dokumentierte
Wiederholungsversuch nach 15 Sekunden tatsächlich gelingt.

Für den Enhancer gilt dasselbe mit `IMAGEINT_PE_PRELOAD=false`; der Gateway
behandelt beide Komponenten gleich. Der Aufruf ist zustandslos und damit
idempotent — ein bereits ladender oder fertiger Server antwortet einfach mit
seinem aktuellen Zustand. Ein 404 wäre die folgenlose Antwort eines Servers, der
die Route nicht kennt.

```bash
curl -s -X POST http://image:8000/v1/warmup -H 'X-Auth-Token: <token>'
```

```json
{"ok": true, "state": "loading", "loading": true,
 "message": "Das Bildmodell wird geladen."}
```

Der Vertrag zwischen Gateway und Bildserver wird von
`image_server/tests/test_gateway_contract.py` geprüft: der Test schickt den
echten Request-Body des Gateways an die echte Bildserver-API und stellt sicher,
dass die Antwort als PNG zurückkommt.

---

## Der Enhancerserver

Der Container, der den Prompt-Enhancer ausführt (`enhancer_server/`), ist
ebenfalls **nicht** veröffentlicht und spricht die **OpenAI
Chat-Completions API** — nicht die Images API. Er läuft mit `transformers` statt
vLLM, weil der CPU-Build von vLLM dieses Checkpoint nicht initialisieren kann
(Begründung im README).

| Methode | Pfad | Zweck |
|---|---|---|
| `GET` | `/` | Banner mit Engine, Modell, dtype, device |
| `GET` | `/health` | 200 bereit / 503 lädt / 500 Fehler |
| `GET` | `/v1/models` | Modell-IDs |
| `POST` | `/v1/warmup` | Laden starten, ohne zu generieren (202) |
| `POST` | `/v1/chat/completions` | generieren, mit `stream: true` als SSE |

```bash
curl -sN -X POST http://enhancer:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen-Image-2.1-PE-T2I",
       "messages":[{"role":"system","content":[{"type":"text","text":"…"}]},
                   {"role":"user","content":[{"type":"text","text":"Ein roter Panda im Schnee"}]}],
       "temperature":1.0,"top_p":0.95,"top_k":20,"max_tokens":16256,"seed":42,
       "chat_template_kwargs":{"enable_thinking":true},
       "stream":true,"stream_options":{"include_usage":true}}'
```

Die SSE-Antwort ist die Form, die der Gateway schon liest:

```
data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}],"model":"Qwen/Qwen-Image-2.1-PE-T2I"}

data: {"choices":[{"index":0,"delta":{"reasoning_content":"Der Nutzer möchte …"}}]}

data: {"choices":[{"index":0,"delta":{"content":"{\"rewritten_prompt\": …}"}}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"choices":[],"usage":{"prompt_tokens":640,"completion_tokens":128,"total_tokens":768}}

data: [DONE]
```

Der `usage`-Frame kommt getrennt und mit leerem `choices`-Array — genau so,
wie ihn der Gateway beim Zählen erwartet.

Unterschiede, die man kennen muss:

- **Der Denkblock steht in `reasoning_content`, nicht in `content`.** Das
  entspricht `--reasoning-parser qwen3` des vLLM-Betriebs, sodass die
  Nachbearbeitung im Gateway unverändert bleibt. `<think>`-Marker tauchen in
  keinem der beiden Felder auf.
- **`stream: true` ist der Normalfall.** Ohne `stream` antwortet der Server mit
  einem einzelnen JSON-Objekt in der gewohnten Form; der Gateway nutzt diesen
  Pfad nicht.
- **`presence_penalty` ist kein natives Transformers-Sampling-Argument.** Der
  Wert wird herausgefiltert und einmalig geloggt. Der Gateway schickt die
  dokumentierten `1.5`; das Verhalten weicht damit von vLLM ab.
- **`max_tokens` ist eine Obergrenze, keine Vorgabe.** Ein Client kann um eine
  kürzere Antwort bitten, nie um eine längere als `IMAGEINT_PE_MAX_NEW_TOKENS`.
  Eine explizite `0` ist ein fehlerhafter Request (400), kein fehlendes Feld.
- **`enable_thinking` darf pro Request gesetzt werden** und überschreibt
  `IMAGEINT_PE_ENABLE_THINKING`.
- **Unbekannte Felder werden ignoriert**, damit ein Gateway mit einer neueren
  Feldliste nicht sofort scheitert.

Der Vertrag zwischen Gateway und Enhancerserver wird von
`enhancer_server/tests/test_gateway_contract.py` geprüft: der Test startet den
Server auf einem echten Socket und schickt ihm den echten Request-Body des
Gateways, bis hin zum geparsten JSON (`rewritten_prompt`, `negative_prompt`,
`wh_ratio`).
