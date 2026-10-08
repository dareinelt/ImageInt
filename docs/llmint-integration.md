# LLMInt-Integration

Wie ImageInt in LLMInt eingebunden wird: als Bild-Endpunkt, als Tool
`generate_image` für das jeweils laufende Text-LLM und mit der
E-Mail-Benachrichtigung für lange Generierungen.

> **Status.** Die ImageInt-Seite dieser Schnittstelle ist fertig und getestet
> (siehe [`api.md`](api.md)). Die LLMInt-Seite ist in diesem Dokument
> **spezifiziert**, aber noch nicht implementiert. Die Datei- und
> Zeilenangaben beziehen sich auf den Stand des LLMInt-Repositories zum
> Zeitpunkt der Erstellung und dienen als Anknüpfungspunkte.

---

## Inhalt

- [Ausgangslage](#ausgangslage)
- [Was in LLMInt entsteht](#was-in-llmint-entsteht)
- [1. Endpunkte speichern](#1-endpunkte-speichern)
- [2. Endpunkt auflösen](#2-endpunkt-auflösen)
- [3. Tool-Definition](#3-tool-definition)
- [4. Tool-Aufruf abwickeln](#4-tool-aufruf-abwickeln)
- [5. Erkennung durch das Text-LLM](#5-erkennung-durch-das-text-llm)
- [6. Zustimmung und E-Mail-Benachrichtigung](#6-zustimmung-und-e-mail-benachrichtigung)
- [7. Deep-Link in den Chat](#7-deep-link-in-den-chat)
- [8. ComfyUI und AUTOMATIC1111 entfernen](#8-comfyui-und-automatic1111-entfernen)
- [Abnahmekriterien](#abnahmekriterien)

---

## Ausgangslage

LLMInt kennt heute zwei Bild-Tools:

| Tool | Integration | Anknüpfungspunkt |
|---|---|---|
| `generate_image` | AUTOMATIC1111 | `createImageGenerationToolDefinition()` in `api/chat.php` |
| `generate_image_comfy` | ComfyUI | Tool-Definition bei `api/chat.php:1108-1146` |

Beide werden im Tool-Loop in `api/chat.php:3203-3239` abgewickelt und hängen
Zeilen in `sd_endpoints`, `sd_tasks`, `comfy_endpoints` und `comfy_tasks`
(`db.php:279-359`).

**ImageInt ersetzt beide.** Es erbt die Bezeichnung `generate_image`, damit
vorhandene Prompts, Tool-Definitionen und Nutzergewohnheiten unverändert
weiterlaufen. `generate_image_comfy` entfällt ersatzlos.

Der nächste Verwandte im Repository ist **SpeechInt**, weil es dieselbe Bauform
hat: ein eigenständiger Dienst hinter HTTP, dessen Basis-URL zuerst aus einer
Umgebungsvariablen und dann aus einer Endpunkt-Tabelle kommt. ImageInt folgt
diesem Muster (`lib/speech_dictation.php` ist die Vorlage), ergänzt aber zwei
Dinge, die SpeechInt nicht braucht: ein **Job-Polling** und eine
**Benachrichtigungsmail**, weil eine Bildgenerierung auf einer CPU Minuten statt
Sekunden dauert.

---

## Was in LLMInt entsteht

| Baustein | Datei (neu oder geändert) |
|---|---|
| `image_endpoints`-Tabelle + Migration | `db.php` |
| Einstellungen und Platzhalter-Vorlagen | `db.php`, `admin/index.php` |
| Endpunkt-Auflösung, HTTP-Aufruf, Job-Polling | `lib/image_generation.php` (neu) |
| Tool-Definition | `api/chat.php` |
| Tool-Aufruf im Tool-Loop | `api/chat.php` |
| Zustimmungsfrage | `api/chat.php` |
| Benachrichtigungs-Endpunkt (Mail auslösen) | `api/image_notify.php` (neu) |
| Zustandsabfrage für die Admin-Karte | `api/image_health.php` (neu) |
| Deep-Link-Unterstützung | `index.php` |

---

## 1. Endpunkte speichern

Eine eigene Tabelle, nach dem Vorbild von `speech_endpoints` (`db.php:336-359`)
— inklusive `token`-Spalte, die in der Admin-Oberfläche **nur maskiert**
angezeigt wird:

```sql
CREATE TABLE IF NOT EXISTS image_endpoints (
    id          INT          NOT NULL AUTO_INCREMENT,
    alias       VARCHAR(120) NOT NULL DEFAULT '',
    base_url    VARCHAR(500) NOT NULL,
    token       TEXT         NULL,
    timeout     INT          NOT NULL DEFAULT 1800,
    is_active   TINYINT(1)   NOT NULL DEFAULT 1,
    sort_order  INT          NOT NULL DEFAULT 0,
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
                             ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
```

`timeout` steht hier auf **1800** Sekunden, nicht auf den 120 von SpeechInt: eine
CPU-Generierung dauert Minuten. Der Wert ist die Obergrenze des **ganzen**
Vorgangs einschließlich Polling, nicht eines einzelnen HTTP-Aufrufs.

Die Migration läuft wie bei den bestehenden Tabellen idempotent in
`try/catch` (`db.php:361-372`), damit eine ältere Installation beim nächsten
Start nachgezogen wird.

**Aufgabe der Admin-Oberfläche** (`admin/index.php`, Vorbild: die Speech-Karte):

- Liste der Endpunkte mit Alias, Host, Timeout, aktiv/inaktiv
- Token-Feld, das nur `••••` oder die letzten vier Zeichen zeigt
- „Verbindung testen"-Knopf, der `GET /v1/ready` und `GET /v1/models` aufruft
  und den Ladezustand anzeigt (`loading` ≠ `error`!)

---

## 2. Endpunkt auflösen

Genau die Reihenfolge von `speechDictationUrl()` /
`speechDictationToken()` / `speechDictationTimeout()`
(`lib/speech_dictation.php:453-489`):

| Wert | Umgebungsvariable | Fallback |
|---|---|---|
| Basis-URL | `IMAGEINT_URL` | aktive Zeile in `image_endpoints` |
| Token | `IMAGEINT_TOKEN` | `image_endpoints.token` |
| Timeout | `IMAGEINT_TIMEOUT` | `image_endpoints.timeout`, dann 1800 s |

Dazu die Herkunftsauskunft (`image_generationUrlSource()` → `env` / `endpoint` /
`none`), weil die Admin-Karte sonst eine Zeile bearbeiten lässt, die wegen der
Umgebungsvariablen gar keine Wirkung hat.

In `docker-compose.yml` von LLMInt gehören die beiden Variablen in den
`app`-Dienst:

```yaml
environment:
  IMAGEINT_URL: http://imageint:8080
  IMAGEINT_TOKEN: ${IMAGEINT_TOKEN}
```

Das Token wird als `X-Auth-Token` gesendet — dieselbe Kopfzeile wie bei
SpeechInt.

---

## 3. Tool-Definition

`createImageGenerationToolDefinition()` (`api/chat.php:399-429`) wird inhaltlich
ersetzt. **Der Name bleibt `generate_image`**, die Parameterliste wird kleiner,
weil ImageInt das Seitenverhältnis selbst bestimmt:

```php
function createImageGenerationToolDefinition(): array
{
    return [[
        'type' => 'function',
        'function' => [
            'name' => 'generate_image',
            'description' => 'Erzeugt ein Bild mit Qwen-Image-2.1 aus einer '
                . 'deutschen oder englischen Beschreibung. Die Beschreibung darf '
                . 'und soll ausführlich sein; ein Prompt-Enhancer schreibt sie in '
                . 'den Prompt um, auf den das Modell trainiert wurde, und wählt '
                . 'auch das Seitenverhältnis. Die Generierung dauert auf einer CPU '
                . 'mehrere Minuten; das Ergebnis kommt als Bild-URL zurück.',
            'parameters' => [
                'type' => 'object',
                'properties' => [
                    'prompt' => [
                        'type' => 'string',
                        'description' => 'Beschreibung des gewünschten Bildes. '
                            . 'Je genauer (Motiv, Stil, Licht, Perspektive), desto besser.',
                    ],
                    'negative_prompt' => [
                        'type' => 'string',
                        'description' => 'Optional: Elemente, die im Bild vermieden '
                            . 'werden sollen.',
                    ],
                    'size' => [
                        'type' => 'string',
                        'description' => 'Optional: Seitenverhältnis, z. B. "16:9", '
                            . '"1:1" oder "4:3". Ohne Angabe bestimmt es der Enhancer.',
                    ],
                ],
                'required' => ['prompt'],
            ],
        ],
    ]];
}
```

Zwei bewusste Änderungen gegenüber der alten Definition:

- **`width`/`height` entfallen.** Sie waren an das 64er-Raster von
  AUTOMATIC1111 gebunden (512er-Standard). ImageInt arbeitet mit den
  dokumentierten Leinwänden der Modellkarte und rundet selbst auf ein Vielfaches
  von 32; ein Textmodell kann diese Entscheidung schlechter treffen als der
  Enhancer. `size` bleibt als Notausgang.
- **Der Prompt darf ausführlich sein.** Die alte Beschreibung verlangte einen
  „englischen Text-Prompt"; Qwen-Image-2.1 versteht Deutsch und der Enhancer
  übersetzt ohnehin. Ein künstlich verkürzter Prompt kostet nur Qualität.

---

## 4. Tool-Aufruf abwickeln

Im Tool-Loop (`api/chat.php:3203-3239`) wird der `generate_image`-Zweig durch
einen Aufruf der neuen Bibliothek ersetzt:

```php
} elseif ($toolName === 'generate_image' && $useImageTool) {
    $args = json_decode((string) ($toolCall['function']['arguments'] ?? '{}'), true);
    if (!is_array($args)) {
        $args = [];
    }
    $toolResult = imageIntGenerate($args, $sessionUserId, $sessionId);
    if (isset($toolResult['image_url'])) {
        $toolResult['markdown'] = '![Generiertes Bild](' . $toolResult['image_url'] . ')';
    }
}
```

`imageIntGenerate()` kapselt den Ablauf:

1. **Vorprüfung.** `GET {base}/v1/ready`. Antwortet der Dienst 503 mit
   `transient: true`, ist das **kein Fehler**: der Aufruf wird mit dem
   `Retry-After`-Wert erneut versucht (begrenzt), und der Nutzer sieht
   „Das Bildmodell lädt noch, einen Moment bitte" statt einer Fehlermeldung.
   Antwortet er 503 mit `host_unsupported`, ist das eine echte, dauerhafte
   Fehlkonfiguration und wird als solche gemeldet.

2. **Auftrag anlegen.** `POST {base}/v1/images/generations` mit
   `{"prompt": …, "negative_prompt": …, "size": …}`.
   - **200** → das Bild ist fertig (schneller oder GPU-Host). Der Ablauf endet
     sofort.
   - **202** → es gibt eine `job_id`, `status_url` und `image_url`. Weiter mit
     Schritt 3.
   - **4xx/5xx** → Fehler; `error` und `message` aus der Antwort werden
     protokolliert und dem Modell als Tool-Ergebnis übergeben, damit es dem
     Nutzer etwas Sinnvolles sagen kann.

3. **Polling.** `GET {status_url}` im Abstand von `poll_after_seconds`
   (Standard 15 s), solange `status` `queued` oder `running` ist. `stage` wird
   zu einer Fortschrittsmeldung verdichtet:
   - `enhancing` → „Der Bildwunsch wird ausgearbeitet …"
   - `rendering` → „Das Bild wird gezeichnet …"
   Das ist auf einer CPU die einzige ehrliche Rückmeldung, die man geben kann.
   Die Obergrenze ist der Endpunkt-`timeout`.

4. **Ergebnis.** Bei `status = done` liefert `image_url` das fertige PNG. Diese
   URL kommt als Tool-Ergebnis zurück, damit das Text-LLM sie wie bisher als
   Markdown-Bild referenzieren kann. Zusätzlich werden `width`, `height`, `seed`
   und `timings` übernommen — `timings` ist die Grundlage für „hat 2:41 gedauert"
   in der Statusmeldung.

**Warum der Aufruf nicht blockierend im Request bleibt.** `IMAGEINT_SYNC_TIMEOUT`
steht standardmäßig auf 120 Sekunden. Ein PHP-Request, der zwei Minuten auf ein
Modell wartet, läuft in jedes `max_execution_time` und hinter jedem Reverse Proxy
in einen Gateway-Timeout. Deshalb ist der Normalfall: ImageInt antwortet nach
kurzer Zeit mit **202**, PHP gibt die Kontrolle zurück und das Ergebnis wird
über `api/image_status.php` (neu) nachgeladen, genau wie bei
`api/document_status.php`.

---

## 5. Erkennung durch das Text-LLM

Das Modell soll **selbst** erkennen, wann `generate_image` nötig ist. Das
geschieht auf zwei Ebenen, ohne eigene Heuristik im PHP-Code:

1. **Tool-Beschreibung.** Sie sagt, was das Tool tut und dass es Minuten dauert.
   Modelle wählen Tools anhand dieser Beschreibung.

2. **System-Prompt.** Hier stehen die eindeutigen Aufforderungen, die einen
   Tool-Aufruf auslösen sollen:

   > Wenn der Nutzer ausdrücklich ein Bild, eine Grafik, eine Illustration, ein
   > Foto oder eine Visualisierung verlangt — erkennbar an Formulierungen wie
   > „Generiere ein Bild", „Erstelle eine Grafik", „Zeichne …", „Zeige mir ein
   > Bild von …", „Mach mir ein Bild von …" —, dann rufe **immer** das Tool
   > `generate_image` auf. Antworte in diesem Fall **nicht** mit einer
   > Textbeschreibung und **nicht** mit einem Platzhalterbild. Kündige den
   > Aufruf kurz an und weise darauf hin, dass die Generierung einige Zeit
   > dauern kann.

   Ebenso wichtig ist die Gegenrichtung:

   > Rufe `generate_image` **nicht** auf, wenn der Nutzer nur über Bilder
   > *spricht* („wie funktioniert Bildgenerierung?", „welches Modell nutzt du?")
   > oder wenn er ein vorhandenes Bild beschreiben lässt.

Die Liste der Auslöse-Formulierungen gehört in die Einstellungen, nicht in den
Code, damit sie ohne Deployment erweiterbar ist.

---

## 6. Zustimmung und E-Mail-Benachrichtigung

Weil die Generierung dauern kann, kann der Nutzer per Mail benachrichtigt
werden — **nur mit Zustimmung**. Beide Texte sind über Platzhalter voll
individualisierbar, mit demselben `{placeholder}`- + `str_replace()`-Muster wie
die Registrierungsmail (`db.php:1064-1135`,
`renderRegistrationEmailTemplate()`, Admin-Oberfläche `admin/index.php:5128-5148`).

### Zustimmungsfrage

Erscheint im Chat, sobald der Auftrag mit 202 zurückkommt:

> **Standardtext:**
> Das Generieren dieser Antwort kann einige Zeit in Anspruch nehmen. Möchten Sie
> per E-Mail über die Fertigstellung benachrichtigt werden?

Die Frage wird als eigener Chat-Block gerendert („Ja, per E-Mail benachrichtigen"
/ „Nein, ich warte hier"), nicht als Tool-Ergebnis — das Text-LLM soll sie nicht
formulieren und nicht beantworten. Die Antwort des Nutzers geht an
`api/image_notify.php`.

### Einstellungen

| Schlüssel | Standard |
|---|---|
| `image_notify_consent_text` | der Text oben, mit Platzhaltern |
| `image_notify_email_subject` | `Dein Bild ist fertig – {sitename}` |
| `image_notify_email_body` | der Text unten, mit Platzhaltern |
| `image_notify_enabled` | `1` |

**Standardtext der Mail:**

```
Hallo {username},

Dein Bild ist fertig.

Bildwunsch: {prompt}
Dauer: {duration}

Zum Bild im Chat: {chat_url}
Direkt zum Bild: {image_url}

Viele Grüße,
Dein {sitename}-Team
```

### Platzhalter

| Platzhalter | Inhalt |
|---|---|
| `{sitename}` | Name der Installation (`smtp_from_name`) |
| `{username}` | Anzeigename des Nutzers |
| `{email}` | E-Mail-Adresse des Nutzers |
| `{prompt}` | der ursprüngliche Bildwunsch |
| `{chat_url}` | Deep-Link in den Chat mit dem fertigen Bild |
| `{image_url}` | direkte URL des PNG |
| `{duration}` | Dauer der Generierung, z. B. „2:41 Minuten" |
| `{width}` / `{height}` | Maße des Bildes |
| `{job_id}` | die ImageInt-Job-ID |

### Ablauf

```
202 vom Gateway
      │
      ├─ Nutzer will keine Mail  →  Statusabfrage im Browser (api/image_status.php)
      │
      └─ Nutzer will eine Mail   →  users.email
                                    api/image_notify.php
                                      ├─ Eintrag in image_notifications
                                      │  (job_id, user_id, session_id, status=pending)
                                      ├─ sendMail() aus lib/mailer.php
                                      └─ Bestätigung im Chat:
                                         „Wir benachrichtigen Dich unter …"
```

Die Zustellung selbst braucht einen Auslöser, weil ImageInt nichts von LLMInt
weiß. Zwei gangbare Wege:

- **Cron / CLI.** `php api/image_notify_worker.php` in kurzem Intervall: holt
  offene `image_notifications`-Zeilen, fragt `status_url` ab und verschickt die
  Mail, sobald der Job `done` ist. Das ist der robuste Weg — er überlebt einen
  geschlossenen Browser.
- **Rein clientseitig.** `api/image_status.php` wird ohnehin gepollt, solange
  der Tab offen ist; beim Wechsel auf `done` wird die Mail ausgelöst. Einfacher,
  aber eine Mail kommt nur, wenn der Nutzer den Tab offen lässt — was den Zweck
  der Funktion verfehlt.

Empfohlen ist die Worker-Variante; sie braucht nur eine Tabelle:

```sql
CREATE TABLE IF NOT EXISTS image_notifications (
    id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    job_id        VARCHAR(64)  NOT NULL,
    user_id       INT          NOT NULL,
    session_id    CHAR(64)     NOT NULL,
    prompt        TEXT         NOT NULL,
    status_url    VARCHAR(500) NOT NULL,
    image_url     VARCHAR(500) NOT NULL,
    status        ENUM('pending','sent','failed','expired') NOT NULL DEFAULT 'pending',
    error         VARCHAR(500) NOT NULL DEFAULT '',
    created_at    TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    sent_at       TIMESTAMP(3) NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uniq_job_user (job_id, user_id),
    KEY idx_status (status, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
```

Wichtig: Der Job bleibt in ImageInt nur `IMAGEINT_JOB_RETENTION_SECONDS`
(Standard 24 h) abrufbar. Der Worker muss **innerhalb** dieses Fensters
zustellen. Läuft die Retention ab, bevor die Mail raus ist, ist der Link in der
Mail wertlos — ein `404 not_found` ist dann die ehrliche Antwort, und die
Fehlermeldung im Chat sollte das sagen.

---

## 7. Deep-Link in den Chat

Die Mail verlinkt in den Chat **mit** dem fertigen Bild. Der Link muss also die
Chat-Sitzung und den Auftrag tragen:

```
{app_base}/index.php?session=<session_id>&job=<job_id>
```

`appPublicBaseUrl()` (`lib/reverse_proxy.php:278-301`) liefert den Präfix,
`session_id` kommt aus `conversation_sessions` bzw. aus dem Chat-Kontext.

**Was in `index.php` dafür fehlt.** Die Sitzungs-ID liegt heute ausschließlich in
`sessionStorage` (`index.php:2555-2560`). Ein Link von außen kann sie deshalb
nicht setzen. Es braucht eine kleine Ergänzung beim Start:

```js
// Sitzung aus der URL übernehmen (Deep-Link aus der Benachrichtigungsmail).
const params = new URLSearchParams(window.location.search);
const deepLinkSession = params.get('session') || '';
const deepLinkJob     = params.get('job') || '';
if (deepLinkSession) {
    sessionId = deepLinkSession;
    sessionStorage.setItem('chat_session_id', sessionId);
}
// … nach dem Laden der Sitzung:
if (deepLinkJob) {
    scrollToJobResult(deepLinkJob);   // scrollt zur Bildnachricht und hebt sie hervor
}
```

Zu beachten:

- **Zugriffsprüfung.** Die Sitzung aus der URL muss gegen `user_id` geprüft
  werden (`api/chat_sessions.php?action=load` tut das bereits). Ein fremder Link
  darf keine fremde Sitzung öffnen.
- **Token wegwerfen.** `session` und `job` nach dem Verarbeiten per
  `history.replaceState()` aus der Adresse entfernen, damit der Link nicht
  weitergeteilt wird.
- **Fallback.** Ist der Job abgelaufen, lädt die Sitzung trotzdem — der Nutzer
  landet im richtigen Chat und bekommt einen Hinweis statt einer leeren Seite.
  Der Chat-Verlauf selbst muss den Job-Bezug speichern, damit das Bild auch nach
  dem Ablaufen der Job-Retention noch sichtbar ist.

---

## 8. ComfyUI und AUTOMATIC1111 entfernen

Mit ImageInt entfallen beide Integrationen. Zu entfernen:

| Was | Wo |
|---|---|
| Tool `generate_image_comfy` | Definition `api/chat.php:1108-1146` |
| ComfyUI-Dispatch | `api/chat.php:3224-3234` |
| `callComfyGenerate()` | `api/chat.php` |
| `callSdGenerate()` | `api/chat.php` |
| `hasSdEndpoints()` / `hasComfyEndpoints()` | `api/chat.php:432-445` |
| `sd_generate.php`, `comfy_generate.php`, `sd_health.php`, `comfy_health.php` | `api/` |
| Admin-Karten „Stable Diffusion" und „ComfyUI" | `admin/index.php` |
| Tabellen `sd_endpoints`, `sd_tasks`, `comfy_endpoints`, `comfy_tasks` | `db.php:279-359` |

Bei den Tabellen ist **Vorsicht** geboten: ein `DROP TABLE` in `db.php` würde
bei jedem Start laufen und vorhandene Daten endgültig löschen. Der bestehende
Migrationsstil kennt nur `CREATE TABLE IF NOT EXISTS` und
`ALTER TABLE … ADD COLUMN` in `try/catch`. Eine Entfernung gehört deshalb in eine
bewusste, einmalige Migration oder — besser — die Tabellen bleiben liegen und
werden nur nicht mehr angesprochen.

Die Einstellungen, die an die alten Integrationen gebunden waren
(`sd_*`, `comfy_*`), werden in der Admin-Oberfläche ausgeblendet, aber nicht
gelöscht.

---

## Abnahmekriterien

- [ ] Eine Frage wie „Zeige mir ein Bild von einem roten Panda im Schnee" löst
      ohne weitere Hinweise einen `generate_image`-Aufruf aus.
- [ ] Eine Frage *über* Bilder („Wie funktioniert Bildgenerierung?") löst
      **keinen** Aufruf aus.
- [ ] Der Aufruf liefert ein Bild, das im Chat als Markdown-Bild erscheint.
- [ ] Bei einem 202 bekommt der Nutzer die Zustimmungsfrage; „Nein" erzeugt
      keine Mail, „Ja" genau eine.
- [ ] Die Mail enthält einen funktionierenden Deep-Link, der im Chat beim
      fertigen Bild landet.
- [ ] Ein individualisierter Text in `image_notify_consent_text` und
      `image_notify_email_body` wird mit aufgelösten Platzhaltern ausgegeben.
- [ ] Während der Dienst lädt (503, `transient: true`), erscheint kein Fehler,
      sondern eine Wartemeldung, und der Auftrag wird später fortgesetzt.
- [ ] Ein abgelaufener Job führt zu einer verständlichen Meldung, nicht zu einer
      leeren Seite.
- [ ] `generate_image_comfy` ist nicht mehr im Tool-Angebot.
