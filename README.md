# Multilingual Voice AI Agent (Pipecat + Sarvam AI)

A real-time voice assistant that listens and replies in an Indian language of
your choice. Speech recognition, speech synthesis and (by default) the language
model all come from [Sarvam AI](https://docs.sarvam.ai); the real-time pipeline
is [Pipecat](https://docs.pipecat.ai) 1.12. Phase 1 talks to the agent from a
browser over WebRTC. Telephony via Plivo is a later phase and is **not**
implemented yet (see [Next steps](#next-steps-plivo)).

## How it works

```
Browser mic ──WebRTC (Opus)──▶ SmallWebRTCTransport.input()   (resampled to 16 kHz PCM)
                                        │
                                        ▼
                               SarvamSTTService  (websocket, saaras:v4)
                                        │  TranscriptionFrame
                                        ▼
                               LLMUserAggregator (Silero VAD + Smart Turn v3, barge-in)
                                        │  LLMContextFrame
                                        ▼
                               LLM  (SarvamLLMService, or any OpenAI-compatible API)
                                        │  streamed text
                                        ▼
                               SarvamTTSService  (websocket, bulbul:v3, 24 kHz)
                                        │  TTSAudioRawFrame
                                        ▼
Browser speaker ◀──WebRTC──── SmallWebRTCTransport.output()
                                        │
                                        ▼
                               LLMAssistantAggregator (keeps what was actually spoken)
```

Pipecat's development runner (`pipecat.runner.run`) provides the FastAPI
server, the `/api/offer` WebRTC signalling endpoint and a prebuilt web client at
`/client`. One `PipelineWorker` runs per connected browser; the session ends on
disconnect, after `SESSION_IDLE_TIMEOUT_SECS` of silence, or when a Sarvam
service becomes unusable (for example an invalid API key).

## Project structure

Two top-level folders: `backend/` holds all the Python (the voice agent, the
Plivo integration, the portal API, scripts and tests); `frontend/` holds the
React portal UI. Paths below are relative to `backend/` unless they start with
`frontend/`.

| Path | Responsibility |
| --- | --- |
| `bot.py` | Entrypoint for the Pipecat runner: validates config, builds a transport per client, runs a session |
| `voice_agent/config.py` | `Settings` (pydantic-settings) loaded from env / `.env`, with validation and readable errors |
| `voice_agent/languages.py` | The 11 languages supported by both Sarvam STT and TTS, mapped to Pipecat `Language` values |
| `voice_agent/prompts.py` | System prompt (speaking style, language lock, greeting) |
| `voice_agent/services.py` | Factories for `SarvamSTTService`, `SarvamTTSService` and the LLM service |
| `voice_agent/pipeline.py` | Builds the Pipecat pipeline and `PipelineWorker`, wires connect/disconnect handlers |
| `voice_agent/session.py` | Runs one session with `WorkerRunner` and logs its lifecycle |
| `voice_agent/logging_config.py` | Loguru setup with API-key redaction |
| `voice_agent/transports/` | Transport registry and audio rates; `webrtc.py` (browser) and `plivo.py` (telephony) |
| `scripts/webrtc_probe.py` | Headless WebRTC client: connects like the browser, records the bot's audio, prints latency figures |
| `scripts/plivo_call.py` | Places an outbound Plivo call that streams to the running bot |
| `voice_agent/tracking/` | Postgres call tracking: queue-backed recorder, latency/transcript tracker, config store |
| `voice_agent/knowledge/` | Document knowledge base: text extraction, chunking, OpenAI embeddings, Qdrant store, ingestion service |
| `portal/app.py`, `portal.py` | Portal backend (separate process): JSON API for stats, calls, turns and config versions; serves the built UI |
| `frontend/` | Portal UI: React 19 + Vite + TypeScript, using the Cube product's design tokens and stylesheet |
| `tests/` | Unit tests (no network) and `tests/integration/` (real Sarvam calls, opt-in) |

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- A Sarvam AI API key (https://dashboard.sarvam.ai)
- Chrome/Firefox/Safari with microphone access

## Setup

```bash
cd backend
uv sync                      # creates backend/.venv from uv.lock (pinned versions)
cp .env.example .env         # then edit .env and set SARVAM_API_KEY
```

All `uv run ...` commands below are run from `backend/`.

Key settings (all documented in `.env.example`):

| Variable | Default | Notes |
| --- | --- | --- |
| `SARVAM_API_KEY` | required | Used by STT, TTS and the Sarvam LLM |
| `AGENT_LANGUAGE` | `hi-IN` | `hi-IN en-IN bn-IN gu-IN kn-IN ml-IN mr-IN od-IN pa-IN ta-IN te-IN` |
| `STT_AUTO_DETECT_LANGUAGE` | `false` | `true` lets Sarvam detect the spoken language; replies still use `AGENT_LANGUAGE` |
| `SARVAM_STT_MODEL` / `SARVAM_STT_MODE` | `saaras:v4` / `transcribe` | Also `saaras:v3`; modes `transcribe translate verbatim translit codemix` |
| `SARVAM_TTS_MODEL` / `SARVAM_TTS_SPEAKER` | `bulbul:v3` / model default (`shubh`) | Speaker names are validated against the model at startup |
| `LLM_PROVIDER` / `LLM_MODEL` | `sarvam` / `sarvam-105b` | `openai` works with any OpenAI-compatible API via `OPENAI_API_KEY` + `OPENAI_BASE_URL` |
| `SESSION_IDLE_TIMEOUT_SECS` | `300` | Session is closed after this much inactivity |

Invalid values fail fast at startup with a message naming the field, e.g.
`Invalid configuration: agent_language: Unsupported language 'fr-FR'. Supported languages: hi-IN, ...`.

## Run the web agent

```bash
uv run python bot.py                     # http://localhost:7860  (redirects to /client)
uv run python bot.py --host 0.0.0.0 --port 8000   # expose on the LAN
```

Open the URL, allow microphone access and press **Connect**. The agent greets
you in the configured language; speak naturally, and interrupt it whenever you
like. The client shows connection state and transcripts, and server-side errors
are logged with the API key redacted.

Browsers only allow microphone access on `localhost` or HTTPS. For remote
testing put the server behind an HTTPS reverse proxy or an `ngrok http 7860`
tunnel and pass `--proxy <public-host>`.

**Security note.** The runner's `/api/offer` endpoint is unauthenticated and
CORS allows every origin by default. Anyone who can reach the port can start a
session and spend your Sarvam credits, so never use `--host 0.0.0.0` on a
network you do not control without `--allowed-origins` (or
`PIPECAT_ALLOWED_ORIGINS`) and a firewall or tunnel in front of it.

### Test call from the portal

The portal's **Test call** page places the same WebRTC call from inside the
portal, in the portal's design: Start call, mute, hang up, a live transcript
(what the agent heard and said, as it happens), and, after hang-up, a link to
the tracked call record with its latency, searches and cost. It uses the
official browser SDK (`@pipecat-ai/client-js` with the SmallWebRTC transport)
against the bot's `/start` and `/api/offer` endpoints. `BOT_PUBLIC_URL` in
`backend/.env` says where the browser finds the bot (default
`http://localhost:7860`; `https://callagent.wizzgeeks.com` in
`.env.docker`), and the page's header shows whether the bot answers.

Leave `BOT_TRANSPORT` unset so one bot process serves phone calls on `/ws`
and browser calls over WebRTC at the same time; `BOT_TRANSPORT=plivo`
switches WebRTC (and this page) off. Browser audio is peer-to-peer UDP to the
server's public IP, so a deployed server needs its UDP ports open or a TURN
server; Cloudflare's HTTP proxy does not carry it.

### Browser calls from outside the server: TURN relay

Browser audio is peer-to-peer UDP. The bot container is on a Docker bridge
network and picks random UDP ports, so a browser outside the server cannot
reach it directly: the signalling over HTTPS succeeds, the call shows
"connecting", and no audio flows. `compose.turn.yaml` runs a coturn relay with
fixed ports that both sides use:

```bash
# backend/.env: TURN_PUBLIC_IP=<server IPv4>  TURN_USER=voice  TURN_PASSWORD=<openssl rand -hex 16>
docker compose -f compose.turn.yaml up -d   # reads them through the ./.env symlink
# firewall (ufw and the cloud provider's rules): 3478/tcp, 3478/udp, 49160-49200/udp
# backend/.env.docker: PIPECAT_ICE_SERVERS=[{"urls":"turn:<ip>:3478?transport=udp","username":"voice","credential":"<password>"},{"urls":"stun:<ip>:3478"}]
docker compose up -d                    # the bot reads the ICE servers at start-up
```

The runner hands the same servers to the browser in its `/start` response, so
the Test call page needs no change. About 20 simultaneous browser calls fit in
the relay range; widen it for more. The TURN password is visible to anyone
who can call `/start`, so rotate it if the bot host is ever exposed without
`PIPECAT_ALLOWED_ORIGINS`.

### Headless check without a browser

```bash
uv run python scripts/webrtc_probe.py --duration 20 --out greeting.wav
```

This performs the same WebRTC handshake as the browser, records what the agent
says (the greeting, with a valid key) and prints time-to-connect and
time-to-first-speech. Pass `--wav question_16k.wav` to feed a recorded question.

## Tests

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q                                   # unit tests, no network
SARVAM_API_KEY=... uv run pytest -m integration -q # real Sarvam TTS, STT and LLM calls
```

## Verification status (2026-10-18)

What has been run in this environment:

| Check | Result |
| --- | --- |
| Lint (`ruff check`, `ruff format --check`) | pass |
| Unit tests (config, languages, prompts, service factories, pipeline wiring, session, log redaction) | 42 passed |
| Startup without configuration, with an unknown TTS speaker, with a bad prompt placeholder | exits with code 1 and a message naming the setting |
| Server start, `/` redirect, `/client` UI, `/api/offer` validation | pass |
| Headless WebRTC session (`scripts/webrtc_probe.py`) | connected in 0.6 s, bot audio track received |
| Session cleanup on Sarvam failure (invalid key: HTTP 403 from STT, TTS and LLM) | pipeline ended itself, peer connection discarded, no dangling tasks |
| API key redaction and `LOG_LEVEL` after the runner's own logging reset | key absent from logs, no DEBUG lines at `LOG_LEVEL=INFO` |
| Integration tests against Sarvam (`-m integration`) | **not run: no `SARVAM_API_KEY` available here** (3 tests skip) |
| A real spoken conversation in a browser | **not yet verified** for the same reason |

So the web phase is **not** signed off yet: everything up to the Sarvam
websockets is verified, and the first run with a real key should be the
integration tests followed by a browser conversation.

## Code review

An independent review pass (against the installed Pipecat source) found and
led to these fixes: tracebacks no longer dump local variables such as the API
key; logging is re-applied after the runner resets it; settings are loaded once
after the runner's own `.env` load; `WorkerRunner.add_workers` replaces the
deprecated `run(worker)`; dead STT connection-event handlers were removed;
OpenAI-compatible endpoints get the `developer` role rewritten; models,
speakers, pace and prompt placeholders are validated at startup instead of per
connection; a failed session closes the WebRTC peer instead of leaving the
browser on "connecting".

## Known limitations

- The agent speaks only `AGENT_LANGUAGE`. With `STT_AUTO_DETECT_LANGUAGE=true`
  it understands other Indian languages, but TTS language is fixed per session.
- Pipecat 1.12 only accepts `bulbul:v2`/`v3` for Sarvam TTS; `bulbul:v4-flash`
  needs a newer Pipecat release.
- `SarvamSTTService` uses Sarvam's streaming endpoint that Sarvam now labels
  "legacy"; Pipecat also ships `SarvamRealtimeSTTService` for the newer realtime
  endpoint. Switching is a one-line change in `voice_agent/services.py` once both
  have been compared with real audio.
- End-of-turn detection uses Pipecat's bundled Smart Turn v3 model, which was
  trained mostly on English. If Indic turns end too early or late, switch the
  stop strategy to `SpeechTimeoutUserTurnStopStrategy` or Sarvam's server-side
  `vad_signals=True`.
- The development runner binds to `localhost` and allows all origins; it is for
  local testing, not public deployment.

## Portal: prompt, temperature and call tracking

Set `DATABASE_URL` (a local Postgres database; the schema is created on first
use), build the UI once, and start the portal next to the bot:

```bash
(cd ../frontend && npm install && npm run build)   # Node 20.19+ or 22 recommended
uv run python portal.py                            # http://localhost:8090, serves ../frontend/dist
```

For UI work, `npm run dev` inside `frontend/` serves the app on port 5174
with hot reload and proxies `/api` to the portal backend on 8090.

The UI follows the Cube product's design system: its `tokens.css` and
`base.css` are copied verbatim into `frontend/src/styles/`, as are the
shell components (rail, topbar, stat tile, bar list, toast, icons). Only
`portal.css` is console-specific. Pages: **Dashboard** (calls today, average and
p95 latency, turns, barge-ins, recent calls, latency distribution, channels),
**Calls** (searchable, filter by web/phone), **Call** (transcript with a
per-turn latency breakdown bar: end-of-turn wait, STT, LLM, TTS, other) and
**Prompt & temperature** (versioned, with history and restore).

- **Agent config** lets you edit the system prompt and temperature. Every save
  is a new version; the next session picks it up (calls in progress keep
  theirs). Empty fields mean "use the environment default".
- **Calls** lists every web and phone session with duration, turn count and
  average/max response latency. A call page shows the conversation turn by turn
  with: latency (user stopped speaking to bot audio start; turn 0 is connect to
  greeting), turn release (VAD/end-of-turn wait), and STT/LLM/TTS time to first
  byte, plus whether the user barged in.

Performance isolation: the bot only *enqueues* records; one background task per
session writes them, dropping records with a warning if the queue ever fills
instead of slowing the conversation. The single read at session start (the
active config) is capped at one second with env fallback. The portal runs as
its own process with its own connection pool. If Postgres is unreachable,
tracking is disabled for that process and the agent keeps running.

## Deploy with Docker Compose

Postgres, Qdrant and the `dev-nginx` reverse proxy already run on the server;
the compose file only adds the bot and the portal, from one shared image, on
the same `shared-app-network` as nginx. Nothing is published on host ports:
nginx reaches the containers by name.

```bash
cp backend/.env.docker.example backend/.env.docker   # container-side hostnames (Postgres by name, Qdrant via host gateway)
docker compose build                                  # multi-stage: node builds the UI, uv installs locked deps, python:slim runtime
docker compose up -d
docker compose logs -f bot                            # transcripts, latency and knowledge-search steps
```

nginx server blocks (in `newserver/sites/local-apps.conf`, TLS at Cloudflare):

```nginx
server {
    listen 80;
    server_name callagent.wizzgeeks.com;
    location / {
        proxy_pass http://voice-agent-bot:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
}
server {
    listen 80;
    server_name agentportal.wizzgeeks.com;
    location / {
        proxy_pass http://voice-agent-portal:8090;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        # the portal has no login: protect it with Cloudflare Access or basic auth
    }
}
```

- `backend/.env` keeps the secrets (shared with local runs); `backend/.env.docker`
  overrides only `DATABASE_URL`, `QDRANT_URL` and `PORTAL_HOST` for the container
  network. Both are git-ignored and never baked into the image.
- `PUBLIC_HOST` (in `backend/.env.docker`) is the hostname Plivo reaches the
  bot through (`wss://<host>/ws`) and `BOT_TRANSPORT=plivo` starts the runner
  in phone mode; the bot turns them into the runner's `--proxy` and `-t` flags
  when those are not given on the command line. Set the Plivo application's
  Answer URL to `https://<PUBLIC_HOST>/`. `BOT_ARGS` passes extra runner flags
  such as `--allowed-origins`.
- Phone calls (websocket through nginx) work in this bridge-network setup.
  The bot's built-in WebRTC test page does not, because browsers cannot reach
  container IPs for media; use the web client against a local `uv run python bot.py`
  for development instead.
- The image is built from the lock file with no dev dependencies, runs as a
  non-root user, and strips bundled test suites. Health checks hit `/status`
  and `/api/stats`; log rotation is on.

## Langfuse: cost tracking and evaluation (optional)

`compose.langfuse.yaml` runs **Langfuse v2** as a single container,
`langfuse-web`, on `shared-app-network`, listening on port **7310** (the next
free port in the nginx site list), storing its data in a `langfuse` database on
the server's existing Postgres. No ClickHouse, Redis or object store is needed
for v2, and no host port is published: nginx serves it at
**https://langfuse.wizzgeeks.com**. One production instance is shared by the
deployed agent and by developers' laptops, so every call lands in the same
project.

### Deploy Langfuse on the server (once)

```bash
# 1. DNS: in Cloudflare add langfuse.wizzgeeks.com as a proxied A record to the server,
#    like the other *.wizzgeeks.com names.

# 2. Database on the existing Postgres container
docker exec -it oasys-postgres psql -U admin -d postgres -c "CREATE DATABASE langfuse OWNER voice_agent;"

# 3. Settings (in the repo checkout on the server)
cp langfuse.env.example langfuse.env     # set DATABASE_URL (oasys-postgres, user voice_agent),
                                          # SALT and NEXTAUTH_SECRET (openssl rand -hex 32),
                                          # NEXTAUTH_URL=https://langfuse.wizzgeeks.com, the INIT_* login
# backend/.env must hold the project key pair the bootstrap creates:
#   LANGFUSE_PUBLIC_KEY=pk-lf-<uuid>   LANGFUSE_SECRET_KEY=sk-lf-<uuid>   (uuidgen for each)
ln -sf backend/.env .env                  # compose reads the two keys from here

# 4. Start it and check health from inside the network
docker compose -f compose.langfuse.yaml up -d
docker exec langfuse-web wget -qO- http://127.0.0.1:7310/api/public/health

# 5. nginx: pull the newserver repo (block for langfuse.wizzgeeks.com -> langfuse-web:7310) and reload
cd ~/workspace/newserver && git pull && docker exec dev-nginx nginx -t && docker exec dev-nginx nginx -s reload

# 6. Prices, so traces show cost (idempotent; run from any machine with the keys)
cd backend && uv run python scripts/langfuse_models.py --host https://langfuse.wizzgeeks.com
```

Log in at https://langfuse.wizzgeeks.com with the `LANGFUSE_INIT_USER_*`
credentials from `langfuse.env`; the project "Voice Agent" already exists with
the key pair from `backend/.env`.

### Point the agent at it

Both environments use the same project keys; only the host differs:

| Where the agent runs | `LANGFUSE_HOST` | File |
|---|---|---|
| deployed container | `http://langfuse-web:7310` (same docker network) | `backend/.env.docker` |
| a laptop | `https://langfuse.wizzgeeks.com` | `backend/.env` |

```
LANGFUSE_ENABLED=true                         # off by default; true turns tracing on
LANGFUSE_PUBLIC_KEY=pk-lf-...                 # one key pair: used by the agent AND by Langfuse's first-start bootstrap
LANGFUSE_SECRET_KEY=sk-lf-...
```

Processes read `.env` at start-up, so restart the bot (and `docker compose up -d`
on the server) after changing these. To run a private Langfuse on a laptop
instead, add the overlay that publishes port 3100:
`docker compose -f compose.langfuse.yaml -f compose.langfuse.local.yaml up -d`
and use `LANGFUSE_HOST=http://localhost:3100`.

With the switch on and all three set, the call tracker sends every call as one Langfuse trace
(id = the call id in the portal) containing one span per turn with the user
and assistant text, the latency breakdown, any knowledge searches, and a
generation per model call carrying usage: LLM prompt/completion tokens,
TTS characters, STT audio seconds. Events are batched on a background thread,
so the audio pipeline never waits on Langfuse.

- **Cost**: Langfuse prices usage only for models it has a definition for.
  `scripts/langfuse_models.py` adds them (or Settings → Models) with Sarvam's list prices, which are in INR, so read every cost Langfuse
  shows as rupees even though the UI prints a dollar sign:
  `sarvam-105b` ₹15 / 1M input tokens and ₹60 / 1M output (unit TOKENS,
  input 0.000015, output 0.00006); `bulbul:v3` ₹30 / 10k characters (unit
  CHARACTERS, output 0.003); `saaras:v4` ₹30 / hour (unit SECONDS, input
  0.008333). Costs are computed when a trace is ingested, so definitions
  apply to calls made after they exist, not retroactively.
- **Usage accuracy**: token, character and second counts come from Pipecat's
  own usage metrics (the provider's reported token usage, the text sent to
  TTS, the audio sent to STT), counted once per metrics frame. They match
  Sarvam's billing units; the remaining gaps are Sarvam's cheaper cached-input
  token rate (not tracked) and per-second rounding of STT audio. Usage recorded
  before 2026-10-21 was over-counted by a constant factor per service (TTS ×4,
  LLM ×4, STT ×5-6) due to frames being counted at every pipeline hop.
- **Evaluation**: score traces manually or via the API, build datasets of
  test questions with expected answers, and compare runs. Automated
  LLM-as-a-judge evaluators are a Langfuse v3 feature and are not available
  in v2.

## Knowledge base (documents in Qdrant)

The portal's **Knowledge** page uploads documents, extracts their text, indexes
each document as **one chunk** (`KNOWLEDGE_WHOLE_DOCUMENT=true`; an FAQ sheet
is a single topic and its steps only make sense together; documents over ~24k
characters fall back to ~1,200-character chunks with overlap), embeds the
chunks with OpenAI's multilingual `text-embedding-3-small`, and stores them in
a dedicated Qdrant collection (`voice_agent_knowledge` by default, one
collection for all documents, filterable by document). A "Test retrieval" box
runs a sample question in any language and shows the best-matching chunks with
their scores and the embed/search times. Documents are listed in Postgres
(`documents` table) with chunk counts and status.

Requirements: a local Qdrant (`QDRANT_URL`, default `http://localhost:6333`)
and `OPENAI_API_KEY` for the embeddings. Supported files: PDF, DOCX, TXT, MD,
CSV (up to 25 MB). Ingestion runs in the portal process.

**Questions in the conversation language** (`AGENT_LANGUAGE`, e.g. Tamil) are
translated to English with Sarvam's translate API before embedding, since the
documents are English. This happens both in the portal's test box, which shows
the translated query, and on calls; text that is already English is embedded
directly. Measured:
a Tamil question scored 0.17 against the right chunk untranslated and 0.5+
translated; translation takes 0.6 – 1 s.

**On calls**, when `KNOWLEDGE_ENABLED=true` (default) and the embedding key is
set, the LLM gets a `search_knowledge_base` tool that returns the best
`KNOWLEDGE_TOP_K` chunk(s) (default 1: the single most relevant sheet, whole) and a prompt rule to use it for
procedure, document, status and complaint questions and to answer only from
what it returns. The moment the model decides to search, the agent speaks a
short line in the conversation language ("ஒரு நிமிடம், நான் பார்க்கிறேன்.")
so the caller is not left in silence during translation, embedding and
search; the filler is not added to the context. Small talk does not trigger a
search. The function call shows up in each turn's latency breakdown, and every
search is stored with the turn (question, English query, timings, each
retrieved chunk with its score) and shown under the exchange on the call page.
The terminal logs each step at INFO: the question, translation, embedding and
search times, every chunk with its score and whether it was used, and what was
passed to the LLM.
`backend/scripts/xlsx_to_pdfs.py` turns a FAQ workbook into one PDF per sheet
for uploading.

## Idle callers

After the agent finishes speaking, if the caller stays silent for
`IDLE_PROMPT_SECS` (default 10) the agent asks once whether they are still
there, in the conversation language ("நீங்க லைன்ல இருக்கீங்களா?"). If the
silence continues for the same period, it says a short goodbye and ends the
call: the pipeline shuts down after the goodbye finishes playing, and on
Plivo the call is hung up (needs `PLIVO_AUTH_ID` / `PLIVO_AUTH_TOKEN`). Any
speech from the caller resets the counter. The call is stored with
`end_reason = idle_timeout`. Set `IDLE_PROMPT_SECS=0` to disable, and
`IDLE_CHECK_LINE` / `IDLE_GOODBYE_LINE` to change the wording.

## Models per step (Sarvam)

The **Models** page of the portal picks the Sarvam model for
each pipeline step: speech to text, language model, text to speech plus voice,
and the translation model used for knowledge-base queries. The lists come from
`voice_agent/models.py`, a catalogue checked against Sarvam's API on
2026-10-21, and `GET /api/models?language=` returns only the entries that
support the selected language, so a Tamil agent is never offered a model or
voice that cannot speak Tamil (bulbul:v2, for example, rejects it).

| Step | Offered | Notes |
|---|---|---|
| speech to text | `saaras:v4` (recommended), `saaras:v3` | the two models that work over the streaming websocket |
| language model | `sarvam-105b` | Sarvam's own model only, by choice. Sarvam's API also serves GLM, Gemma and DeepSeek; they passed the Tamil and tool-call probes and can be added to the catalogue if wanted |
| text to speech | `bulbul:v3` (recommended) with its 37 voices; `bulbul:v4-flash` with 207 persona voices (Sarvam lists 209; the two Assamese ones are left out because the agent does not support as-IN) | v3 voices speak every language and Sarvam's per-language picks are listed first with its quality tier (1 best). v4-flash voices are recorded for one language and one style (`aravind_ta_ads`), so only the voices for the selected language are offered, and the model is hidden for Malayalam and Odia, which have none. The voice list comes from Sarvam's API (`voice_agent/data/bulbul_v4_flash_voices.json`). Pipecat 1.12 does not know v4-flash, so `services.py` registers it; it was checked over the websocket with a pronunciation dictionary on 2026-10-22 |
| translation | `sarvam-translate:v1` (recommended), `mayura:v1` | Tamil to English before searching |
| embedding (OpenAI) | `text-embedding-3-small` (recommended), `text-embedding-3-large`, `text-embedding-ada-002` | embeds documents and questions. Changing it re-embeds every document on save: the portal keeps each document's extracted text for this. Documents uploaded before that (before 2026-10-21) are marked `needs-reupload` and must be uploaded again |

Choices equal to the environment default (`SARVAM_STT_MODEL`, `LLM_MODEL`,
`SARVAM_TTS_MODEL`, `SARVAM_TTS_SPEAKER`, `TRANSLATE_MODEL`, `EMBEDDING_MODEL`) are stored as
null, so they keep following `.env`. Each session validates the stored choices
against the catalogue and the language; a choice that no longer fits is
ignored with a warning rather than failing the call. Adding a model is a
catalogue edit, after checking it with the same probes (streaming support,
Tamil reply, tool call).

## Pronunciation (Sarvam dictionary)

The **Pronunciation** page of the portal holds a table of entries, one row
per word: the written form and how it should be spoken (exact matches, plain
text, up to 100 entries), for example `TNPDS` = `டி என் பி டி எஸ்`. The API
stores them as `word = spoken` lines. Saving it creates a Sarvam
pronunciation dictionary for the conversation language, deletes the previous
one, and stores the entries and the `dict_id` with the config version; new
calls pass that `dict_id` to Sarvam TTS over the websocket (Pipecat does not
send it, so `SarvamTTSWithDictionary` in `voice_agent/services.py` adds it
to the config message). "Listen before saving" synthesises a sample sentence
with the entries as typed (through a temporary dictionary that is deleted
afterwards). Saves from the configuration page keep the active dictionary; restoring an
older config version does not change it.

## Latency: what was measured and what to tune

Per-turn latency (user silent → first bot audio) is stored with Pipecat's
breakdown into non-overlapping parts, shown as the bar on each call page.
Findings from real Tamil sessions on 2026-10-19:

| Part | Before | After | Change made |
| --- | --- | --- | --- |
| Greeting (connect → first audio) | 2.0 – 3.5 s | **0.45 – 0.5 s** | Optional fixed opening line spoken straight through TTS (`AGENT_GREETING` or the portal's "Opening line"); the LLM no longer composes the greeting |
| LLM inference per reply | 1.0 – 3.3 s | **0.35 – 0.4 s** | `sarvam-105b` thinks before answering (500 – 1,500 hidden characters); `LLM_REASONING_EFFORT=none` turns that off. `low` only shortens it |
| Duplicate greeting on web sessions | one extra LLM run | fixed | the WebRTC transport can fire "connected" twice; the greeting is now triggered once |
| Date questions | hallucinated "May 2024" | correct | the current date and time are appended to the system prompt per session |

Typical reply turn is now about 1.3 – 1.5 s: endpointing wait 200 ms (VAD
`stop_secs`), Smart Turn end-of-turn detection 40 – 460 ms, Sarvam STT
~280 ms, LLM ~400 ms, Sarvam TTS 200 – 350 ms. The remaining levers are the
VAD stop time and the turn-detection strategy; the portal shows which part
dominates on any call. Pipeline start-up is ~330 ms and is not a factor.

Why the first turns felt slowest: open-ended prompts (a greeting, "what's
today's date") made the reasoning model think longest, so turns 0 and 1 carried
1.5 – 6 s of hidden reasoning while short follow-ups thought less.

## Phone calls with Plivo

The same agent answers phone calls through Plivo Audio Streaming. The runner
serves the websocket at `/ws` and, in Plivo mode, answers Plivo's webhook with
the `<Stream>` XML that points at it; audio is 8 kHz μ-law and the pipeline
switches to 8 kHz automatically for telephony sessions.

```bash
# 1. expose the bot
ngrok http 7860                                   # note the https host
uv run python bot.py -t plivo --proxy <ngrok-host>

# 2. point a Plivo Application at it (console or API):
#    Answer URL = https://<ngrok-host>/   method POST
#    and attach your Plivo number to that application.
curl -u "$PLIVO_AUTH_ID:$PLIVO_AUTH_TOKEN" -H 'content-type: application/json' \
  -d '{"answer_url":"https://<ngrok-host>/","answer_method":"POST"}' \
  https://api.plivo.com/v1/Account/$PLIVO_AUTH_ID/Application/<app_id>/

# 3a. inbound: call your Plivo number
# 3b. outbound: ring MOBILE_NUMBER from PLIVO_FROM_NUMBER
uv run python scripts/plivo_call.py --answer-url https://<ngrok-host>/
```

Each new ngrok session gets a new host on the free plan, so repeat step 2
after restarting ngrok. `scripts/plivo_point.py` does step 2 for you and
switches the dev number between the laptop and the server:

```bash
uv run python scripts/plivo_point.py status   # where the number rings now, and whether that bot answers
uv run python scripts/plivo_point.py local    # the current ngrok tunnel (from ngrok's local API)
uv run python scripts/plivo_point.py prod     # https://callagent.wizzgeeks.com/ (PLIVO_PROD_HOST)
```

It only ever edits the application named `PLIVO_APP_NAME` (default
`TNPDS_Custom_Development`) and refuses to touch one whose answer URL points
at apps.sarvam.ai, which is the live app on the other number. `PLIVO_AUTH_ID`/`PLIVO_AUTH_TOKEN` are required for
the bot to hang up calls it ends itself.

Verified on 2026-10-19 with a real Plivo number: webhook answered through
ngrok, inbound and outbound calls both opened `/ws`, 8 kHz sessions started,
and multi-turn Tamil conversations ran (TTS reconnects between turns are
normal). Transcripts are logged at INFO as `User:` / `Assistant:` lines.
Hang-up handling from both sides and call-quality tuning still need a
dedicated test pass.
# tnpds-call-agent
