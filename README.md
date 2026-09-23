# liuer-bot

`liuer-bot` is a Windows 11 Discord Bot project with provider-neutral,
local-model-first architecture. It supports LM Studio as the local fallback and
Venice as an explicitly selected API backend. The project provides one dedicated chat channel, text and image generation, plus
SQLite-backed community Persona commands and public status presentation; it
does not use Web Search or provider-owned conversation memory.

## Current behavior

- Exactly one configured guild text/news channel is the conversation stream.
  A human message must begin with an explicit leading address to trigger. The
  primary form is the literal call name `六耳`; no Discord @mention or reply
  relationship is required. As a compatibility form, a structured mention of
  the current Bot is also accepted only when it is the leading address. The
  prefix is removed before routing, model input,
  and persistence. Ordinary conversation, later call-name/mention occurrences,
  bot authors, DMs, threads, and other channels are ignored. A supported image
  also requires a leading address; an image alone never triggers.
- Supported PNG, JPEG/JPG, WebP, and GIF attachment candidates are downloaded
  only through Discord's `Attachment.read(use_cached=True)` boundary.
- Image candidates are limited to four per message, 10 MiB each, 20 MiB total,
  and 40 megapixels after decode. A selected-image failure aborts that message
  before it enters the generation queue.
- Pillow validates decoded bytes and dimensions. Static PNG/JPEG/WebP retain
  their validated original bytes; GIF and animated images become a PNG of
  frame 0. Image bytes, filenames, and URLs are never logged.
- Decode-safety limits are separate from model-input limits. Oversized VLM
  inputs are proportionally downscaled in memory before Base64 serialization;
  images already within model limits retain their original prepared bytes and
  are never upscaled.
- Prepared PNG, JPEG, and WebP data is serialized in memory as standard-Base64
  data URIs for OpenAI-compatible `image_url` blocks. Discord CDN URLs are
  never sent to either provider and no images are written to disk.
- Multiple images remain one user message, one globally serialized model
  request, and one logical response delivered through one or more Discord
  messages. GIF and animated images use their
  first frame only, normalized to PNG before model delivery.
- Every triggered request passes through one global FIFO async generation
  queue. At most one `Responder.generate` call runs at once.
- With LM Studio selected, `CHAT_SHORT` text-only requests use its native
  `/api/v1/chat` endpoint with `reasoning=off` and `store=false` for low-latency
  casual chat. LM Studio `NORMAL`, `DEEP`, and every multimodal request use
  `/v1/chat/completions`; no native multimodal path is used.
- With Venice selected, all modes and multimodal requests use
  `/api/v1/chat/completions`. Venice `CHAT_SHORT` sends
  `reasoning.enabled=false`; `NORMAL` and `DEEP` retain provider-default
  reasoning without invented effort levels.
- `LIUER_LLM_PROVIDER` selects exactly one provider for the process lifetime.
  LM Studio preserves its existing `/v1/models` preflight and transport matrix.
  Venice uses one `GET /models` response to validate exact text and vision model
  IDs before Discord login, and stateless
  OpenAI-compatible `POST /chat/completions` for every response mode.
- Every Venice request disables Venice's default system prompt and explicitly
  disables web search, scraping, citations, and X search. 六耳 Persona,
  Nickname, prompt policy, and conversation memory remain owned by liuer-bot;
  Venice characters and provider-side conversation state are not used.
- Venice config has separate exact text and vision model IDs. Text-only Human,
  BotChat, `CHAT_SHORT`, `NORMAL`, and `DEEP` requests use the text model.
  Only current prepared images select the vision model; historical image and
  ReplyContext markers do not. The text model may be text-only. Startup fails
  closed unless the vision model explicitly confirms vision and sufficient
  multiple-image support. Both IDs may be identical for one multimodal model.
- Community Persona commands are guild-scoped to the guild owning the
  configured public status channel and never enter the model generation queue.
- Every production inference includes one compact system prompt with immutable
  core rules, a fixed 六耳 identity, the optional current Active Nickname, and
  the current Active Persona, followed by the conversation-context policy and
  response-mode layer. When no
  community Persona is active, the packaged Default Persona is used.
- Every request is routed deterministically into `CHAT_SHORT`, `NORMAL`, or
  `DEEP` using only the current message text and prepared-image presence. The
  router does not call a model, read SQLite, inspect conversation history, or
  mutate the request. Ambiguous input defaults to `NORMAL`; multimodal input
  has a `NORMAL` minimum unless the user explicitly asks for a deep answer.
- Production startup loads the packaged Default Persona, creates/migrates the
  configured SQLite database to schema v6, loads the current Active Persona and
  optional Active Nickname snapshots into memory, then performs the selected
  provider's non-generation model preflight before Discord startup.
- Successful community Persona publishes are immutable, versioned, globally
  cooldown-limited, and subject to the configured per-user daily quota.
- Reset-to-default and rollback are domain APIs used by the Discord Persona
  commands; their policy remains authoritative in the domain service.
- Startup preflights the selected provider's exact configured model ID(s)
  before Discord connects.
- Successful model turns are persisted in SQLite schema v6 with bounded recent
  in-memory state. At generation time, the serialized worker reads that snapshot
  and injects it into the selected provider's stateless request.
- `/context clear` is available only to guild Administrators or members with
  Manage Server in the configured chat channel. It publicly announces a
  successful clear with mentions suppressed; permission, channel, and service
  failures remain ephemeral. A clear advances a persistent conversation epoch
  without deleting prior rows, so the active in-memory and model context starts
  empty while old turns remain available for audit.
- The shared channel has one recent completed-turn timeline, defaulting to the
  latest 12 turns. Multiple speakers use sanitized stored display-name snapshots
  and explicit `HUMAN` or `BOT` author kinds.
  Historical user/assistant text is rendered as explicit context; historical
  images are represented only by an unavailable-image-count marker and are never
  re-sent.
- The 12-turn window is followed by a deterministic rendered-character budget:
  `CHAT_SHORT` defaults to 1000 characters, `NORMAL` to 12000, and `DEEP` to
  24000. The planner keeps the newest complete user/assistant turns as one
  contiguous suffix, never splits a turn, and never summarizes or deletes
  trimmed rows. These are Unicode character budgets for prior completed
  history, not exact LM Studio token limits. Multimodal requests use the
  `NORMAL` history budget. The intentionally lower `CHAT_SHORT` default
  prioritizes local conversational latency; current content, ReplyContext, Persona,
  Nickname, and system policy are outside this historical budget.
- Guild Administrators and members with Manage Server can use `/status` and
  `/health` only in `LIUER_PERSONA_STATUS_CHANNEL_ID`. Both commands are
  ephemeral and mention-suppressed; they never update the public Persona
  status board. `/status` is a fast local snapshot of runtime, queue,
  conversation, identity, schema, and history-budget state and explicitly does
  not actively probe the provider. `/health` performs read-only runtime,
  SQLite, and one active-provider `GET /models` check, verifies the exact
  configured model ID and required compatibility, and
  never generates model output or enters the generation queue. Operational
  responses are intentionally temporary; no exact Discord disappearance time
  is promised.
- A Discord reply relationship never triggers the Bot by itself. After the
  current message passes the leading `六耳`/nickname/address check, a resolved,
  cached, or REST-fetched reply may be added as quoted current-request context.
  Referenced image bytes, URLs, and filenames are not downloaded or re-sent;
  only a safe image-count marker is included. Deleted or unavailable targets
  become an explicit unavailable marker, while the current addressed message
  still proceeds. Reply context is not persisted as part of the current turn.
- `CHAT_SHORT` keeps native `/api/v1/chat`, `reasoning=off`, and `store=false`,
  using a compact transcript. `NORMAL`, `DEEP`, and multimodal requests use
  alternating historical user/assistant roles in Chat Completions.
- Generation is non-streaming. One model generation may produce multiple
  Discord messages when the complete answer exceeds Discord's 2000-character
  message limit. The deterministic splitter prefers paragraph, newline,
  sentence, and whitespace boundaries before a hard split, and repairs
  standard fenced code blocks across chunks. The first chunk replies to the
  triggering message; later chunks are sequential channel messages. Every
  chunk suppresses mentions, and a partial Discord delivery failure never
  regenerates the model response.
- Other Bot messages remain ignored unless an Administrator or Manage Server
  member uses the guild message context menu `讓六耳回覆此 Bot` on one eligible
  Bot message in the configured chat channel. That selected message is the
  opening Bot turn; the first answer is public and replies to it. The active
  peer is matched by exact Discord user ID, and later peer messages do not need
  a `六耳` or nickname prefix. Human routing is unchanged, Liuer's own
  messages and webhook messages are ignored, and only one process-local BotChat
  session may be active at a time.
- BotChat uses a six-reply safety limit and a five-minute lazy idle timeout.
  `/botchat stop` is available to administrators in the chat or status
  channel; `/botchat status` is an ephemeral metadata-only view in the status
  channel. Sessions are not persisted to SQLite. Bot turns are persisted with
  `author_kind=BOT`, rendered as Bot speakers in context, and use the same
  global generation queue, image preparation, response modes, and safe Discord
  delivery as Human turns. `/context clear` also closes the active BotChat
  session after the epoch commit succeeds.

The current immutable primary call name is `六耳`. The Bot can also have one
optional global Active Nickname, managed with `/set name`; it is not per-user,
per-guild, or per-channel. Persona and nickname updates reuse the same
20-minute/global and three-successful-updates-per-user/day defaults, but their
cooldown and quota accounting are independent. Accepted examples include
`六耳 你好`, `六耳你好`, and `六耳：幫我看看`. The leading name and adjacent
separator punctuation/whitespace are transport addressing metadata and are not
sent to LM Studio or persisted in conversation turns. A later occurrence of
`六耳` or the active nickname in an otherwise unaddressed sentence is not a
trigger. A bare leading address uses the single character `…` as its
deterministic conversational representation, including addressed image-only
requests, so provider text fields are never empty.

## Requirements

- Windows 11
- Python `>=3.13,<3.14`
- Either a locally running LM Studio server with its configured model loaded,
  or Venice credentials plus exact text and vision model IDs
- Pillow `>=12.3,<13` (installed automatically with the project)

## Setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Configure `.env` locally; it is ignored by Git:

```text
LIUER_DISCORD_TOKEN=replace-with-your-discord-bot-token
LIUER_CHAT_CHANNEL_ID=123456789012345678
LIUER_PERSONA_STATUS_CHANNEL_ID=223456789012345678
LIUER_CONTEXT_RECENT_TURNS=12
LIUER_CONTEXT_HISTORY_MAX_CHARS_CHAT_SHORT=1000
LIUER_CONTEXT_HISTORY_MAX_CHARS_NORMAL=12000
LIUER_CONTEXT_HISTORY_MAX_CHARS_DEEP=24000
LIUER_LLM_PROVIDER=lm_studio
LIUER_VENICE_BASE_URL=https://api.venice.ai/api/v1
LIUER_VENICE_API_KEY=
LIUER_VENICE_TEXT_MODEL_ID=
LIUER_VENICE_VISION_MODEL_ID=
LIUER_LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LIUER_LM_STUDIO_MODEL=replace-with-lm-studio-model-id
LIUER_LM_STUDIO_TIMEOUT_SECONDS=300
LIUER_LM_STUDIO_TEMPERATURE=0.7
LIUER_LM_STUDIO_MAX_TOKENS=4096
LIUER_RESPONSE_CHAT_SHORT_MAX_TOKENS=1024
LIUER_RESPONSE_NORMAL_MAX_TOKENS=1536
LIUER_LM_STUDIO_API_TOKEN=
LIUER_IMAGE_DOWNLOAD_TIMEOUT_SECONDS=30
LIUER_MAX_IMAGES_PER_MESSAGE=4
LIUER_MAX_IMAGE_BYTES=10485760
LIUER_MAX_TOTAL_IMAGE_BYTES=20971520
LIUER_MAX_IMAGE_PIXELS=40000000
LIUER_MODEL_IMAGE_MAX_LONG_EDGE=2048
LIUER_MODEL_IMAGE_MAX_PIXELS=4000000
LIUER_MODEL_TOTAL_IMAGE_PIXELS=8000000
LIUER_PERSONA_MAX_CHARS=4000
LIUER_PERSONA_GLOBAL_COOLDOWN_SECONDS=1200
LIUER_PERSONA_DAILY_LIMIT=3
LIUER_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS=8
LIUER_NICKNAME_MAX_CHARS=32
```

`LIUER_LM_STUDIO_API_TOKEN` is optional. When configured, it is sent only as
`Authorization: Bearer ...` to the configured LM Studio base URL and is never
shown in normal logs, diagnostics, or configuration representation.

`LIUER_LLM_PROVIDER` defaults to `lm_studio`. Select `venice` only after adding
the API key and exact text and vision model IDs returned by Venice `/models`; the key belongs
only in the ignored local `.env`. Provider selection is fixed until restart.
Venice does not replace the local SQLite conversation source of truth or any
Persona, Nickname, routing, context-budget, queue, typing, or delivery policy.
For backward-compatible environment naming, the existing
`LIUER_LM_STUDIO_TIMEOUT_SECONDS`, `LIUER_LM_STUDIO_TEMPERATURE`, and
`LIUER_LM_STUDIO_MAX_TOKENS` values provide the shared request timeout,
temperature, and `DEEP` output ceiling for either active provider.

## Default Persona resource

The owner-editable fallback persona is packaged at
`src/liuer_bot/prompts/default_persona.txt`. The Bot reads it as UTF-8 once
when its production prompt builder starts; edit the file and restart the Bot
for a manual change to take effect. The file is not an environment setting and
is never logged.

Phase 3B-A stores only successful community Persona versions and active state
in SQLite; it never copies `default_persona.txt` into the database. A null
`active_version_id` means the packaged Default Persona is active. Community
Persona commands update runtime Active Persona state, not this file. Resetting
Active Persona returns to the packaged fallback. Core Rules
and Fixed Identity remain code-controlled and cannot be replaced by a
community persona.

Enable the Message Content Intent in the Discord Developer Portal before
starting the Bot.

## Runtime and preflight

```powershell
.\.venv\Scripts\python.exe -m liuer_bot doctor
.\.venv\Scripts\python.exe -m liuer_bot run
```

`doctor` is offline: it validates local Python and configuration, but does not
open or create SQLite. `run` initializes/migrates the Persona, nickname, and
conversation database, loads the in-memory Active Persona, Active Nickname, and
recent-turn snapshots, performs
a non-inference `GET /models` preflight, requires an exact model-ID match, then
starts the Discord runtime. A local startup failure or failed preflight prevents
Discord startup. Generation uses one native or Chat Completions POST per queue
job and has no automatic retry. A successful turn is committed before the queue
worker advances to the next request; a persistence failure prevents Discord
success delivery.

Image limits are independently validated; in particular, a total-image limit
smaller than the per-image limit is valid and simply becomes the stricter
effective cap. All selected images are prepared in their Discord-message order
before the one possible queue submission.

The response-mode budgets are separately configurable. `CHAT_SHORT` defaults
to 1024 output tokens, `NORMAL` to 1536, and `DEEP` uses the existing LM Studio
maximum of 4096. Configuration is fail-closed and requires
`1 <= CHAT_SHORT <= NORMAL <= LM_STUDIO_MAX_TOKENS`; values are never silently
clamped. The model-input limits are separately configurable: a longest edge of 2048,
four million pixels per image, and eight million pixels total by default.
For multiple images, each receives the equal effective budget
`min(per-image limit, total limit // image count)`. This bounds visual input
size without changing acquisition/decode-safety limits. The global output
token default is 4096; high-resolution input is controlled by image
normalization rather than a larger output-token ceiling.

`NORMAL` and `DEEP` text-only requests retain plain-string user content and add
exactly one assembled system prompt through Chat Completions:

```json
{
  "model": "configured-model-id",
  "messages": [
    {"role": "system", "content": "Assembled 六耳 system prompt"},
    {"role": "user", "content": "Conversational content after address removal"}
  ],
  "temperature": 0.7,
  "max_tokens": 1536,
  "stream": false
}
```

`CHAT_SHORT` text-only requests use the same assembled system prompt through
the stateless native endpoint. The payload is deliberately separate from the
Chat Completions schema:

```json
{
  "model": "configured-model-id",
  "input": "Conversational content after address removal",
  "system_prompt": "Assembled 六耳 system prompt",
  "reasoning": "off",
  "max_output_tokens": 1024,
  "temperature": 0.7,
  "stream": false,
  "store": false
}
```

The native URL is derived from the configured `/v1` base URL on the same
server, for example `http://127.0.0.1:1234/v1` becomes
`http://127.0.0.1:1234/api/v1/chat`. Native failures do not fall back to Chat
Completions and do not retry. Native response statistics are diagnostics only;
reported reasoning tokens produce a warning but never become answer text.

Requests with prepared images use the same one system message, followed by one
user content array with a text block and then the original image order as
`image_url` blocks. Each URL is an in-memory
`data:image/{png|jpeg|webp};base64,...` representation. No implicit image
prompt, model-side URL fetch, vision probe, fallback to text-only inference,
or automatic retry is performed.

The single system message layers immutable Core Rules, Fixed Identity, the
active Persona, and one deterministic `[RESPONSE MODE]` instruction in that
order. The response instruction is policy text selected locally from the
request; there is no extra classifier inference, reasoning/thinking control,
streaming, or response history in this phase.

## Community Persona commands and public status

Phase 3B-B exposes the Bot-global Community Persona through a dedicated,
read-only-friendly guild text or announcement channel configured with
`LIUER_PERSONA_STATUS_CHANNEL_ID`. It remains separate from
`LIUER_CHAT_CHANNEL_ID`, and both configured channels must belong to the same
guild. The status channel is presentation state, not a message trigger channel.

The available guild-scoped commands are:

- `/set name` is available to all members and sets the one global Active
  Nickname. The normalized nickname is limited by `LIUER_NICKNAME_MAX_CHARS`
  (32 by default), and uses the Persona policy values with independent
  cooldown/quota accounting. A successful update is announced publicly with
  mentions suppressed; validation, cooldown, quota, and sync-warning responses
  remain ephemeral. The update does not enter the generation queue.

- `/persona set` opens a required multiline Modal with a maximum of 4000
  characters. A successful publish uses the existing 20-minute global
  cooldown and three-successful-publishes-per-user-per-day policy (UTC+8
  reset by default).
- `/persona status` shows the active version, cooldown, daily usage, reset
  time, and a reference to the public status channel without echoing Persona
  text.
- Administrators or members with Manage Server may use `/persona reset`,
  `/persona rollback`, and `/persona history`.

There is intentionally no `/persona show` command. The public status channel is
the source of truth for active Persona visibility. The Bot maintains one
canonical Bot-owned message there and edits it after set, reset, or rollback;
startup recovers the stored message or creates a replacement when needed.
Persona text is shown in full when it fits the Discord embed description budget;
otherwise a deterministic opening excerpt and character count are shown. Persona
command responses are ephemeral, while the successful nickname announcement
may be public. Responses and the persistent status message use
`AllowedMentions.none()` so nickname or Persona text cannot create notification pings.
Presentation sync failure does not roll back a committed Persona mutation; the
next startup or mutation retries the status refresh.
The same canonical message also shows the immutable fixed name `六耳`, the
current nickname (or `未設定`), and the `/set name` instruction. Nickname
updates edit this message rather than creating a second public board.

## Conversation context commands

The guild-scoped `/context clear` command starts a new conversation epoch in the
configured `LIUER_CHAT_CHANNEL_ID`. Only Administrators and members with Manage
Server may use it. A successful clear is a public chat-channel announcement with
`AllowedMentions.none()`; failed checks and failed clear operations are
ephemeral. The command does not enter the generation queue or call LM Studio.

Clearing never deletes the previous SQLite conversation rows. The database
commit advances the active epoch before the in-memory snapshot changes, and
startup restores only the active channel and epoch. A generation already
captured before the clear may finish and is retained under its old epoch for
audit, but it cannot re-enter the active history. Persona and Nickname state
are unaffected. An explicitly resolved Discord `ReplyContext` remains current
request enrichment even if it refers to a pre-clear message; it is not written
into the conversation turn.

When an Active Nickname exists, the dynamic prompt layer treats it as the
preferred conversational self-name: natural self-reference and ordinary name
answers should prefer the nickname without repeatedly mentioning `六耳`.
`六耳` remains the canonical fixed identity and may be explained when a user
specifically asks about the formal name or the relationship between the names.

Recommended status-channel permissions are read-only access for members and
`View Channel`, `Send Messages`, `Embed Links`, and `Read Message History` for
the Bot. Configure permission overwrites in Discord; the Bot does not rewrite
channel permission settings.

## Runtime status and health commands

`/status` and `/health` are guild-scoped control-plane commands restricted by
the same Administrator-or-Manage-Server policy used by the other admin
commands. The Bot enforces the existing `LIUER_PERSONA_STATUS_CHANNEL_ID`
restriction at execution time; Discord's command picker may still display a
guild command in other channels. Every response is ephemeral with
`AllowedMentions.none()`.

`/status` reads already-loaded in-memory state only. It does not make an HTTP
request, open a health-check SQLite connection, call a provider, enqueue work,
or mutate state. Its output includes local runtime readiness and monotonic
uptime, Discord websocket latency when available, generation worker/busy/
  waiting state, schema v6 and active conversation epoch/recent count, Persona
and Nickname labels, and the configured `CHAT_SHORT`/`NORMAL`/`DEEP` history
character budgets. It shows the selected provider and configured model roles,
and labels the active probe as not performed by `/status`.

`/health` performs bounded read-only checks independently of model generation:
Discord/runtime readiness, generation-worker liveness, `SELECT 1` plus the
expected SQLite schema version, and exactly one `GET /models` request to only
the active provider. LM Studio preserves its existing optional-token behavior;
Venice uses its Bearer API key and checks both exact model roles from one
catalog response. The vision role requires explicit vision and multiple-image
compatibility. The probe has a
five-second application-level timeout without retry. It reports safe classified
states such as `READY`, `MODEL_MISSING`, `INCOMPATIBLE_MODEL`,
`UNREACHABLE`, `TIMEOUT`, `AUTH_ERROR`, `HTTP_ERROR`, or `INVALID_RESPONSE`
without exposing tokens, headers, response bodies, prompts, Persona text, or
conversation contents. Queue load alone does not make health unhealthy.

## Discord typing indicator

Once a valid Human or BotChat request is accepted, Discord shows 六耳 as
typing while the request is being prepared, waiting in the serialized queue,
generated, and persisted. Ignored or unaddressed messages do not produce a
typing indicator. Typing is best-effort UX and does not guarantee a successful
response; it ends when actual Discord response delivery begins.

## Limitations

- Recent context first uses the turn-count window
  (`LIUER_CONTEXT_RECENT_TURNS`, default 12), then the mode-specific rendered
  character budgets (`CHAT_SHORT` 1000, `NORMAL` 12000, `DEEP` 24000 by
  default). The planner removes only the oldest complete turns and preserves
  a contiguous newest suffix. This is not exact tokenizer accounting, and
  there is no summarization or token-budget conversion.
- Conversation history is captured at serialized generation time, not Discord
  intake time, so a completed queued turn is visible to the next generation.
- Reply context is resolved only after channel, author, and leading-address
  validation. It can duplicate a recent turn, and no complex deduplication is
  attempted.
- Historical image bytes, Base64, URLs, and attachment metadata beyond the
  image count are never persisted.
- Persona state is global across the Bot, not per guild, channel, or user.
- There is no automatic hot reload of database edits or the packaged default;
  restart the Bot to load external changes.
- No reasoning/thinking control beyond the existing `CHAT_SHORT` native
  `reasoning=off`, and no streaming router.
- No streaming or generation failure retry UX; long responses are split for
  Discord delivery without truncating the generated answer.
- The FIFO queue is unbounded; there is no queue position, fairness, quota,
  rate limit, or generation timeout user feedback.
- Provider model auto-download, auto-load, auto-selection, retry, failover, and
  rate-limit scheduling are absent.
- Video/audio understanding, multi-frame GIF analysis, and OCR are absent.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m liuer_bot --help
```
