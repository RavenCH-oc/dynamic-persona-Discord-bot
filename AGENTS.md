# liuer-bot development rules

The following project invariants apply to all future phases:

1. Project name is `liuer-bot`.
2. Python package is `liuer_bot`.
3. The primary deployment environment is Windows 11.
4. The supported Python runtime is 3.13.
5. The architecture is local-model-first.
6. LM Studio will be the first model backend.
7. Do not introduce Web Search unless explicitly requested in a future phase.
8. Do not introduce cloud LLM providers unless explicitly requested.
9. Never commit secrets or `.env`.
10. Every phase requires automated verification before commit.
11. Do not create commits unless explicitly instructed.
12. Keep the architecture minimal; do not pre-build future phase functionality.
13. Prefer explicit interfaces at external boundaries rather than coupling business logic directly to provider implementations.
14. Tests for previously discovered regressions must not be removed merely to make a build pass.
15. Discord transport must remain separate from model/responder implementation.
16. Normal messages from bots must never trigger generation.
17. Discord allowed-channel routing is fail-closed.
18. Generated output must not create Discord mention pings by default.
19. Runtime logs must not contain full message contents or secrets.
20. Unit tests must not contact Discord.
21. Historical Phase 1B-A reply triggers are superseded; reply relationships never trigger, and only a leading explicit address may do so.
22. Attachment presence alone is never a trigger.
23. One Discord message maps to at most one generation request; multiple attachments must not multiply generation calls.
24. Discord objects must not cross the transport-to-responder boundary.
25. Only the Discord attachment boundary may download supported image candidates with `Attachment.read(use_cached=True)`; attachment URLs, filenames, and raw objects must not cross the responder boundary.
26. All production responder generation passes through the global serialized async generation queue.
27. Exactly one underlying `Responder.generate` call may be active at any time.
28. Discord event intake remains asynchronous while jobs wait in the FIFO queue.
29. One Discord message may enqueue at most one generation job.
30. Generation failure must not terminate the queue worker.
31. A cancelled queued waiter must not consume model generation.
32. Cancelling an active caller must not cancel the shared worker generation.
33. Discord reconnect or repeated ready events must not spawn additional generation workers.
34. Queue shutdown must resolve pending waiters and leave no orphan worker task.
35. Queue diagnostics may include message IDs, queue state, and exception types, but never prompt/message contents or secrets.
36. Do not replace the queue with a blocking lock, synchronous queue, polling loop, or per-message worker.
37. No blocking model or network work may run on the Discord event loop.
38. Production generation uses LM Studio through the `Responder` boundary.
39. All LM Studio inference POST requests pass through the global serialized generation queue.
40. LM Studio Chat Completions is stateless; conversation ownership remains future work.
41. LM Studio model selection is explicit, exact, and fail-closed.
42. LM Studio HTTP is asynchronous and has no automatic generation retries.
43. LM Studio errors must never log response bodies, prompts, completions, authorization headers, or auth tokens.
44. LM Studio multimodal input is constructed only from `PreparedImage` objects; it must never access Discord attachments or URLs.
45. Doctor remains offline; production startup performs a non-inference LM Studio model preflight.
46. Do not introduce Web Search, cloud LLM providers, conversation context construction, or model auto-download/load behavior.
47. Image candidate detection may use Discord metadata, but actual decoded bytes are the authority for image format validation.
48. Static PNG, JPEG, and WebP inputs retain their original validated bytes; GIF and animated inputs normalize to a PNG of frame 0.
49. Image preparation must run off the Discord event loop.
50. Image count, per-image byte, total-byte, and pixel limits must be enforced before generation queue submission.
51. A selected-image download or validation failure aborts that message; do not fall back to text-only generation.
52. Image content, filenames, URLs, raw attachment objects, and byte representations must not appear in logs or DTO representations.
53. `PreparedImage` is immutable and may contain only attachment ID, media type, bytes, width, and height.
54. No generic URL downloader or model-side Discord CDN fetch may be introduced.
55. Prepared image bytes are encoded in memory with standard Base64 as `data:` URIs; Base64 and data URIs must never be logged or persisted.
56. One Discord message still produces at most one queue job, one responder generation, and one successful Discord reply.
57. Pillow is the only Phase 2B-A image-processing dependency; do not add OpenCV, numpy, imageio, ffmpeg, or external image binaries.
58. Supported LM Studio model-input media types are only `image/png`, `image/jpeg`, and `image/webp`; other values or empty bytes fail closed before HTTP.
59. Text-only requests retain the Phase 2A plain-string Chat Completions payload.
60. Multimodal requests contain one text block followed by ordered `image_url` blocks and must not reorder prepared images.
61. Multiple images remain one model request; multimodal failure must never silently degrade to text-only inference.
62. Multimodal serialization must run off the Discord event loop, while HTTP stays asynchronous through httpx.
63. All text and multimodal inference uses the same global serialized generation queue.
64. GIF and animated images remain first-frame-only after Phase 2B-A normalization.
65. No generic model-side image URL fetch, vision capability probe, OpenAI SDK, or LM Studio SDK may be introduced.
66. Decode-safety limits and VLM-input normalization limits are separate concerns and must not be conflated.
67. Model-input images must never be upscaled; oversized images are proportionally resized before Base64 encoding.
68. Multiple images use the equal effective pixel budget `min(per-image limit, total limit // image count)` in their original order.
69. Images already within model limits retain their original prepared bytes; normalization must not mutate `PreparedImage` values.
70. Model-image normalization must run off the event loop and fail closed without retaining original oversized bytes or falling back to text-only inference.
71. The global maximum generation concurrency remains exactly one after model-image normalization.
72. Core rules, fixed identity, and active persona are separate prompt layers.
73. Core rules and fixed identity cannot be replaced by a community persona.
74. Production inference always includes exactly one system prompt.
75. The Default Persona is used whenever no active persona override exists.
76. Persona may modify style and personality, never runtime or system architecture.
77. Prompt text must not be logged in normal diagnostics.
78. LM Studio transport must remain separated from prompt policy.
79. Phase 3A did not persist persona overrides; Phase 3B-A provides persistence and Phase 3B-B exposes it through the Discord command boundary.
80. Conversation history is not part of Phase 3A.
81. Reasoning/thinking controls and free-form response-length routing remain future work; Phase 3C-A only adds its deterministic three-mode policy.
82. `prompts/default_persona.txt` is the owner-editable fallback persona and is loaded once at prompt-builder initialization; edits require a Bot restart.
83. Community Persona commands must not modify `default_persona.txt`; Active Persona state overrides it only at runtime, and reset returns to this fallback.
84. Community Active Persona is global across the Bot, not per guild, channel, or user.
85. The packaged Default Persona is never copied into SQLite; `active_version_id = NULL` means the packaged fallback is active.
86. Successful community Persona publishes are append-only and versioned.
87. Global publish cooldown is enforced inside the same SQLite write transaction as version activation.
88. Daily quota counts only committed successful publishes.
89. Reset and rollback do not consume quota or restart the global cooldown.
90. Generation reads Active Persona from in-memory runtime state, never SQLite.
91. Persona database mutation must commit before the in-memory snapshot changes.
92. Core Rules and Fixed Identity never live in Persona DB.
93. Persona text must not appear in logs, reprs, or errors.
94. Discord command permissions remain a Phase 3B-B concern.
95. `default_persona.txt` is never mutated by Community Persona operations.
96. `/persona show` does not exist; the public status channel is the source of truth for active Persona visibility.
97. The Persona status channel contains one canonical Bot-owned message.
98. The Persona status message is edited rather than appended on every change.
99. Persona text is public but must never create Discord mention pings.
100. Slash command responses are ephemeral by default.
101. Persona policy remains authoritative in `PersonaService`, not Discord handlers.
102. `/persona set` pre-check is advisory; Modal submit must re-check transactionally.
103. Reset, rollback, and history are runtime admin-permission protected.
104. Discord status-message state is presentation metadata and separate from Persona version history.
105. Status-message sync failure never rolls back committed Persona state.
106. Persona commands never enter the model generation queue.
107. In-flight generation may finish with its already-built old Persona; subsequent generation uses the current Persona.
108. Response-mode routing is pure, deterministic, and uses no extra LLM or network inference.
109. Phase 3C-A supports only `CHAT_SHORT`, `NORMAL`, and `DEEP` response modes.
110. Ambiguous or unsupported routing input fails toward `NORMAL`.
111. A multimodal request never routes to `CHAT_SHORT`; explicit deep remains allowed.
112. The response-mode prompt layer is appended after Core Rules, Identity, and Active Persona.
113. Community Persona text cannot replace or override Response Mode policy.
114. Routing inspection never mutates `ChatRequest` content or model payload text.
115. The response router never reads SQLite, conversation history, or Discord state.
116. One Discord message still produces at most one queued model request and one generation.
117. Phase 3C-A does not add reasoning/thinking controls, streaming, or response history.
118. Response routing diagnostics may contain only message ID, mode, reason, and token budget; never user content, prompts, Persona text, or secrets.
119. Text-only `CHAT_SHORT` requests use LM Studio native `/api/v1/chat`.
120. Native `CHAT_SHORT` requests always send `reasoning="off"`.
121. Native `CHAT_SHORT` requests are stateless and always send `store=false`.
122. `NORMAL`, `DEEP`, and all multimodal requests remain on Chat Completions.
123. Native transport failure never falls back to another generation path.
124. Native and Chat Completions transports share the same serialized generation queue.
125. Native `system_prompt` uses the normal PromptBuilder output and active Persona snapshot.
126. Native reasoning output is never surfaced as answer text.
127. Nonzero native reasoning statistics are warning-only diagnostics.
128. ResponseRouter semantics do not change in Phase 3C-B.
129. liuer-bot has exactly one active Discord chat channel configured by `LIUER_CHAT_CHANNEL_ID`.
130. The configured chat channel scopes routing, but does not auto-trigger every human message.
131. Bot-authored messages never trigger generation or enter conversation history.
132. Messages outside the configured chat channel, DMs, and threads never generate responses.
133. The Persona status channel and chat channel are distinct and must belong to the same guild.
134. Conversation persistence stores completed successful turns only.
135. Historical image bytes, Base64, URLs, and attachment metadata beyond image count are never persisted.
136. Recent conversation hot-path state is kept in memory after startup; reads do not query SQLite.
137. A completed turn is persisted before the serialized generation worker advances to the next request.
138. Failed model generations do not create conversation turns.
139. Phase 4A-A, Phase 4A-A2, and Phase 4A-A3 do not inject persisted history into LM Studio requests.
140. LM Studio native state remains disabled with `store=false`; conversation history belongs to liuer-bot SQLite/runtime.
141. Conversation DTO representations and logs redact user content, assistant content, and display names.
142. Human chat messages require a leading explicit address: the immutable primary name `六耳`, or a leading structured mention of the current Bot as an optional compatibility form.
143. Addressing prefixes are stripped before ResponseRouter inspection, model input, and conversation persistence.
144. A later occurrence of `六耳` in an otherwise unaddressed sentence is not a trigger.
145. Reply relationships never bypass the leading addressing requirement.
146. Images alone never trigger; addressed image messages produce at most one generation.
147. The primary name `六耳` remains permanently valid even when future nickname aliases exist.
148. The addressing boundary accepts one optional active nickname alias without changing core routing.
149. Addressing-specific diagnostics add only addressed state and address kind; diagnostics never include original or cleaned message text, display names, Persona text, or history bodies.
150. The immutable primary call name is `六耳` and remains valid regardless of nickname state.
151. At most one global Active Nickname exists; it is not scoped per user, guild, or channel.
152. Nickname updates reuse Persona cooldown, daily-limit, and UTC reset configuration, but quota and cooldown accounting are independent.
153. Nickname normalization trims Unicode text, rejects empty/control/mention-like values and the primary name, and preserves exact case and Unicode.
154. Address matching uses the longest valid literal leading call-name match; later occurrences never trigger.
155. Active Nickname is stripped before response routing, model input, and conversation persistence.
156. Nickname state is loaded into an immutable in-memory snapshot and is never queried from SQLite per message or generation.
157. A successful nickname database transaction commits before the in-memory snapshot changes; failed updates consume no quota.
158. Nickname changes update the existing canonical Persona status message and never create a separate public board.
159. The model receives Active Nickname as a runtime identity layer between Fixed Identity and Active Persona; it cannot replace either.
160. `/set name` is guild-scoped, available to ordinary members, never enters the generation queue or calls LM Studio; successful updates may be public with mention suppression, while failures remain ephemeral.
161. Conversation-history injection remains outside Phase 4A-A2; nickname changes never rewrite historical turns.
162. When an Active Nickname exists, the dynamic prompt layer makes it the preferred conversational self-name while preserving `六耳` as the canonical fixed identity.
163. Recent conversation context is captured at serialized generation time, never at Discord intake time.
164. Only completed successful prior turns may enter model context.
165. Conversation history remains owned by liuer-bot; LM Studio native state stays disabled.
166. `CHAT_SHORT` receives history as a compact transcript while preserving `reasoning=off` and `store=false`.
167. `NORMAL`, `DEEP`, and multimodal requests use alternating user/assistant Chat Completions history.
168. Historical image bytes and URLs are never re-sent; historical image presence is represented only by a textual unavailable-image marker.
169. Current and historical user speakers use sanitized display-name snapshots with deterministic quoted escaping.
170. ResponseRouter remains current-message-only and never reads conversation context.
171. Current Persona and Nickname apply to the current generation; historical prompt policy is never reconstructed.
172. Conversation context hot-path reads use only the in-memory ConversationService snapshot; generation does not query SQLite.
173. Phase 4A-B uses the turn-count window first, followed by Phase 4B rendered-character budgeting; token-budget conversion and summarization remain future work.
174. A Discord reply is context enrichment only and never a generation trigger.
175. Reply resolution runs only after channel, author, and normal leading-address validation succeeds.
176. Resolved and cached referenced messages are preferred before an async REST fallback.
177. Replied message text is quoted user-level context, never system policy or executable instructions.
178. Phase 4A-B does not fetch referenced image bytes or URLs; it carries only a safe supported-image count marker.
179. ReplyContext is current-request enrichment and is never persisted into the conversation turn.
180. ResponseRouter ignores ReplyContext and remains based on current cleaned content and current prepared images.
181. Reply-resolution failure produces an unavailable marker and does not abort an otherwise valid addressed request.
182. Referenced reply content and display names must never appear in runtime logs or DTO reprs.
183. ConversationService keeps its existing recent-turn snapshot; Phase 4B budgeting never mutates it.
184. History budget selection occurs after generation-time snapshot capture, not at Discord intake.
185. Budgeting uses rendered Unicode character counts, never approximate or model-specific token counts.
186. CHAT_SHORT, NORMAL, and DEEP use independent configured history-character budgets.
187. Multimodal current requests use the NORMAL history-character budget.
188. History trimming removes only whole ConversationTurn user/assistant pairs.
189. Selected history is always a contiguous newest suffix of the available snapshot.
190. Current content, ReplyContext, Persona, Nickname, and system policy are never removed by history budgeting.
191. Trimmed turns remain persisted and remain available to future requests and response modes.
192. Phase 4B has no summarization, summary persistence, or automatic database pruning.
193. ResponseRouter does not inspect history or context-budget state.
194. Context-budget diagnostics must never log conversation content, display names, prompts, ReplyContext bodies, or secrets.
195. One logical model response may map to multiple Discord messages while generation happens exactly once.
196. Every emitted Discord response chunk is at most 2000 Unicode characters.
197. The original assistant response is persisted; transport-modified chunks are never persisted as assistant content.
198. Response splitting prefers paragraph, newline, sentence, whitespace, then hard-split boundaries.
199. A split crossing a standard fenced code block closes and reopens the fence with size accounting.
200. Every emitted Discord response chunk suppresses mentions.
201. The first response chunk replies to the triggering message; later chunks are channel sends.
202. Chunks from different logical responses must not interleave.
203. Discord delivery failure must never cause model generation to run again.
204. Output bodies and response chunk bodies must never be logged.
205. Phase 5A does not change ResponseRouter, conversation context, Persona, Nickname, SQLite schema, or LM Studio transport.
206. `/context clear` is guild-scoped and requires Administrator or Manage Server permission using the same runtime policy as Persona administration.
207. A successful context clear is publicly announced in the configured chat channel with mentions suppressed.
208. Context-clear permission, channel, and service failures are ephemeral interaction responses.
209. Context clear advances a persistent conversation epoch and never deletes prior conversation rows.
210. Each generation atomically captures the active epoch and recent snapshot at serialized generation start.
211. A completed turn is persisted with the captured epoch used for its model context.
212. A pre-clear in-flight completion may be retained under its old epoch but never re-enters the active in-memory snapshot.
213. Startup loads only the configured channel and active conversation epoch into memory.
214. Phase 4B history budgets apply only to the captured active epoch and remain unchanged by context clear.
215. An explicitly resolved ReplyContext remains current-request enrichment even when it references a pre-clear Discord message.
216. Context clear does not mutate Persona or Nickname state.
217. Context clear never enters the generation queue and never calls LM Studio.
218. The persistent conversation-state commit completes before the active in-memory epoch and snapshot are updated.
219. Even an empty context clear always advances the epoch monotonically.
220. A public Discord response failure after a committed clear never rolls back or retries the clear.
221. Conversation bodies, Persona/Nickname text, and secrets must never be logged during context clear or schema migration.
222. `/status` and `/health` require Administrator or Manage Guild permission.
223. Both runtime control-plane commands execute only in `LIUER_PERSONA_STATUS_CHANNEL_ID`.
224. All status/health responses are ephemeral and mention-suppressed.
225. `/status` is local-only and performs no HTTP request or active database health probe.
226. `/health` is read-only and may actively probe SQLite and LM Studio `/v1/models`.
227. Neither runtime command enters the generation queue or invokes model generation.
228. Queue observability exposes only worker-alive, busy, and waiting-job counts, never job contents.
229. `/health` distinguishes LM unreachable, timeout, auth error, HTTP error, invalid response, and configured-model missing.
230. Exact configured model matching remains required; health never auto-selects another model.
231. A busy or non-empty generation queue alone is not an unhealthy runtime state.
232. Health probes never modify Persona, Nickname, context epoch, history, queue, or model state.
233. `/status` reads conversation epoch and recent-turn count atomically from in-memory service state.
234. LM Studio tokens, response bodies, prompts, Persona text, and conversation contents are never exposed in command output, logs, or DTO reprs.
235. Runtime status/health commands never update the canonical Persona status-board message.
236. Phase 5C introduced no SQLite migration; Phase 5C's final schema was v5.
237. Phase 6A migrates conversation storage transactionally to schema v6 with
    `conversation_turns.author_kind` defaulting existing rows to `HUMAN`.
238. Every current and persisted conversation request carries explicit
    `AuthorKind.HUMAN` or `AuthorKind.BOT`; display names never infer speaker kind.
239. Historical HUMAN context uses the existing user label; historical BOT context
    uses an explicit Bot label, and Bot messages remain Chat Completions `user`
    content rather than assistant messages.
240. Current Bot turns, including image-only turns, reuse the existing response
    router, image preparation, serialized generation queue, persistence, and
    safe Discord delivery without inventing content.
241. Other Bot messages remain ignored unless an Administrator or Manage Guild
    member activates the exact message context menu `讓六耳回覆此 Bot`.
242. BotChat activation validates guild, configured chat channel, bot author,
    non-self author, non-webhook origin, and usable text or supported images.
243. BotChat has one process-local active peer, exact peer-ID matching, six
    maximum successful Liuer replies including the opening reply, and a lazy
    300-second idle timeout; the session is never persisted to SQLite.
244. BotChat has no `/botchat start`; `/botchat stop` and `/botchat status` are
    admin-only ephemeral controls, and status exposes metadata only.
245. A selected Bot message is the opening turn and receives the first public
    reply; active peer messages auto-trigger without a leading address, while
    Human routing, self messages, other Bots, threads, and webhooks retain their
    existing fail-closed behavior.
246. A BotChat turn is gated by `awaiting_peer`; at most one peer turn can be
    claimed at once. Generation or Discord delivery failure closes the session
    and never retries or regenerates the model output.
247. `/context clear` closes BotChat only after its persistent conversation epoch
    commit succeeds; it does not cancel an already accepted generation.
248. The single canonical Persona status board now uses Traditional Chinese
    guidance and titles `六耳 Persona — 預設 Persona` or `六耳 Persona — 自訂 Persona`.
249. Phase 6A adds no config fields or dependencies, no Bot allowlist, no session
    table, no automatic peer discovery, no unlimited loop, and no new queue.
250. Typing begins only after a message is accepted for model processing; ignored
    or unaddressed messages never acquire a typing lease.
251. Typing covers request preparation, queue wait, generation, and successful
    current-turn persistence, then ends before normal Discord answer delivery.
252. Accepted overlapping requests share one runtime-owned reference-counted
    typing mechanism; one completion cannot stop typing while another lease is
    active.
253. Typing API failure is best-effort and never fails or regenerates the
    underlying model request.
254. Typing state is process-local and is never persisted.
255. BotChat turn-gate rejected peer messages do not acquire typing.
256. Typing manager logs and reprs contain no conversation, prompt, Persona,
    nickname, image, or secret data.
257. Phase 6B does not modify routing, conversation context, SQLite schema, model
    transport, or generation concurrency.
258. LLM provider selection is explicit and fixed for the process lifetime.
259. LM Studio and Venice share provider-neutral Discord, context, persistence,
    generation-queue, typing, and response-delivery boundaries.
260. Venice generation uses `/chat/completions`; LM Studio retains its native
    text-only `CHAT_SHORT` path.
261. Every Venice generation request disables Venice's default system prompt.
262. Venice web search, scraping, citation, and X-search augmentation remain
    explicitly disabled.
263. Venice characters are never used to implement or replace 六耳 Persona.
264. Venice production model selection requires exact configured TEXT and VISION
    model IDs; they may be equal.
265. The active provider's non-generation model preflight completes before
    Discord login.
266. Venice has independent configured TEXT and VISION model roles. Only current
    prepared images select VISION; historical image and ReplyContext markers do
    not. Text-only Human and BotChat requests use TEXT, which need not support
    vision. VISION must satisfy current vision and multi-image requirements.
267. Provider reasoning content is never surfaced, persisted, or logged.
268. Conversation state remains owned by liuer-bot SQLite/runtime; provider
    state is never authoritative.
269. `/status` is provider-neutral, local-only, and performs zero HTTP probes.
270. `/health` probes only the active provider and never generates model output.
271. Provider/API tokens are secrets and never appear in output, logs, reprs,
    status, health, or doctor diagnostics.
272. Phase 7A adds no generation retry, failover, or rate-limit scheduling.
273. Phase 7A keeps SQLite schema v6 and introduces no provider-specific tables.
274. Phase 7A introduces no Ubuntu deployment behavior.
275. One Venice `/models` response validates both exact model roles at startup;
    `/health` also probes Venice once and evaluates both roles.
276. Venice TEXT and VISION roles may use the same exact model ID. Neither role
    automatically substitutes for the other after a failure.
277. Persona, context, generation queue, and persistence remain independent of
    Venice model role; Phase 7A retains SQLite schema v6.
