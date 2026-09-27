# JanSahayak — Master Implementation Plan

## Top-Level Overview

**Goal:** Build a production-grade, multilingual Indian government-scheme voice agent called JanSahayak — entirely from scratch in custom Python without any voice-agent framework.

**Scope:** Full 16-phase implementation from raw audio transport through Sarvam STT/LLM/TTS integration, session memory, deterministic tool calling, barge-in, multilingual switching, evaluation harness, and deployment to Render.

**Non-Negotiable Constraints:**
- Zero frameworks: No Pipecat, LiveKit, Retell, Vapi, LangChain, or LangGraph
- Core stack: FastAPI + pure asyncio + WebSockets + sarvamai==0.1.34
- Audio pipeline: Browser AudioWorklet → 16kHz mono LINEAR16 PCM → binary WebSocket frames
- LLM orchestration: Native Sarvam tool calling only — no separate intent classifier
- Single Uvicorn worker (session state is in-process memory)
- Python 3.12, sarvamai pinned to exactly 0.1.34

**Execution Rule for First Coding Batch:** After plan approval, coding begins with **Phase 1 & Phase 2 only** (FastAPI WebSocket + frontend audio capture). No Sarvam STT, LLM, or TTS code is written until raw bidirectional audio transport is verified end-to-end.

---

## Repository Structure

```
jansahayak/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── websocket/
│   │   ├── handler.py
│   │   └── protocol.py
│   ├── audio/
│   │   ├── buffer.py
│   │   ├── playback.py
│   │   └── generation.py
│   ├── sarvam/
│   │   ├── client.py
│   │   ├── stt.py
│   │   ├── llm.py
│   │   └── tts.py
│   ├── agent/
│   │   ├── manager.py
│   │   ├── state.py
│   │   ├── prompts.py
│   │   └── tools.py
│   ├── knowledge/
│   │   ├── retriever.py
│   │   └── schemas.py
│   ├── persistence/
│   │   └── callback_store.py
│   ├── evaluation/
│   │   ├── runner.py
│   │   ├── scenarios.py
│   │   └── metrics.py
│   └── logging/
│       └── events.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── audio-capture.js
│   ├── audio-playback.js
│   └── style.css
├── knowledge/
│   ├── pm_kisan.json
│   ├── ayushman_bharat.json
│   ├── pm_awas.json
│   ├── atal_pension.json
│   └── mgnrega.json
├── evaluation/
│   ├── scenarios.json
│   └── results.json
├── tests/
│   ├── test_retriever.py
│   ├── test_eligibility.py
│   ├── test_callback.py
│   ├── test_state.py
│   ├── test_tool_validation.py
│   ├── test_context.py
│   ├── test_generation.py
│   └── test_audio_buffer.py
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Sub-Tasks

---

### Phase 1 — Project Foundation & FastAPI WebSocket Server

**Status:** `[x] done`

**Intent:**
Establish the project skeleton, dependency manifest, environment config, and a running FastAPI server that accepts concurrent WebSocket connections with no AI logic. This is the foundation every subsequent phase builds on.

**Expected Outcomes:**
- `requirements.txt` and `.env.example` committed
- `backend/main.py` starts cleanly with `uvicorn backend.main:app --reload --port 8000`
- `GET /healthz` returns `{"status": "ok"}` without calling any Sarvam API
- `WS /ws` accepts a connection, echoes a `session_ready` JSON message, and handles disconnect cleanly
- Multiple simultaneous WebSocket connections are each assigned an independent `session_id`
- All async operations run inside the `asyncio` event loop — no blocking calls

**Todo List:**
1. Create `requirements.txt` pinning: fastapi, uvicorn[standard], sarvamai==0.1.34, pydantic>=2,<3, python-dotenv, pytest, pytest-asyncio, httpx
2. Create `.env.example` with all variables from spec Section 103
3. Create `.gitignore` — exclude `.env`, `.env.*` (except `.env.example`), `__pycache__`, `.venv`, `*.pyc`
4. Write `backend/config.py` — load all env vars via pydantic `BaseSettings`; fail fast if `SARVAM_API_KEY` is missing
5. Write `backend/websocket/protocol.py` — define all message types as typed dataclasses/Pydantic models (control, audio, client events, backend events as per spec Section 10)
6. Write `backend/websocket/handler.py` — `WebSocketHandler` class that creates `session_id = uuid4().hex`, manages connection lifecycle, dispatches message types, and sends `session_ready`
7. Write `backend/main.py` — FastAPI app, mount `frontend/` as StaticFiles at `/`, `GET /healthz`, `WS /ws` routing to handler
8. Create placeholder `frontend/index.html` with a "JanSahayak" heading and a Start button
9. Verify: `uvicorn backend.main:app --reload` starts, `/healthz` returns 200, WebSocket connects/disconnects without error

**Relevant Context:**
- Spec Sections 10, 11, 102–112, 114, 187
- `session_id = uuid4().hex` is the required session ID format
- Every event must carry `session_id`, `turn_id`, `generation_id`, `timestamp`
- Health check must NOT call Sarvam API

---

### Phase 2 — Frontend Audio Capture (AudioWorklet → PCM → WebSocket)

**Status:** `[x] done`

**Intent:**
Implement the complete browser-side audio capture pipeline: `getUserMedia` → `AudioWorklet` → mono / 16kHz / INT16 PCM → binary WebSocket frames. This phase ends when the backend can log received audio bytes continuously, proving the raw transport layer is solid before any AI is introduced.

**Expected Outcomes:**
- Clicking Start in the browser requests mic permission, resumes `AudioContext`, and opens WebSocket
- `AudioWorklet` processes audio in real time: stereo → mono, resample to 16kHz, Float32 → Int16
- Backend receives a continuous stream of binary WebSocket frames and logs byte counts
- Mic stays active during full-duplex scenarios (echo cancellation, noise suppression, AGC all enabled)
- No audio file upload — true streaming pipeline
- Frontend state machine cycles: IDLE → CONNECTING → LISTENING

**Todo List:**
1. Write `frontend/audio-capture.js`:
   - `AudioCaptureWorklet` processor that runs in the audio worklet thread
   - Accepts `Float32` samples, converts stereo to mono, resamples from device rate to 16kHz using linear interpolation, converts to Int16, posts `ArrayBuffer` to the main thread
2. Write `frontend/app.js`:
   - `JanSahayakApp` class managing WebSocket, AudioContext, and UI state machine
   - `start()` — `getUserMedia({echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1})`, resume AudioContext, connect worklet, open WebSocket
   - On binary frame from worklet → send as binary WebSocket frame
   - On JSON message from backend → dispatch to event handlers
   - Frontend state machine: IDLE, CONNECTING, LISTENING, PROCESSING, SPEAKING, INTERRUPTED, ERROR
3. Update `frontend/index.html` — Start/Stop button, status display, transcript panel, latency panel
4. Write `frontend/style.css` — clean minimal UI
5. Update `backend/websocket/handler.py` to accept binary frames and log `f"audio bytes received: {len(data)}"` (no AI processing yet)
6. Verify end-to-end: backend logs continuous audio byte counts while user speaks

**Relevant Context:**
- Spec Sections 7–10, 115, 175
- AudioWorklet is mandatory — no ScriptProcessor
- Device sample rate is commonly 44100 or 48000; must resample to exactly 16000
- Binary WebSocket frames only for audio (not base64)
- `getUserMedia` + `AudioContext.resume()` must be triggered by the user gesture (Start button click) to avoid browser autoplay restrictions

---

### Phase 3 — Sarvam Realtime STT Integration

**Status:** `[x] done`

**Intent:**
Connect the backend to Sarvam's `saaras:v3-realtime` WebSocket STT endpoint. Audio bytes flowing from the browser are forwarded to Sarvam STT, which returns partial and final transcripts along with VAD events. This phase ends when real speech is transcribed and partial/final events are logged and forwarded to the browser.

**Expected Outcomes:**
- `backend/sarvam/stt.py` manages a persistent WebSocket connection to Sarvam's STT endpoint
- Audio bytes are forwarded from the browser → backend STT sender task in real time
- Partial transcript events (`transcript_partial`) are sent to the browser
- Final transcript events (`transcript_final`) are sent to the browser
- VAD `speech_start` / `speech_end` events are received and logged
- Language auto-detection via `language_code="auto"` works
- STT WebSocket reconnects on transient failure; fatal errors surface gracefully

**Todo List:**
1. Write `backend/sarvam/client.py` — thin wrapper around `sarvamai` SDK providing authenticated base config; never exposes API key to the frontend
2. Write `backend/sarvam/stt.py` — `SarvamSTT` class:
   - Opens `saaras:v3-realtime` WebSocket at session start
   - Configuration: `model=saaras:v3-realtime`, `language_code=auto`, `stream_type=fast`, `endpointing=vad`, `encoding=linear16`, `sample_rate=16000`, `mode=transcribe`
   - `send_audio(bytes)` coroutine — forwards raw PCM
   - Receiver task parses `partial`, `final`, `vad.*`, `error` events
   - On `error`: classify fatal vs non-fatal; reconnect if non-fatal
   - Keepalive ping every 20–30 seconds (configurable)
3. Wire `SarvamSTT` into `WebSocketHandler` — audio from browser is queued and forwarded; transcript events are sent back as JSON
4. Write `backend/logging/events.py` — structured event logger emitting JSON with `session_id`, `turn_id`, `generation_id`, `event`, `timestamp`
5. Test: speak "PM Kisan kya hai?" — confirm partial + final transcript appears in browser and logs

**Relevant Context:**
- Spec Sections 5, 6, 58–60, 67, 69, 116, 215
- Do NOT use `language_code="unknown"` — use `"auto"`
- `stream_type=fast` is mandatory for low-latency partials
- STT socket opens at session start to avoid first-turn connection latency
- VAD thresholds (threshold=0.3, silence_duration_ms=400, min_speech_duration_ms=200) are initial values to tune

---

### Phase 4 — Basic LLM Integration (No Tools)

**Status:** `[ ] pending`

**Intent:**
Pass STT final transcripts to `sarvam-105b-conversations` and return plain text responses. No tool calling, no session memory yet. Validates that the STT → LLM → response chain works with correct latency instrumentation.

**Expected Outcomes:**
- `backend/sarvam/llm.py` calls the Sarvam V1 chat completions endpoint
- STT `transcript_final` triggers an LLM request with `reasoning_effort=None`
- Plain text response is received and sent to the browser as `agent_text`
- `t_llm_start` and `t_llm_first_token` timestamps are captured
- Responses are concise (max_tokens ~256 for spoken response)
- Basic system prompt establishes JanSahayak persona and scope

**Todo List:**
1. Write `backend/sarvam/llm.py` — `SarvamLLM` class:
   - `POST /v1/chat/completions` with `model=sarvam-105b-conversations`
   - `reasoning_effort=None`, `max_tokens=256`, `stream=True` for direct answers
   - `stream=False`, `tool_choice="auto"` for tool-routing requests (Phase 7)
   - Record `t_llm_start`, `t_llm_first_token` timestamps
2. Write `backend/agent/prompts.py` — base system prompt: JanSahayak role, scope, language behavior, grounding rules, voice conciseness requirement
3. Wire `SarvamLLM` into `WebSocketHandler`: `transcript_final` → `llm.chat()` → send `agent_text` event to browser
4. Add `GET /readyz` endpoint checking API key presence and KB loaded status
5. Test: "Hello" → JanSahayak greeting; "What is JanSahayak?" → descriptive answer; "PM Kisan?" → basic response

**Relevant Context:**
- Spec Sections 23–29, 117, 136
- `sarvam-105b-conversations` is the correct model — NOT `sarvam-105b`
- `reasoning_effort=None` is the latency-first setting for voice turns
- Tool-call LLM requests use `stream=False`; direct-answer requests use `stream=True`

---

### Phase 5 — Sarvam TTS Integration (Full Response)

**Status:** `[ ] pending`

**Intent:**
Connect `bulbul:v3` WebSocket TTS. LLM text responses are sent to TTS and audio chunks are forwarded to the browser as binary PCM frames. Browser `AudioWorklet` plays them back. This phase uses full-response TTS (no streaming overlap yet) to establish the complete audio loop.

**Expected Outcomes:**
- `backend/sarvam/tts.py` manages a persistent `bulbul:v3` WebSocket
- LLM text → TTS → audio chunks received as base64/LINEAR16 → decoded to raw PCM → sent as binary to browser
- `frontend/audio-playback.js` `AudioWorklet` receives PCM binary frames and plays them back
- Full voice conversation loop works: speak → transcribe → LLM → TTS → hear response
- `t_tts_start` and `t_tts_first_audio` timestamps captured
- TTS keepalive ping every 20–30 seconds implemented

**Todo List:**
1. Write `backend/sarvam/tts.py` — `SarvamTTS` class:
   - Opens `bulbul:v3` WebSocket TTS connection
   - Configuration: `language_code` from session state, `output_audio_codec=linear16`, `speech_sample_rate=16000`, `pace=1.0`, `min_buffer_size=50`, `max_chunk_length=200`, `send_completion_event=true`
   - `synthesize(text)` — sends text, receives audio chunk events
   - Decodes base64 audio → raw PCM bytes
   - Keepalive ping implementation
2. Write `frontend/audio-playback.js` — `PlaybackQueue` AudioWorklet:
   - Receives PCM binary frames from app.js via `port.postMessage`
   - Buffers and plays them in order without gaps
   - `clear()` method for immediate barge-in stop
   - Tracks `generation_id` to discard stale audio
3. Wire TTS into the pipeline: LLM response → `tts.synthesize()` → binary PCM to browser → playback
4. Update `app.js` to receive binary frames and forward to `PlaybackQueue`
5. Test Hindi, English, and Kannada TTS voices

**Relevant Context:**
- Spec Sections 50–55, 118, 139, 142
- Use LINEAR16 output codec for browser — avoids MP3 browser dependency
- Flow: `bulbul:v3` → base64 audio → backend decode → raw PCM → binary WebSocket → browser
- TTS socket opens at session start alongside STT socket

---

### Phase 6 — Streaming TTS with Sentence Chunking

**Status:** `[ ] pending`

**Intent:**
Replace full-response TTS buffering with sentence-level streaming. LLM tokens are chunked by sentence boundaries and dispatched to TTS immediately. This creates the overlapping pipeline — LLM generation + TTS generation + playback happen concurrently — and drives first-audio latency down.

**Expected Outcomes:**
- `backend/audio/generation.py` splits LLM token stream at sentence boundaries (`।`, `.`, `?`, `!`)
- First complete sentence is sent to TTS before LLM finishes generating
- First audio reaches the browser measurably faster than full-response mode
- No 5-word micro-chunks; no 1000-character giant chunks
- `t_tts_first_audio` latency is logged and displayed in the browser UI
- Latency dashboard in browser shows STT / LLM / TTS / E2E metrics

**Todo List:**
1. Write `backend/audio/generation.py` — `TextChunker`:
   - Consumes LLM streaming token iterator
   - Emits complete sentences (split on `.`, `।`, `?`, `!`, natural pauses)
   - Enforces max chunk length ~200 chars and min meaningful length
2. Update `backend/sarvam/llm.py` `stream=True` path to yield via `TextChunker`
3. Update TTS pipeline to accept chunked text stream: each chunk → TTS → audio forwarded immediately
4. Update `backend/audio/buffer.py` — `PlaybackQueue` with bounded size, ordered delivery, `generation_id` gating
5. Add latency timestamps: `t_speech_end_received`, `t_stt_final`, `t_llm_start`, `t_llm_first_token`, `t_tts_start`, `t_tts_first_audio`
6. Forward latency metrics to browser as `{"type": "latency", "metrics": {...}}`
7. Display latency dashboard in `frontend/index.html`
8. Measure and log first-audio latency improvement vs Phase 5

**Relevant Context:**
- Spec Sections 55, 56, 119, 129–142, 144
- Sentence chunking is the primary mechanism for overlapping LLM+TTS latency
- Sarvam TTS docs recommend chunks under ~500 chars for lowest latency
- Latency budget targets: STT < 0.6s, LLM first token < 0.8s, TTS first audio < 0.5s, E2E < 1.8–2.0s

---

### Phase 7 — Native Tool Calling + Knowledge Base

**Status:** `[ ] pending`

**Intent:**
Introduce the structured knowledge base and the first tool: `search_knowledge`. The LLM now acts as an orchestrator — it decides whether to call a tool or answer directly. Python executes the tool and passes results back to the LLM for the final response. This removes the need for any separate intent classifier.

**Expected Outcomes:**
- `knowledge/*.json` files loaded into memory at startup for 5 schemes
- `backend/knowledge/retriever.py` does keyword/topic-weighted local retrieval (no vector DB)
- `backend/agent/tools.py` defines `search_knowledge` as a Pydantic-validated tool
- LLM receives tool schema; on tool call, Python executes and returns result; LLM produces final answer
- "What is PM-KISAN?" triggers: LLM → `search_knowledge` → tool result → LLM final answer → TTS
- Tool selection is logged with `session_id`, `turn_id`, `tool`, `arguments`
- Hallucinated government facts are blocked — LLM only answers from tool results

**Todo List:**
1. Create `knowledge/pm_kisan.json`, `knowledge/ayushman_bharat.json`, `knowledge/pm_awas.json`, `knowledge/atal_pension.json`, `knowledge/mgnrega.json` with verified content (overview, eligibility info, documents, steps, FAQs, source URLs, `last_verified`)
2. Write `backend/knowledge/schemas.py` — `KnowledgeRecord` Pydantic model
3. Write `backend/knowledge/retriever.py` — in-memory load at startup, query normalization, scheme/topic filter, keyword scoring, return top 2–3 chunks
4. Write `backend/agent/tools.py` — tool function signatures with Pydantic argument models for `search_knowledge`; Pydantic validation before execution
5. Update `backend/sarvam/llm.py` — first call with `stream=False`, `tool_choice="auto"`, tool schemas in request; detect `tool_calls` in response; execute tool; second LLM call with tool result
6. Update `backend/agent/manager.py` to orchestrate: transcript → tool-routing LLM → (optional) tool exec → final LLM → TTS
7. Add `agent_status` events to browser: `"searching_knowledge"`, `"thinking"`, `"responding"`
8. Test: "PM Kisan kya hai?" — verify tool is called, result used, no hallucination

**Relevant Context:**
- Spec Sections 23–25, 32–33, 78–80, 97, 100, 120
- Maximum 1 tool call per user turn in this phase
- Tool arguments are JSON strings — parse + Pydantic validate before execution
- `search_knowledge` is read-only; tool result length is bounded (top 2–3 chunks)

---

### Phase 8 — Session Memory & Conversation State

**Status:** `[ ] pending`

**Intent:**
Introduce `ConversationState` with structured state (current scheme, workflow, collected slots, language) and bounded conversation history. Context follow-up now works — "What documents do I need?" uses the active `current_scheme` without the user repeating it.

**Expected Outcomes:**
- `backend/agent/state.py` defines `ConversationState` with all required fields
- `current_scheme`, `current_workflow`, `response_language` are updated from tool results and LLM responses
- Conversation history is pruned to last `MAX_RECENT_MESSAGES` (default 24) turns
- LLM system message includes compact structured state injection on every turn
- Follow-up queries work correctly across a 6-turn conversation
- Only finalized turns enter history — no partial STT, no partial LLM tokens
- Interrupted assistant responses are flagged `assistant_response_status=interrupted` and do not become canonical history
- `generation_id` is stored in state and incremented on every new agent response or interruption

**Todo List:**
1. Write `backend/agent/state.py` — `ConversationState` dataclass with: `session_id`, `input_language`, `response_language`, `current_scheme`, `current_workflow`, `required_slots`, `collected_slots`, `conversation_history`, `generation_id`, `active_response`, `active_tts`, `active_llm`, `metrics`
2. Update `backend/agent/prompts.py` — `build_system_message(state)` injects compact structured state block
3. Update `backend/agent/manager.py` — update state after each turn; update `current_scheme` when `search_knowledge` is called; track `generation_id`
4. Implement conversation history pruning: keep last `MAX_RECENT_MESSAGES` turns; structured state persists regardless
5. Ensure only `transcript_final` messages enter history (not partials)
6. Test context follow-up scenarios: PM-KISAN → "What documents?" → "How do I apply?" — all must use `current_scheme=pm_kisan`

**Relevant Context:**
- Spec Sections 14–21, 121, 163–164, 231
- Two types of memory: structured state (authoritative) + conversation history (conversational context)
- Do NOT summarize turns with a second LLM call — structured state + recent history is sufficient
- Session memory is in-process; no cross-session persistence required for take-home

---

### Phase 9 — Eligibility Tool (Deterministic Rules)

**Status:** `[ ] pending`

**Intent:**
Add the `check_eligibility` tool with deterministic Python business rules. The LLM collects user information via multi-turn dialogue but Python makes the final eligibility decision. LLM never independently decides eligibility.

**Expected Outcomes:**
- `check_eligibility` tool is defined with scheme-specific slot requirements
- LLM asks for missing slots (state, age, land ownership, income, etc.) before calling the tool
- Python evaluates eligibility deterministically — result is not a guess
- `collected_slots` in `ConversationState` accumulates across turns until all required slots are filled
- Missing-slot scenario, eligible scenario, and ineligible scenario all work correctly

**Todo List:**
1. Define eligibility criteria for each scheme in `knowledge/*.json` (required fields, rules, thresholds)
2. Add `check_eligibility(scheme, collected_slots)` to `backend/agent/tools.py` with Pydantic argument validation
3. Write deterministic eligibility evaluation functions for each scheme — Python is the authority
4. Update system prompt with eligibility behavior: LLM collects slots → calls tool → explains result; LLM never decides eligibility independently
5. Update `ConversationState` to track `current_workflow=ELIGIBILITY` and `required_slots` / `collected_slots`
6. Test: "PM Kisan ke liye eligible hoon?" → missing-slot prompt → "UP se hoon" → "Age 45" → deterministic result + LLM explanation

**Relevant Context:**
- Spec Sections 34, 83–85, 122, 224
- LLM is the dialogue manager, Python is the eligibility judge
- Slot filling happens over multiple turns — state persists in `ConversationState.collected_slots`

---

### Phase 10 — Documents & Application Steps Tools

**Status:** `[ ] pending`

**Intent:**
Add `get_required_documents` and `get_application_steps` tools. These are read-only and return verified data from the knowledge base. Source attribution is included in the response.

**Expected Outcomes:**
- "Documents kaunse chahiye?" uses `current_scheme` context and returns verified list
- "How do I apply?" returns official application steps with source URL
- Both tools follow the same Pydantic validation and bounded-result patterns as Phase 7
- `current_workflow` transitions to `DOCUMENTS` or `APPLICATION` appropriately

**Todo List:**
1. Add `get_required_documents(scheme)` and `get_application_steps(scheme)` to `backend/agent/tools.py`
2. Populate `documents` and `application_steps` in all 5 knowledge JSON files from official sources
3. Update `agent/manager.py` to handle `DOCUMENTS` and `APPLICATION` workflow states
4. Test: PM-KISAN → "Documents?" (uses context) → correct list with source; → "How to apply?" → steps

**Relevant Context:**
- Spec Sections 35–36, 86, 123
- Source URL and `last_verified` date must be present in tool results
- Return value is bounded — not the entire JSON file

---

### Phase 11 — Language Switching

**Status:** `[ ] pending`

**Intent:**
Implement `set_language` tool and full multilingual behavior. Explicit language-switch instructions update `response_language`; TTS reconnects for the new language. Crucially, all conversation context and scheme/workflow state survives the language switch.

**Expected Outcomes:**
- "English mein continue karo" → `set_language({language_code: "en-IN"})` → TTS reconnects → next response in English
- "Kannada dalli heli" → Kannada response about the same scheme
- `input_language` and `response_language` are tracked separately
- Automatic language mirroring does NOT flip the language on every Hinglish sentence
- Demo scenario works: Hindi → English → Kannada with PM-KISAN context intact throughout
- `language_changed` event is sent to browser UI

**Todo List:**
1. Add `set_language(language_code)` to `backend/agent/tools.py` — updates `state.response_language`
2. Update `backend/sarvam/tts.py` — on language change, close current TTS socket and open new one for new language
3. Update system prompt to specify: respond in `response_language`; only use `set_language` on explicit user instruction; do not auto-flip on Hinglish
4. Update browser UI to display current language
5. Update TTS speaker configuration per language (Hindi: shubh, Kannada: specific voice)
6. Test language-switch scenario: Hindi → English → Kannada → Hindi; verify scheme/workflow context preserved throughout
7. Verify: STT `language_code=auto` correctly detects Kannada and Hinglish

**Relevant Context:**
- Spec Sections 44–48, 90–91, 124
- TTS socket must be reopened — not merely reconfigured — on language change
- Hinglish is a code-mixed mode, not a separate `language_code`
- Explicit language switch only; preserve the user's last explicit language choice

---

### Phase 12 — Callback Workflow (SQLite + Idempotency)

**Status:** `[ ] pending`

**Intent:**
Implement `get_callback_slots` and `book_human_callback` tools backed by SQLite. Booking is transactionally atomic to prevent race conditions. Idempotency key (session_id + turn_id + tool_call_id) prevents duplicate bookings on network retries.

**Expected Outcomes:**
- `get_callback_slots` returns available slots for a given date
- `book_human_callback` atomically reserves a slot and returns a booking reference
- Concurrent booking of the same slot fails gracefully for the second request
- Idempotent re-submission of the same booking returns the original reference
- LLM never claims "your callback is booked" unless the tool returns success
- UI clearly labels this as a demo booking system

**Todo List:**
1. Write `backend/persistence/callback_store.py` — SQLite schema: `callback_slots (slot_id, date, start_time, end_time, booked, booking_reference)` and `bookings (idempotency_key, booking_reference, created_at)`
2. Seed demo slot data on application startup
3. Implement `book_human_callback` with atomic `UPDATE ... WHERE booked=0` and rowcount check
4. Implement idempotency: check idempotency key before booking; return existing reference if already booked
5. Add `get_callback_slots` and `book_human_callback` to `backend/agent/tools.py` with full Pydantic validation
6. Update `ConversationState` — `current_workflow=CALLBACK`
7. Test: successful booking → reference returned; duplicate request → same reference; race condition → second attempt fails with friendly error

**Relevant Context:**
- Spec Sections 38–41, 87–89, 125
- `book_human_callback` is the only state-changing tool (besides `set_language`)
- SQLite is the correct choice — no external DB needed for take-home
- Never claim success without tool confirmation

---

### Phase 13 — Barge-In (Generation ID + TTS Close/Reopen)

**Status:** `[ ] pending`

**Intent:**
Implement full barge-in: browser-local VAD detects user speech during TTS playback → stops playback immediately → clears `PlaybackQueue` → increments `generation_id` → backend closes TTS socket → new conversation turn begins. Any delayed audio from the old generation is discarded.

**Expected Outcomes:**
- User can interrupt agent mid-speech and the old audio stops within ~100–200ms
- New user speech is captured cleanly while old TTS is being cancelled
- Old generation's audio never plays after interruption
- New LLM turn begins immediately after `transcript_final`
- Barge-in works during: short answers, long answers, tool-response answers, in Hindi/English/Kannada
- `barge_in` event is sent to backend; backend increments `generation_id`, cancels active tasks, closes TTS socket

**Todo List:**
1. Implement browser-side local VAD in `audio-capture.js` — energy-threshold RMS detection on audio worklet output; fires `barge_in` event when speech detected during SPEAKING state
2. Update `app.js` — on `barge_in`: stop playback immediately, `PlaybackQueue.clear()`, send `{"type": "client_event", "event": "barge_in"}` to backend, set state to INTERRUPTED
3. Update `backend/websocket/handler.py` — on `barge_in` event: cancel active LLM/TTS tasks, increment `state.generation_id`, close TTS socket
4. Update `backend/audio/playback.py` — `generation_id` check on every PCM chunk sent to browser; stale chunks are silently dropped
5. Update `backend/sarvam/tts.py` — `cancel()` method: closes current WebSocket; `open_new()` reopens for next response
6. Test barge-in at least 10 times: mid-sentence, beginning, end, during tool responses; verify no old audio bleeds through

**Relevant Context:**
- Spec Sections 62–65, 126, 143, 214, 230
- Sarvam TTS has NO in-band cancel — close and reopen the socket
- Local VAD provides immediate browser-side response; Sarvam VAD is authoritative for STT endpointing
- `generation_id` is session-local and monotonically increasing

---

### Phase 14 — Latency Instrumentation & Optimization

**Status:** `[ ] pending`

**Intent:**
Add comprehensive timestamp collection at every pipeline stage and expose P50/P95 metrics. Use measurements to drive targeted optimizations (VAD tuning, TTS chunk size, LLM output length, prompt size). Report actual measured numbers — not assumed ones.

**Expected Outcomes:**
- All backend timestamps collected: `t_speech_end_received`, `t_stt_final`, `t_llm_start`, `t_llm_first_token`, `t_tool_start`, `t_tool_end`, `t_final_llm_first_token`, `t_tts_start`, `t_tts_first_audio`
- Browser timestamps collected via `performance.now()`: `client_speech_end`, `client_playback_started`
- Per-turn latency breakdown sent to browser as `{"type": "latency", "metrics": {...}}`
- P50/P95 computed across 20+ test turns and stored in `evaluation/results.json`
- VAD silence duration tuned across 300ms, 350ms, 400ms, 500ms and best value selected from measurement
- Latency dashboard displays STT, LLM, Tool, TTS, E2E columns in real time

**Todo List:**
1. Update all pipeline stages to record timestamps and attach to `ConversationState.metrics`
2. Forward complete latency breakdown as `latency` WebSocket event after each turn
3. Update `frontend/index.html` latency dashboard to show real values
4. Run 20+ spoken test turns and collect raw timings
5. Tune `STT_SILENCE_DURATION_MS` across 300/350/400/500ms — pick lowest that doesn't cut off speech
6. Confirm `reasoning_effort=None` vs default — measure LLM TTFT difference
7. Tune TTS `min_buffer_size` and `max_chunk_length` — measure first-audio latency
8. Update `backend/evaluation/metrics.py` — `compute_percentiles(measurements)` for P50/P95

**Relevant Context:**
- Spec Sections 57, 130–145, 127
- Client E2E TTFA = `client_playback_started - client_speech_end` (same clock)
- Target budget: E2E < 2.0s for simple queries, < 3.0s for tool queries
- Report measured numbers only — do not state targets as achievements

---

### Phase 15 — Evaluation Harness

**Status:** `[ ] pending`

**Intent:**
Build a scripted evaluation harness that runs 50–75 test scenarios covering all capabilities. Produces `evaluation/results.json` with tool selection accuracy, workflow completion rate, language adherence, context follow-up accuracy, barge-in success rate, and latency percentiles. All metrics must come from actual test runs.

**Expected Outcomes:**
- `evaluation/scenarios.json` contains 50–75 cases across: FAQ, scheme discovery, eligibility, documents, application, callback, language switch, Hinglish, barge-in, unsupported, failure, concurrent
- `backend/evaluation/runner.py` can execute scenarios against local server using pre-recorded audio or text injection
- `backend/evaluation/metrics.py` computes all required metrics from run output
- `evaluation/results.json` is populated with real measured values (no nulls unless genuinely not yet measured)
- Session isolation is verified: 3 concurrent sessions do not cross-contaminate state

**Todo List:**
1. Write `evaluation/scenarios.json` — 50–75 cases with fields: `id`, `category`, `language`, `turns`, `expected_workflow`, `expected_tool`, `expected_scheme`
2. Write `backend/evaluation/scenarios.py` — load and validate scenario file
3. Write `backend/evaluation/runner.py` — text-injection path (bypasses STT for automated testing); voice-injection path (uses pre-recorded PCM/WAV for Level 3 tests)
4. Write `backend/evaluation/metrics.py` — `tool_selection_accuracy`, `tool_argument_accuracy`, `workflow_completion_rate`, `language_adherence`, `context_followup_accuracy`, `barge_in_success_rate`, `compute_percentiles`
5. Write all unit tests in `tests/` for: retriever, eligibility rules, callback booking, state transitions, tool argument validation, context pruning, generation ID, audio buffer
6. Run full evaluation suite locally; populate `evaluation/results.json`
7. Test 3 concurrent sessions: PM-KISAN vs Ayushman Bharat vs callback — verify state isolation

**Relevant Context:**
- Spec Sections 128, 146–165
- Level 1 (unit), Level 2 (LLM/tool integration), Level 3 (voice injection) tests are all required
- Manual review is acceptable for knowledge grounding metric in a small dataset
- Do not write evaluation report until actual numbers exist

---

### Phase 16 — Deployment (Docker + Render + Production Validation)

**Status:** `[ ] pending`

**Intent:**
Containerize the application, deploy to Render Web Service with HTTPS and WSS, run the full evaluation suite against the production URL, and capture production latency metrics. The submission is only complete when all demo scenarios work on the live deployed URL.

**Expected Outcomes:**
- `Dockerfile` builds successfully and runs locally with `--env-file .env`
- Render Web Service deploys from private GitHub repo
- `https://<app>.onrender.com` serves the frontend
- `wss://<app>.onrender.com/ws` accepts WebSocket connections
- `/healthz` returns 200 on Render health probe
- All 6 demo scenarios work on the live URL
- Production latency is measured and recorded (distinct from local latency)
- `README.md` contains complete setup instructions, demo video link, evaluation write-up, and tradeoff note

**Todo List:**
1. Write `Dockerfile` — `python:3.12-slim`, install `requirements.txt`, `CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}`
2. Write `README.md` — setup instructions, environment variables, local run, Docker run, hosted URL, evaluation summary
3. Test Docker locally: `docker build`, `docker run --env-file .env`, verify full voice loop
4. Create private GitHub repository; push all code; grant access to `jayant@sarvam.ai` and `avisolanki@sarvam.ai`
5. Connect repo to Render Web Service; set build command `pip install -r requirements.txt`; set start command `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`; add all env vars; configure `/healthz` health check
6. Deploy; check deploy logs; run production smoke test (10-step checklist from spec Section 202)
7. Run evaluation suite against production URL; record production latency P50/P95
8. Record demo video showing all 6 demo scenarios
9. Write evaluation write-up and tradeoff note; add to `README.md`
10. Verify: no `SARVAM_API_KEY` in browser, no `ws://` on HTTPS page, no committed `.env`

**Relevant Context:**
- Spec Sections 129, 189–209
- Single Uvicorn worker — session state is in-memory; multiple workers would break session isolation
- Render `wss://` required (not `ws://`) when page is served over HTTPS
- Freeze final `requirements.txt` with `pip freeze` before Docker build

---

## Architectural Constraints Summary

| Constraint | Enforcement |
|---|---|
| No Pipecat/LiveKit/Vapi/Retell | Not in requirements.txt; no imports allowed |
| No LangChain | Not in requirements.txt |
| No separate intent classifier | Single LLM call with `tool_choice="auto"` |
| No vector DB | In-memory keyword retrieval only |
| No full-response TTS buffering | TextChunker → sentence-level dispatch (Phase 6) |
| No LLM as eligibility authority | Python deterministic rules (Phase 9) |
| No unbounded history | `MAX_RECENT_MESSAGES=24` prune (Phase 8) |
| No blocking calls in event loop | All I/O via `asyncio`; no `requests` library |
| No API key in browser | Backend-only Sarvam calls |
| No `ws://` on HTTPS | `wss://` enforced in `app.js` |
| Single Uvicorn worker | Documented in `README.md` |
| `sarvamai==0.1.34` pinned | Exact version in `requirements.txt` |

---

## First Coding Batch

After this plan is approved, implementation begins with **Phase 1 and Phase 2 only**.

**Phase 1** produces a running FastAPI server with WebSocket session management, health endpoint, and protocol types.

**Phase 2** produces a browser AudioWorklet that captures microphone audio, converts it to 16kHz mono LINEAR16 PCM, and streams binary frames to the backend — which logs byte counts.

**No Sarvam STT, LLM, or TTS code is written until the backend logs continuous audio bytes from the browser and the bidirectional transport is confirmed stable.**
