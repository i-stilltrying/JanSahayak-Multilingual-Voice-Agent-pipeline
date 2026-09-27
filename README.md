# JanSahayak — Multilingual Citizen-Service Voice Agent

> A production-grade, framework-free voice assistant that helps Indian citizens discover and access government welfare schemes entirely through natural speech. Built exclusively with custom Python `asyncio` and Sarvam AI models — zero reliance on LiveKit, Pipecat, LangChain, or any other voice/orchestration framework.

---

## Table of Contents

1. [Key Architectural Highlights](#key-architectural-highlights)
2. [Supported Government Schemes](#supported-government-schemes)
3. [Quick Start — Local](#quick-start--local)
4. [Quick Start — Docker](#quick-start--docker)
5. [Entering Your API Key](#entering-your-api-key)
6. [Testing the Agent — Evaluation Suite](#testing-the-agent--evaluation-suite)
7. [Project Structure](#project-structure)
8. [Environment Variables](#environment-variables)

---

## Key Architectural Highlights

### Framework-Free Custom Pipeline

Every byte of the audio pipeline is custom Python `asyncio`. The browser captures 16 kHz mono LINEAR16 PCM via an `AudioWorklet`, streams it as binary WebSocket frames to a FastAPI backend, and receives synthesized PCM back over the same connection — with no framework magic in between.

```
Browser AudioWorklet (16 kHz PCM capture)
    ↓  binary WebSocket frames
FastAPI / asyncio WebSocket handler
    ↓
Sarvam saaras:v3-realtime   ← real-time STT with VAD
    ↓  transcript
sarvam-105b-conversations   ← LLM with native tool calling
    ↓  two-call tool pattern (orchestrate → dispatch → stream)
Python tool layer           ← KB retrieval, eligibility, callbacks
    ↓  final text
Sarvam bulbul:v3            ← WebSocket streaming TTS
    ↓  PCM audio chunks
Browser PlaybackQueue AudioWorklet
```

### Sarvam AI Core Models

| Role | Model |
|---|---|
| Speech-to-Text | `saaras:v3-realtime` (WebSocket streaming, VAD endpoint) |
| Language Model | `sarvam-105b-conversations` (native function calling) |
| Text-to-Speech | `bulbul:v3` (WebSocket streaming, per-language speakers) |

### Real-Time Barge-In

A custom `generation_id` integer is incremented on every user interruption. Every in-flight LLM streaming task and TTS context manager checks this value at each token/chunk; a mismatch causes an immediate, clean abort without any shared locks or flags. Partial responses are never committed to conversation history.

### Multilingual Script Enforcement

The system prompt injects a **CRITICAL LANGUAGE OVERRIDE** directive as its final instruction every turn, naming the active BCP-47 code and its native script rule. Supported languages:

| Language | Code | Script Rule |
|---|---|---|
| Hindi | `hi-IN` | Devanagari — romanization strictly prohibited |
| English | `en-IN` | Plain English only |
| Kannada | `kn-IN` | Kannada script — romanization strictly prohibited |
| Tamil | `ta-IN` | Auto |
| Telugu | `te-IN` | Auto |
| Malayalam | `ml-IN` | Auto |
| Marathi | `mr-IN` | Auto |
| Bengali | `bn-IN` | Auto |
| Gujarati | `gu-IN` | Auto |

Code-mixed (Hinglish) input is handled correctly — the STT detects the language and the response mirrors it.

### Deterministic Eligibility Engine

The LLM is **never** the eligibility decision-maker. It extracts slot values from conversation and passes them to a Python rule engine. The engine evaluates hard-coded business rules per scheme and returns one of three typed states: `ELIGIBLE`, `INELIGIBLE`, or `MISSING_DATA` (with a precise list of required fields). The LLM only translates the typed result into empathetic speech.

### Keyword-Based In-Memory RAG

Five scheme JSON files are loaded into memory at startup. Retrieval uses a weighted keyword-scoring function (no embeddings, no vector DB) that returns the top-3 focused knowledge chunks in under 1 ms. Results are deterministic and fully auditable.

### SQLite Callback Booking

Human government-officer callbacks are backed by a local SQLite database. Booking is an atomic `UPDATE … WHERE booked = 0` transaction — race conditions are handled correctly and every booking attempt carries an idempotency key to safely handle LLM retries.

---

## Supported Government Schemes

| Scheme | ID |
|---|---|
| PM-KISAN (Pradhan Mantri Kisan Samman Nidhi) | `pm_kisan` |
| Ayushman Bharat PM-JAY | `ayushman_bharat` |
| PM Awas Yojana (PMAY) | `pm_awas` |
| Atal Pension Yojana | `atal_pension` |
| MGNREGA | `mgnrega` |

---

## Quick Start — Local

### Prerequisites

- Python 3.12+
- A [Sarvam AI](https://www.sarvam.ai/) API key

### Setup

```bash
git clone <repo-url>
cd JanSahayak

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Run

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in Chrome or Edge.

> **No `.env` file required.** See [Entering Your API Key](#entering-your-api-key) below.

---

## Quick Start — Docker

```bash
docker build -t jansahayak .
docker run --rm -p 8000:8000 jansahayak
```

Open **http://localhost:8000** in Chrome or Edge.

> The Docker image does not require any environment variables to be passed at runtime. The API key is entered securely through the UI dashboard.

---

## Entering Your API Key

**You do not need to create or edit any `.env` file.**

The application is designed for reviewer access. When you open `http://localhost:8000`, the dashboard displays a **"Sarvam API Key (Reviewer)"** input field at the top of the control panel. Simply:

1. Paste your Sarvam API key into the field.
2. Click **🎙️ Start Conversation**.

Your key is sent securely over the local WebSocket connection with the `start_session` handshake and used exclusively for that session. It is also saved to `localStorage` so you do not need to re-enter it on the next page load.

---

## Testing the Agent — Evaluation Suite

The UI includes a built-in **Reviewer Evaluation Guide** panel on the right side of the screen. It contains a scripted 8-turn test suite designed to exercise every major capability of the agent. Reviewers are encouraged to follow the turns in order:

| Turn | Capability Tested | Example Prompt |
|---|---|---|
| 1 | Proactive greeting & `search_knowledge` tool dispatch | *"Tell me about PM Awas Yojana and what benefits it provides."* |
| 2 | Barge-in interruption & language switch | *[Interrupt while agent speaks]* *"Mujhe Hindi mein samjhao."* |
| 3 | Cross-turn context memory | *"Iske liye kaun-kaun se documents chahiye?"* |
| 4 | Cross-scheme context pivot | *"What are the eligibility criteria for PM-KISAN?"* |
| 5 | Code-mixed (Hinglish) input | *"MNREGA ka application form kaise bharte hain?"* |
| 6 | Entity extraction & status lookup | *"I want to check my application status. My ID is APP4."* |
| 7 | Workflow reset & Devanagari TTS | *"Mujhe Atal Pension Yojana ke baare mein batao."* |
| 8 | Callback booking workflow | *"I want to book an officer callback."* |

### Real-Time Latency Telemetry

The bottom of the left panel shows a live **Telemetry & Latency Breakdown** table updated after every turn:

| Stage | What it measures |
|---|---|
| STT | Time from end of user speech to final transcript |
| LLM | Time-to-first-token (excluding tool execution) |
| Tool Execution | Time spent in Python tool dispatch |
| TTS | Time-to-first-audio-chunk from TTS |
| **End-to-End TTFA** | Total time from speech end to first PCM byte reaching the browser |

---

## Project Structure

```
JanSahayak/
├── backend/
│   ├── main.py                  # FastAPI app, /healthz, /readyz, /ws
│   ├── config.py                # pydantic-settings configuration
│   ├── agent/
│   │   ├── manager.py           # ConversationManager — two-call LLM orchestration
│   │   ├── state.py             # ConversationState, TurnMetrics, WorkflowState
│   │   ├── prompts.py           # Dynamic system message builder
│   │   ├── tools.py             # 8 tool definitions + Pydantic validators + executors
│   │   └── eligibility.py      # Deterministic eligibility engine (5 schemes)
│   ├── knowledge/
│   │   ├── retriever.py         # In-memory keyword-scoring RAG retriever
│   │   └── schemas.py           # Pydantic KB models
│   ├── persistence/
│   │   └── callback_store.py    # SQLite callback booking store
│   ├── sarvam/
│   │   ├── stt.py               # SarvamSTT — saaras:v3-realtime WebSocket client
│   │   ├── llm.py               # SarvamLLM — orchestrate() + stream_after_tool()
│   │   └── tts.py               # SarvamTTS — bulbul:v3 async context manager
│   ├── audio/
│   │   └── generation.py        # Sentence chunker for streaming TTS
│   └── websocket/
│       ├── handler.py           # Per-session WebSocket handler
│       └── protocol.py          # Typed Pydantic message models
├── frontend/
│   ├── index.html               # Single-page UI with evaluation guide
│   ├── app.js                   # WebSocket client + state machine
│   ├── audio-capture.js         # AudioWorklet PCM capture processor
│   ├── audio-playback.js        # PlaybackQueue for PCM playback
│   └── style.css                # Dashboard styles
├── knowledge/
│   ├── pm_kisan.json
│   ├── ayushman_bharat.json
│   ├── pm_awas.json
│   ├── atal_pension.json
│   └── mgnrega.json
├── evaluation/
│   ├── runner.py                # Automated multi-turn scenario harness
│   ├── metrics.py               # Pydantic evaluation metric models
│   └── scenarios.json           # Test scenario definitions
├── tests/                       # 12 pytest modules (unit + integration)
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Environment Variables

No variables are required to run the application. The API key is provided through the UI.

All settings are loaded from `.env` via `pydantic-settings`. Copy `.env.example` to `.env` only if you want to override a default.

| Variable | Default | Description |
|---|---|---|
| `SARVAM_API_KEY` | *(UI input)* | Sarvam subscription key — provided per-session via the dashboard |
| `APP_ENV` | `development` | Set to `production` to disable `/docs` |
| `LOG_LEVEL` | `DEBUG` | Python logging level |
| `SARVAM_STT_MODEL` | `saaras:v3-realtime` | Realtime STT model |
| `SARVAM_LLM_MODEL` | `sarvam-105b-conversations` | Conversational LLM |
| `SARVAM_TTS_MODEL` | `bulbul:v3` | Streaming TTS model |
| `DEFAULT_RESPONSE_LANGUAGE` | `hi-IN` | Fallback language before STT detection |
| `LLM_MAX_TOKENS` | `256` | Caps spoken responses to ~30–40 words |
| `MAX_RECENT_MESSAGES` | `24` | Conversation history window size |

---

*Built with ❤️ by [Aman Verma](https://www.linkedin.com/in/aman-verma-0a89b8190/) · Powered by [Sarvam AI](https://www.sarvam.ai/)*
