"""
backend/websocket/protocol.py

Typed message models for the browser ↔ backend WebSocket protocol.
All JSON messages are Pydantic models so every outbound message is
schema-validated before being sent to the browser.

Protocol overview (spec §10):
  Browser → Backend (JSON):
    StartSession       {"type": "start_session"}
    ClientEvent        {"type": "client_event", "event": "playback_started"|"barge_in"}

  Browser → Backend (binary):
    Raw LINEAR16 PCM audio frames — 16 kHz, mono, int16

  Backend → Browser (JSON):
    SessionReady       {"type": "session_ready", "session_id": "..."}
    TranscriptPartial  {"type": "transcript_partial", "text": "..."}
    TranscriptFinal    {"type": "transcript_final",   "text": "..."}
    AgentText          {"type": "agent_text",         "text": "..."}
    AgentStatus        {"type": "agent_status",       "status": "..."}
    LanguageChanged    {"type": "language_changed",   "language": "..."}
    LatencyMetrics     {"type": "latency",            "metrics": {...}}
    ErrorEvent         {"type": "error",              "code": "...", "message": "..."}
    ToolCallEvent      {"type": "tool_call",          "tool_name": "...", "display_text": "...", "status": "started"|"completed"}

  Backend → Browser (binary):
    Raw LINEAR16 PCM audio frames for playback — 16 kHz, mono, int16
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class MessageType(str, Enum):
    # Inbound (browser → backend)
    START_SESSION = "start_session"
    CLIENT_EVENT = "client_event"

    # Outbound (backend → browser)
    SESSION_READY = "session_ready"
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    AGENT_TEXT = "agent_text"
    AGENT_STATUS = "agent_status"
    LANGUAGE_CHANGED = "language_changed"
    LATENCY = "latency"
    ERROR = "error"
    # Phase 4 — streaming LLM text chunks
    LLM_CHUNK = "llm_chunk"
    LLM_COMPLETE = "llm_complete"
    # Tool-call visibility events
    TOOL_CALL = "tool_call"


class ClientEventType(str, Enum):
    PLAYBACK_STARTED = "playback_started"
    BARGE_IN = "barge_in"


class AgentStatusValue(str, Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SEARCHING_KNOWLEDGE = "searching_knowledge"
    RESPONDING = "responding"
    IDLE = "idle"


# ---------------------------------------------------------------------------
# Inbound messages (browser → backend, JSON)
# ---------------------------------------------------------------------------


class StartSessionMessage(BaseModel):
    type: MessageType = MessageType.START_SESSION
    api_key: str | None = None


class ClientEventMessage(BaseModel):
    type: MessageType = MessageType.CLIENT_EVENT
    event: ClientEventType
    # Optional payload — e.g. client-side performance timestamps
    data: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Outbound messages (backend → browser, JSON)
# Each message carries session metadata to support debugging.
# ---------------------------------------------------------------------------


class _BaseOutbound(BaseModel):
    session_id: str
    turn_id: int = 0
    generation_id: int = 0
    timestamp_ms: float = 0.0  # unix epoch ms, filled by handler


class SessionReadyMessage(_BaseOutbound):
    type: MessageType = MessageType.SESSION_READY


class TranscriptPartialMessage(_BaseOutbound):
    type: MessageType = MessageType.TRANSCRIPT_PARTIAL
    text: str
    language: str = ""


class TranscriptFinalMessage(_BaseOutbound):
    type: MessageType = MessageType.TRANSCRIPT_FINAL
    text: str
    language: str = ""


class AgentTextMessage(_BaseOutbound):
    type: MessageType = MessageType.AGENT_TEXT
    text: str


class AgentStatusMessage(_BaseOutbound):
    type: MessageType = MessageType.AGENT_STATUS
    status: AgentStatusValue


class LanguageChangedMessage(_BaseOutbound):
    type: MessageType = MessageType.LANGUAGE_CHANGED
    language: str


class LatencyMetricsMessage(_BaseOutbound):
    type: MessageType = MessageType.LATENCY
    metrics: dict[str, float] = Field(default_factory=dict)
    # Tool identity — populated when a tool was executed during this turn.
    # Allows the browser to label the "Tool" latency row with the tool name.
    tool_name: str | None = None
    tool_display: str | None = None


class ErrorMessage(_BaseOutbound):
    type: MessageType = MessageType.ERROR
    code: str
    message: str


class LlmChunkMessage(_BaseOutbound):
    """One streamed text token/chunk from the LLM."""
    type: MessageType = MessageType.LLM_CHUNK
    text: str


class LlmCompleteMessage(_BaseOutbound):
    """Signals that the LLM stream for the current turn has finished."""
    type: MessageType = MessageType.LLM_COMPLETE
    full_text: str = ""          # accumulated full response (optional convenience)
    latency_ms: float = 0.0     # total LLM wall-clock time in ms


class ToolCallEventMessage(_BaseOutbound):
    """
    Emitted twice per tool-using turn:
      • status="started"   — immediately after Call 1 returns a ToolCallRequest.
      • status="completed" — immediately after dispatch_tool() returns.

    display_text is a human-readable label that includes relevant context
    (e.g. scheme name or query) extracted from the tool arguments.
    """
    type: MessageType = MessageType.TOOL_CALL
    tool_name: str
    display_text: str
    status: str   # "started" | "completed"
