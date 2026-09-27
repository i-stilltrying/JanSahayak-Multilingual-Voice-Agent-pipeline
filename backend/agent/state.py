"""
backend/agent/state.py

Phase 8: ConversationState — the authoritative, session-scoped memory object.

Architecture (spec §14, §15, §16):
  Two types of memory live here:
    1. Structured state  — current_scheme, current_workflow, collected_slots,
                           response_language, etc.  Authoritative and accessed
                           directly by the agent manager and tools.
    2. Conversation history — a bounded list of finalized LLM message dicts.
                              Pruned to MAX_RECENT_MESSAGES (spec §18).

  The two are kept separate so the LLM always has precise business state even
  when the raw history is pruned (spec §16).

Session isolation (spec §162–164):
  One ConversationState per WebSocket session.  All fields are session-local.
  They must never be shared across sessions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from backend.config import settings


# ---------------------------------------------------------------------------
# Turn Latency Metrics (Spec §130–132, §144)
# ---------------------------------------------------------------------------

@dataclass
class TurnMetrics:
    """
    Granular timestamps and computed deltas for one conversational turn.
    All raw timestamps (t_*) are recorded using time.perf_counter().
    """

    t_speech_end: float | None = None
    t_stt_final: float | None = None
    t_llm_start: float | None = None
    t_llm_first_token: float | None = None
    t_tool_start: float | None = None
    t_tool_end: float | None = None
    t_tts_start: float | None = None
    t_tts_first_audio: float | None = None

    def calculate_breakdown(self) -> dict[str, float]:
        """
        Compute latency durations in milliseconds for each pipeline stage.
        Missing stages (e.g. non-tool turns) will be omitted or None-safe.
        """
        breakdown: dict[str, float] = {}

        # STT latency: time from user finished speaking to STT transcript_final
        if self.t_speech_end is not None and self.t_stt_final is not None:
            breakdown["stt_ms"] = max(0.0, (self.t_stt_final - self.t_speech_end) * 1000)

        # Tool latency: time spent in tool dispatch/execution
        if self.t_tool_start is not None and self.t_tool_end is not None:
            breakdown["tool_ms"] = max(0.0, (self.t_tool_end - self.t_tool_start) * 1000)

        # LLM TTFT: time from LLM start to first token yielded (excluding tool wait if tool was called)
        if self.t_llm_start is not None and self.t_llm_first_token is not None:
            raw_llm_ms = (self.t_llm_first_token - self.t_llm_start) * 1000
            if "tool_ms" in breakdown and self.t_tool_start is not None:
                # Disaggregate pure LLM computation from tool execution
                breakdown["llm_ms"] = max(0.0, raw_llm_ms - breakdown["tool_ms"])
            else:
                breakdown["llm_ms"] = max(0.0, raw_llm_ms)

        # TTS TTFA: time from first sentence sent to TTS until first PCM bytes received
        if self.t_tts_start is not None and self.t_tts_first_audio is not None:
            breakdown["tts_ms"] = max(0.0, (self.t_tts_first_audio - self.t_tts_start) * 1000)

        # Backend Total E2E: from speech_end (or stt_final) to first audio emitted
        t_origin = self.t_speech_end if self.t_speech_end is not None else self.t_stt_final
        if t_origin is not None and self.t_tts_first_audio is not None:
            breakdown["e2e_ms"] = max(0.0, (self.t_tts_first_audio - t_origin) * 1000)

        return breakdown


# ---------------------------------------------------------------------------
# Workflow state enum (spec §22)
# ---------------------------------------------------------------------------

class WorkflowState:
    NONE                = "NONE"
    SCHEME_DISCOVERY    = "SCHEME_DISCOVERY"
    FAQ                 = "FAQ"
    ELIGIBILITY         = "ELIGIBILITY"
    DOCUMENTS           = "DOCUMENTS"
    APPLICATION         = "APPLICATION"
    APPLICATION_STATUS  = "APPLICATION_STATUS"
    CALLBACK            = "CALLBACK"


# ---------------------------------------------------------------------------
# ConversationState dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConversationState:
    """
    All per-session state that the agent manager reads and writes.

    Designed to be passed into build_system_message() and
    to be updated after every turn without copying (mutations in-place).
    """

    session_id:          str
    input_language:      str = ""          # detected by STT (BCP-47, e.g. "hi-IN")
    response_language:   str = ""          # language the agent responds in
    current_scheme:      str | None = None # e.g. "pm_kisan"
    current_workflow:    str = WorkflowState.NONE
    required_slots:      list[str] = field(default_factory=list)
    collected_slots:     dict[str, Any] = field(default_factory=dict)
    generation_id:       int = 0
    created_at_ms:       float = field(default_factory=lambda: time.time() * 1000)

    # Conversation history — list of OpenAI-compatible message dicts.
    # Only finalized turns are stored here (spec §20).
    conversation_history: list[dict[str, Any]] = field(default_factory=list)

    # Latency metrics for the current turn (reset on each new turn).
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.response_language:
            self.response_language = settings.default_response_language

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def add_user_message(self, text: str) -> None:
        """Append a finalized user turn to conversation history."""
        self.conversation_history.append({"role": "user", "content": text})
        self._prune_history()

    def add_assistant_message(self, text: str) -> None:
        """Append a finalized assistant response to conversation history."""
        self.conversation_history.append({"role": "assistant", "content": text})
        self._prune_history()

    def add_tool_messages(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_result_json: str,
    ) -> None:
        """
        Append the assistant tool-call message and the tool result.
        Both must be added together so the LLM history is valid.
        """
        # Assistant message that requested the tool call.
        self.conversation_history.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": "{}"},
                    }
                ],
            }
        )
        # Tool result.
        self.conversation_history.append(
            {
                "role": "tool",
                "content": tool_result_json,
                "tool_call_id": tool_call_id,
            }
        )
        self._prune_history()

    def _prune_history(self) -> None:
        """Keep only the most recent N messages, avoiding orphaned tool calls."""
        max_msgs = settings.max_recent_messages
        if len(self.conversation_history) <= max_msgs:
            return

        slice_idx = len(self.conversation_history) - max_msgs

        # An OpenAI-compatible message history cannot start with a 'tool' role message.
        # If we slice exactly on a tool message, shift the index forward to drop it.
        while slice_idx < len(self.conversation_history) and self.conversation_history[slice_idx].get("role") == "tool":
            slice_idx += 1

        self.conversation_history = self.conversation_history[slice_idx:]

    # ------------------------------------------------------------------
    # Language helpers
    # ------------------------------------------------------------------

    def update_detected_language(self, language: str) -> None:
        """
        Update input_language from STT detection.
        Always mirrors response_language — the handler already set both fields
        aggressively before calling run_turn, so this keeps the manager
        consistent without re-introducing a sticky guard.
        """
        if language:
            self.input_language = language
            self.response_language = language

    def set_response_language(self, language_code: str) -> None:
        """Explicit language switch via set_language tool (spec §44–46)."""
        self.response_language = language_code
