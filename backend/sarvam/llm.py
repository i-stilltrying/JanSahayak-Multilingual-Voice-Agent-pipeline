"""
backend/sarvam/llm.py

Wraps Sarvam's async chat completions endpoint for conversational turns.

Phase 4 scope (stateless streaming):
  - stream_response() yields text chunks from the LLM as an async generator.

Phase 7 addition — two-call tool-calling pattern (spec §23, §30):
  - orchestrate() performs Call 1: non-streaming, tool_choice="auto".
    Returns either:
      (a) A direct answer text (when the model answers directly), or
      (b) A ToolCallRequest with tool name and raw JSON arguments.
  - stream_after_tool() performs Call 2: streaming, tool_choice="none".
    Called after the tool has been executed; yields response tokens.

Phase 8 addition:
  - Both methods accept a full messages list so ConversationState history
    and the system message are injected by the caller (agent/manager.py).

Model choice (spec §26):
  sarvam-105b-conversations — conversational variant for real-time voice.

Latency configuration (spec §27, §28, §136):
  - reasoning_effort is NOT passed (omitted → API default disabled).
  - max_tokens caps spoken responses to ~30-40 words.
  - stream=True on final answer overlaps generation with downstream TTS.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from sarvamai import AsyncSarvamAI

from backend.agent.tools import TOOL_DEFINITIONS
from backend.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return type for tool call branch
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRequest:
    """Represents the LLM's request to call a tool (Call 1 result)."""
    tool_call_id: str
    tool_name: str
    raw_arguments: str   # JSON string — validated by tools.py before execution


# ---------------------------------------------------------------------------
# SarvamLLM
# ---------------------------------------------------------------------------

class SarvamLLM:
    """
    Async LLM wrapper for one session.

    Usage (two-call pattern for tool-using turns):
        result = await llm.orchestrate(messages)
        if isinstance(result, ToolCallRequest):
            tool_result = await dispatch_tool(result.tool_name, result.raw_arguments)
            async for chunk in llm.stream_after_tool(messages, result, tool_result_json):
                yield chunk
        else:
            async for chunk in llm.stream_direct(messages):
                yield chunk   # (or re-call stream_after_tool with no tool turn)

    For direct answers (no tool call), orchestrate() returns the full text
    directly to avoid a redundant second call.
    """

    def __init__(self, api_key: str | None = None) -> None:
        _key = api_key or settings.sarvam_api_key
        self._client = AsyncSarvamAI(
            api_subscription_key=_key
        )

    # ------------------------------------------------------------------
    # Call 1 — orchestration (non-streaming, tool_choice="auto")
    # ------------------------------------------------------------------

    async def orchestrate(
        self,
        messages: list[dict[str, Any]],
    ) -> ToolCallRequest | str:
        """
        Send messages to the LLM with tool definitions and tool_choice="auto".

        Returns:
            ToolCallRequest  — if the model decided to call a tool.
            str              — the full direct answer text if no tool is needed.

        Raises on API error (caller should catch and send error to browser).

        Spec §30: stream=False + tool_choice="auto" for this call so we can
        read the complete tool name and JSON arguments before executing.
        """
        t_start = time.monotonic()
        logger.info(
            "[LLM] orchestrate() — %d messages, model=%s",
            len(messages),
            settings.sarvam_llm_model,
        )

        response = await self._client.chat.completions(
            model=settings.sarvam_llm_model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            stream=False,
            max_tokens=settings.llm_max_tokens,
        )

        elapsed_ms = (time.monotonic() - t_start) * 1000
        logger.info("[LLM] orchestrate() completed in %.0f ms", elapsed_ms)

        choice = response.choices[0]
        finish_reason = choice.finish_reason

        # --- Tool call branch ---
        if finish_reason == "tool_calls":
            tool_calls = choice.message.tool_calls
            if tool_calls:
                tc = tool_calls[0]  # spec §97: at most 1 tool per turn
                req = ToolCallRequest(
                    tool_call_id=tc.id,
                    tool_name=tc.function.name,
                    raw_arguments=tc.function.arguments,
                )
                logger.info(
                    "[LLM] Tool call requested: %s  args=%r",
                    req.tool_name,
                    req.raw_arguments[:120],
                )
                return req

        # --- Direct answer branch ---
        content = choice.message.content or ""
        logger.info(
            "[LLM] Direct answer (%d chars) in %.0f ms", len(content), elapsed_ms
        )
        return content

    # ------------------------------------------------------------------
    # Call 2 — final streaming response after tool execution
    # ------------------------------------------------------------------

    async def stream_after_tool(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """
        Stream the final LLM answer.  Called after the tool result has been
        appended to `messages` by the caller.

        tool_choice="none" ensures the model produces prose, not another tool call.

        Yields non-empty text chunks as they arrive.
        """
        t_start = time.monotonic()
        first_token_logged = False

        logger.info(
            "[LLM] stream_after_tool() — %d messages", len(messages)
        )

        stream = await self._client.chat.completions(
            model=settings.sarvam_llm_model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="none",
            stream=True,
            max_tokens=settings.llm_max_tokens,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = delta.content if delta else None
            if not text:
                continue

            if not first_token_logged:
                ttft = (time.monotonic() - t_start) * 1000
                logger.info("[LLM] stream_after_tool first token in %.0f ms", ttft)
                first_token_logged = True

            yield text

        elapsed_ms = (time.monotonic() - t_start) * 1000
        logger.info("[LLM] stream_after_tool complete in %.0f ms", elapsed_ms)

    # ------------------------------------------------------------------
    # Phase 4 legacy — simple streaming without tools (still used internally
    # when orchestrate() returns a direct-answer string: we stream nothing
    # further because the text is already complete.  Kept as a fallback.)
    # ------------------------------------------------------------------

    async def stream_response(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """
        Simple streaming call with no tools.  Used as the final-answer path
        when orchestrate() already returned a direct str (we re-stream it as
        a single synthetic chunk so the sentence chunker still works).

        NOTE: this is a thin generator that yields the pre-formed answer
        string as one chunk.  The real streaming path is stream_after_tool().
        """
        # This method is kept for backward compatibility.
        # orchestrate() handles the direct-answer case now.
        raise NotImplementedError(
            "Use orchestrate() + stream_after_tool() in Phase 7+. "
            "Direct streaming is handled inside ConversationManager."
        )
