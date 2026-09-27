"""
backend/agent/manager.py

Phase 7+8: ConversationManager — orchestrates one complete user turn.

Responsibilities:
  1. Build the LLM messages list from ConversationState + system message.
  2. Call LLM orchestrate() (Call 1) to get a direct answer or tool request.
  3. If tool call: validate → execute → record → Call 2 stream.
  4. If direct answer: yield it as a single async chunk.
  5. Update ConversationState after the turn.

Architecture (spec §23, §30, §56):
  Two-call pattern:
    Call 1: stream=False, tool_choice="auto"
      → returns ToolCallRequest or direct answer string

    If tool call:
      Python executes tool (dispatch_tool)
      Call 2: stream=True, tool_choice="none"
      → yields final answer tokens

    If direct answer:
      yield the text directly (already complete, no second call needed)

  The sentence chunker + TTS pipeline is driven by the CALLER (handler.py),
  which consumes the async generator yielded by run_turn().

State updates (spec §8, §20):
  - conversation_history is updated AFTER the turn completes.
  - current_scheme and current_workflow are inferred from tool calls.
  - set_language tool result updates response_language immediately.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator

from backend.agent.prompts import build_system_message
from backend.agent.state import ConversationState, TurnMetrics, WorkflowState
from backend.agent.tools import dispatch_tool
from backend.sarvam.llm import SarvamLLM, ToolCallRequest

logger = logging.getLogger(__name__)

# Maximum tool result length sent back to the LLM (spec §182).
_MAX_TOOL_RESULT_LEN = 16000

# Human-readable labels shown in the UI when each tool executes.
_TOOL_DISPLAY_LABELS: dict[str, str] = {
    "search_knowledge":       "🔍 Searching Knowledge Base",
    "check_eligibility":      "📋 Evaluating Eligibility Criteria",
    "get_required_documents": "📄 Fetching Document Checklist",
    "get_application_steps":  "📝 Retrieving Application Steps",
    "get_application_status": "🔎 Tracking Application Status",
    "get_callback_slots":     "📅 Loading Available Callback Slots",
    "book_human_callback":    "📞 Reserving Human Officer Callback",
    "set_language":           "🌐 Switching Conversation Language",
}


def _tool_display_text(tool_name: str, raw_args: str) -> str:
    """
    Build a human-readable display string for a tool call.
    Appends relevant argument context (scheme, query, language) when present.
    """
    base = _TOOL_DISPLAY_LABELS.get(tool_name, f"⚙ Executing {tool_name}")
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        return base

    # Append the most informative argument as context.
    for key in ("scheme_id", "scheme", "query", "language_code", "date"):
        val = args.get(key)
        if val:
            return f"{base} — {val}"
    return base

# Scheme name → scheme_id mapping for workflow tracking.
_TOOL_TO_WORKFLOW: dict[str, str] = {
    "search_knowledge":        WorkflowState.SCHEME_DISCOVERY,
    "check_eligibility":       WorkflowState.ELIGIBILITY,
    "get_required_documents":  WorkflowState.DOCUMENTS,
    "get_application_steps":   WorkflowState.APPLICATION,
    "get_application_status":  WorkflowState.APPLICATION_STATUS,
    "get_callback_slots":      WorkflowState.CALLBACK,
    "book_human_callback":     WorkflowState.CALLBACK,
}


class ConversationManager:
    """
    One instance per session — holds the LLM client and drives turns.

    The handler creates this in __init__; state is passed in per turn.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._llm = SarvamLLM(api_key=api_key)

    async def run_turn(
        self,
        user_text: str,
        detected_lang: str,
        state: ConversationState,
        *,
        generation_id: int,
        turn_metrics: TurnMetrics | None = None,
        on_status: "AsyncStatusCallback | None" = None,
        on_tool_event: "AsyncToolEventCallback | None" = None,
    ) -> "AsyncIterator[str]":
        """
        Execute one complete user turn and yield final-answer text chunks.

        Args:
            user_text:      Finalized STT transcript.
            detected_lang:  BCP-47 language detected by STT (e.g. "hi-IN").
            state:          Mutable ConversationState for this session.
            generation_id:  Current generation counter (barge-in guard).
            on_status:      Optional async callback for agent_status events
                            (e.g. "searching_knowledge", "responding").
            on_tool_event:  Optional async callback emitted before and after
                            each tool execution with (tool_name, display_text,
                            status) where status is "started" or "completed".

        Yields:
            str chunks of the final LLM answer (same interface as Phase 6's
            _token_gen so stream_sentences() works unchanged).

        Side effects:
            - Updates state.conversation_history.
            - Updates state.current_scheme / current_workflow from tool calls.
            - Updates state.response_language if set_language is called.
        """
        # Update detected language (spec §46: mirrors input language until
        # the user issues an explicit set_language).
        state.update_detected_language(detected_lang)
        state.generation_id = generation_id

        t_start = time.monotonic()

        # --- Build messages list ----------------------------------------
        system_msg = build_system_message(state)
        messages = [system_msg] + list(state.conversation_history) + [
            {"role": "user", "content": user_text}
        ]

        # Resolve a human-readable language name for the recency-bias reminders.
        lang_code = state.response_language or "hi-IN"
        lang_name = {
            "hi-IN": "Hindi",
            "en-IN": "English",
            "kn-IN": "Kannada",
            "ta-IN": "Tamil",
            "te-IN": "Telugu",
            "ml-IN": "Malayalam",
            "mr-IN": "Marathi",
            "bn-IN": "Bengali",
            "gu-IN": "Gujarati",
        }.get(lang_code, "Hindi")

        # --- Call 1: orchestrate ----------------------------------------
        if on_status:
            await on_status("thinking")

        if turn_metrics and turn_metrics.t_llm_start is None:
            turn_metrics.t_llm_start = time.perf_counter()

        try:
            result = await self._llm.orchestrate(messages)
        except Exception as exc:
            logger.exception("[Manager] orchestrate() failed: %s", exc)
            yield "I'm having trouble connecting right now. Please try again."
            return

        # --- Branch: tool call or direct answer -------------------------
        if isinstance(result, ToolCallRequest):
            # Notify browser of tool execution (generic status).
            if on_status:
                await on_status("searching_knowledge")

            display_text = _tool_display_text(result.tool_name, result.raw_arguments)

            logger.info(
                "[Manager] Executing tool: %s  args=%r",
                result.tool_name, result.raw_arguments[:120],
            )

            # Emit tool-started event so the UI can show the activity badge.
            if on_tool_event:
                await on_tool_event(result.tool_name, display_text, "started")

            # Execute tool with timing instrumentation
            t_tool = time.monotonic()
            if turn_metrics:
                turn_metrics.t_tool_start = time.perf_counter()

            tool_result_dict = await dispatch_tool(
                result.tool_name,
                result.raw_arguments,
                session_id=state.session_id,
            )

            if turn_metrics:
                turn_metrics.t_tool_end = time.perf_counter()

            # Emit tool-completed event.
            if on_tool_event:
                await on_tool_event(result.tool_name, display_text, "completed")

            tool_elapsed_ms = (time.monotonic() - t_tool) * 1000

            state.metrics["tool_latency_ms"] = tool_elapsed_ms
            logger.info("[Manager] Tool %s done in %.0f ms", result.tool_name, tool_elapsed_ms)

            # Update workflow / scheme from tool call.
            self._update_state_from_tool(result.tool_name, result.raw_arguments, tool_result_dict, state)

            # Truncate tool result to keep prompt size bounded (spec §182).
            # IMPORTANT: never slice the serialised JSON string — that destroys
            # the closing braces/quotes and produces unparseable JSON that the
            # LLM silently ignores. Instead wrap a safe string snippet inside a
            # well-formed JSON envelope so the LLM always receives valid JSON.
            raw_json = json.dumps(tool_result_dict, ensure_ascii=False)
            if len(raw_json) > _MAX_TOOL_RESULT_LEN:
                safe_snippet = str(tool_result_dict)[:_MAX_TOOL_RESULT_LEN] + "..."
                tool_result_json = json.dumps(
                    {"status": "truncated", "data": safe_snippet},
                    ensure_ascii=False,
                )
            else:
                tool_result_json = raw_json

            # Append tool call + result to messages for Call 2.
            messages_with_tool = list(messages) + [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": result.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": result.tool_name,
                                "arguments": result.raw_arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": tool_result_json,
                    "tool_call_id": result.tool_call_id,
                },
            ]

            # --- Call 2: stream final answer ----------------------------
            if on_status:
                await on_status("responding")

            # Re-evaluate language AFTER tool execution: set_language may have
            # just updated state.response_language, so lang_name from the start
            # of the turn could be stale.
            post_tool_lang_code = state.response_language or "hi-IN"
            post_tool_lang_name = {
                "hi-IN": "Hindi",
                "en-IN": "English",
                "kn-IN": "Kannada",
                "ta-IN": "Tamil",
                "te-IN": "Telugu",
                "ml-IN": "Malayalam",
                "mr-IN": "Marathi",
                "bn-IN": "Bengali",
                "gu-IN": "Gujarati",
            }.get(post_tool_lang_code, "Hindi")

            # Second late reminder — appended as a user instruction to maintain valid chat sequence
            messages_with_tool.append({
                "role": "user",
                "content": (
                    f"CRITICAL REMINDER: Synthesize the tool result above and "
                    f"formulate your final response ENTIRELY in {post_tool_lang_name}."
                ),
            })

            full_answer_parts: list[str] = []
            try:
                async for chunk in self._llm.stream_after_tool(messages_with_tool):
                    # Record TTFT when first token arrives
                    if turn_metrics and turn_metrics.t_llm_first_token is None:
                        turn_metrics.t_llm_first_token = time.perf_counter()

                    # Check if generation was interrupted mid-stream (Spec §21, §64)
                    if state.generation_id != generation_id:
                        logger.info(
                            "[Manager] Interrupted turn (gen %d != %d) — discarding partial response from history",
                            generation_id, state.generation_id,
                        )
                        return
                    full_answer_parts.append(chunk)
                    yield chunk
            except Exception as exc:
                logger.exception("[Manager] stream_after_tool() failed: %s", exc)
                yield "I encountered an issue generating a response. Please try again."
                return

            # Check again before committing to canonical history
            if state.generation_id != generation_id:
                logger.info(
                    "[Manager] Turn interrupted before completion (gen %d != %d) — state discarded",
                    generation_id, state.generation_id,
                )
                return

            full_answer = "".join(full_answer_parts)

            # Update history: tool call + result + assistant response.
            state.add_tool_messages(
                tool_call_id=result.tool_call_id,
                tool_name=result.tool_name,
                tool_result_json=tool_result_json,
            )
            state.add_user_message(user_text)
            state.add_assistant_message(full_answer)

        else:
            # Direct answer — result is already the complete text.
            direct_answer: str = result
            if turn_metrics and turn_metrics.t_llm_first_token is None:
                turn_metrics.t_llm_first_token = time.perf_counter()

            if on_status:
                await on_status("responding")

            # Check if generation was interrupted before direct answer emission
            if state.generation_id != generation_id:
                logger.info(
                    "[Manager] Interrupted direct answer (gen %d != %d) — state discarded",
                    generation_id, state.generation_id,
                )
                return

            if direct_answer.strip():
                yield direct_answer
            else:
                yield "I'm not sure I understood that. Could you please repeat?"
                direct_answer = "I'm not sure I understood that. Could you please repeat?"

            if state.generation_id != generation_id:
                return

            # Update history.
            state.add_user_message(user_text)
            state.add_assistant_message(direct_answer)

        elapsed_ms = (time.monotonic() - t_start) * 1000
        state.metrics["turn_latency_ms"] = elapsed_ms
        logger.info(
            "[Manager] Turn complete — %.0f ms  scheme=%s  workflow=%s  lang=%s",
            elapsed_ms, state.current_scheme, state.current_workflow, state.response_language,
        )

    # ------------------------------------------------------------------
    # State update helpers
    # ------------------------------------------------------------------

    def _update_state_from_tool(
        self,
        tool_name: str,
        raw_args: str,
        tool_result: dict,
        state: ConversationState,
    ) -> None:
        """
        Update structured state based on which tool was called and its result.
        """
        # Update workflow.
        new_workflow = _TOOL_TO_WORKFLOW.get(tool_name)
        if new_workflow:
            state.current_workflow = new_workflow

        # Extract scheme from tool arguments.
        try:
            args_dict = json.loads(raw_args)
            scheme = args_dict.get("scheme") or args_dict.get("scheme_id")
            if scheme:
                state.current_scheme = scheme
        except (json.JSONDecodeError, AttributeError):
            args_dict = {}

        # For search_knowledge, also track which scheme was matched.
        if tool_name == "search_knowledge":
            matched = tool_result.get("scheme_matched")
            if matched and not state.current_scheme:
                state.current_scheme = matched

        # For get_application_status, track scheme if returned in lookup.
        if tool_name == "get_application_status" and tool_result.get("found"):
            app_scheme = tool_result.get("scheme")
            if app_scheme and not state.current_scheme:
                state.current_scheme = app_scheme

        # For check_eligibility, merge provided slots into state.collected_slots (Spec §84).
        if tool_name == "check_eligibility":
            provided = args_dict.get("provided_slots") or args_dict.get("user_data") or {}
            if isinstance(provided, dict):
                state.collected_slots.update(provided)
                logger.info("[Manager] Merged slots into state.collected_slots: %s", list(provided.keys()))

            # Track missing slots in state.required_slots if returned
            missing = tool_result.get("missing_fields")
            if missing is not None:
                state.required_slots = missing

        # For set_language, update response_language immediately (spec §44).
        if tool_name == "set_language" and tool_result.get("success"):
            lang = tool_result.get("language_code", "")
            if lang:
                state.set_response_language(lang)
                logger.info("[Manager] Language switched to: %s", lang)


# ---------------------------------------------------------------------------
# Type aliases for the manager callbacks
# ---------------------------------------------------------------------------

from collections.abc import Awaitable, Callable
AsyncStatusCallback    = Callable[[str], Awaitable[None]]
# (tool_name, display_text, status) → None
AsyncToolEventCallback = Callable[[str, str, str], Awaitable[None]]
