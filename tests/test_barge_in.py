"""
tests/test_barge_in.py

Phase 13: Tests for Barge-In Interruption, Generation ID incrementing,
and Conversation History Memory Guard (Spec §21, §61–65).
"""

from __future__ import annotations

import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure SARVAM_API_KEY is set for test environment
os.environ.setdefault("SARVAM_API_KEY", "dummy_test_key")

from backend.agent.manager import ConversationManager
from backend.agent.state import ConversationState, WorkflowState
from backend.sarvam.llm import ToolCallRequest
from backend.websocket.handler import WebSocketHandler
from backend.websocket.protocol import ClientEventMessage, ClientEventType


class TestBargeInMemoryGuard:
    @pytest.mark.asyncio
    async def test_direct_answer_interrupted_discards_history(self):
        """
        Spec §21: If the generation ID changes before the direct answer is committed,
        the turn is discarded from conversation_history.
        """
        manager = ConversationManager()
        state = ConversationState(session_id="test_sess_bargein", generation_id=1)

        async def mock_orchestrate_with_interrupt(messages):
            # Barge-in happens while orchestrate is computing
            state.generation_id = 2
            return "This is a direct response from JanSahayak."

        with patch.object(manager._llm, "orchestrate", side_effect=mock_orchestrate_with_interrupt):
            chunks = []
            async for chunk in manager.run_turn(
                user_text="Tell me about PM Kisan",
                detected_lang="en-IN",
                state=state,
                generation_id=1,  # Turn started with gen=1, but state is now gen=2
            ):
                chunks.append(chunk)

            # Nothing should be committed to state.conversation_history
            assert len(state.conversation_history) == 0
            assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_tool_call_interrupted_stream_discards_history(self):
        """
        Spec §21: If an active streaming response after tool execution is interrupted,
        the partial text is discarded and not appended to canonical history.
        """
        manager = ConversationManager()
        state = ConversationState(session_id="test_tool_bargein", generation_id=1)

        mock_tool_request = ToolCallRequest(
            tool_call_id="call_doc_1",
            tool_name="get_required_documents",
            raw_arguments=json.dumps({"scheme_id": "pm_kisan"}),
        )

        async def mock_streaming_chunks(messages):
            yield "Here are the documents: "
            # Interrupt occurs mid-stream
            state.generation_id = 2
            yield "1. Aadhaar Card "
            yield "2. Land records."

        with patch.object(manager._llm, "orchestrate", new_callable=AsyncMock) as mock_orch, \
             patch.object(manager._llm, "stream_after_tool", side_effect=mock_streaming_chunks):
            mock_orch.return_value = mock_tool_request

            chunks = []
            async for chunk in manager.run_turn(
                user_text="What documents do I need for PM Kisan?",
                detected_lang="en-IN",
                state=state,
                generation_id=1,
            ):
                chunks.append(chunk)

            # First chunk yielded before interruption
            assert chunks == ["Here are the documents: "]

            # Crucial: History must NOT contain the interrupted assistant message
            assert len(state.conversation_history) == 0


class TestWebSocketHandlerBargeIn:
    @pytest.mark.asyncio
    async def test_barge_in_event_increments_generation_id_and_cancels_tasks(self):
        """
        Verify that receiving a client barge_in event increments generation_id
        and cancels the active LLM task.
        """
        mock_ws = MagicMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.send_bytes = AsyncMock()

        handler = WebSocketHandler(mock_ws)
        assert handler.generation_id == 0
        assert handler._state.generation_id == 0

        # Simulate an active LLM task
        async def dummy_long_task():
            await asyncio.sleep(10)

        llm_task = asyncio.create_task(dummy_long_task())
        handler._active_llm_task = llm_task

        # Send barge-in event
        barge_msg = ClientEventMessage(
            event=ClientEventType.BARGE_IN,
            data={"reason": "user_spoke"},
        )
        await handler._on_client_event(barge_msg)
        await asyncio.sleep(0)  # Yield to event loop to let cancellation propagate

        # Generation ID must increment
        assert handler.generation_id == 1
        assert handler._state.generation_id == 1

        # Active LLM task must be cancelled
        assert llm_task.cancelled() or llm_task.done()
        assert handler._active_llm_task is None

    @pytest.mark.asyncio
    async def test_subsequent_turns_use_incremented_generation_id(self):
        """
        Verify that after a barge-in, new turns proceed with the new generation ID.
        """
        mock_ws = MagicMock()
        mock_ws.send_text = AsyncMock()
        handler = WebSocketHandler(mock_ws)

        # 1st Barge-in
        await handler._on_client_event(ClientEventMessage(event=ClientEventType.BARGE_IN))
        assert handler.generation_id == 1
        assert handler._state.generation_id == 1

        # 2nd Barge-in
        await handler._on_client_event(ClientEventMessage(event=ClientEventType.BARGE_IN))
        assert handler.generation_id == 2
        assert handler._state.generation_id == 2
