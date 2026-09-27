"""
tests/test_callback.py

Phase 12: Tests for SQLite callback store, atomic booking, race conditions, idempotency,
and tool integration.
"""

from __future__ import annotations

import json
import os
import pytest
from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

# Ensure SARVAM_API_KEY is set for test environment
os.environ.setdefault("SARVAM_API_KEY", "dummy_test_key")

from backend.persistence.callback_store import CallbackStore
from backend.agent.tools import dispatch_tool, GetCallbackSlotsArgs, BookCallbackArgs
from backend.agent.manager import ConversationManager
from backend.agent.state import ConversationState, WorkflowState
from backend.sarvam.llm import ToolCallRequest


@pytest.fixture
def temp_store(tmp_path):
    """Provide an isolated, freshly initialized SQLite database for each test."""
    db_file = tmp_path / "test_callbacks.db"
    return CallbackStore(db_path=db_file)


class TestCallbackStoreDirect:
    def test_get_available_slots(self, temp_store):
        today_str = date.today().strftime("%Y-%m-%d")
        slots = temp_store.get_available_slots(today_str)
        assert len(slots) >= 5
        assert all(s["date"] == today_str for s in slots)
        assert any("10:00 AM" in s["start_time"] for s in slots)

    def test_successful_atomic_booking(self, temp_store):
        today_str = date.today().strftime("%Y-%m-%d")
        res = temp_store.book_slot(
            date_str=today_str,
            time_slot="10:00 AM",
            idempotency_key="sess1:today:10am",
            language="hi-IN",
        )

        assert res["success"] is True
        assert res["status"] == "CONFIRMED"
        assert res["booking_reference"].startswith("CBK-")
        assert res["date"] == today_str
        assert "10:00 AM" in res["time_slot"]
        assert res["idempotent_replay"] is False

        # Verify that slot is no longer in available list
        available = temp_store.get_available_slots(today_str)
        assert not any(s["start_time"] == "10:00 AM" for s in available)

    def test_race_condition_second_booking_fails(self, temp_store):
        today_str = date.today().strftime("%Y-%m-%d")

        # User A books slot
        res_a = temp_store.book_slot(
            date_str=today_str,
            time_slot="11:30 AM",
            idempotency_key="user_A:11:30",
        )
        assert res_a["success"] is True
        assert res_a["status"] == "CONFIRMED"

        # User B tries to book the exact same slot concurrently
        res_b = temp_store.book_slot(
            date_str=today_str,
            time_slot="11:30 AM",
            idempotency_key="user_B:11:30",
        )
        assert res_b["success"] is False
        assert res_b["status"] == "ALREADY_BOOKED"
        assert res_b["booking_reference"] is None
        assert "no longer available" in res_b["message"]

    def test_idempotency_graceful_replay(self, temp_store):
        today_str = date.today().strftime("%Y-%m-%d")
        idempotency_key = "sess_retry:02:00PM"

        # First call (original attempt)
        res_1 = temp_store.book_slot(
            date_str=today_str,
            time_slot="02:00 PM",
            idempotency_key=idempotency_key,
        )
        assert res_1["success"] is True
        assert res_1["idempotent_replay"] is False
        ref_1 = res_1["booking_reference"]

        # Second call with exact same idempotency_key (simulating LLM / network retry)
        res_2 = temp_store.book_slot(
            date_str=today_str,
            time_slot="02:00 PM",
            idempotency_key=idempotency_key,
        )
        assert res_2["success"] is True
        assert res_2["idempotent_replay"] is True
        assert res_2["booking_reference"] == ref_1
        assert res_2["status"] == "CONFIRMED"

    def test_booking_nonexistent_slot_fails(self, temp_store):
        res = temp_store.book_slot(
            date_str="2099-01-01",
            time_slot="03:00 AM",
            idempotency_key="sess_invalid",
        )
        assert res["success"] is False
        assert res["status"] == "NOT_FOUND"


class TestCallbackToolsIntegration:
    @pytest.mark.asyncio
    async def test_get_callback_slots_tool(self, monkeypatch, temp_store):
        import backend.agent.tools
        monkeypatch.setattr(backend.agent.tools, "callback_store", temp_store)

        today_str = date.today().strftime("%Y-%m-%d")
        payload = json.dumps({"date": today_str})
        result = await dispatch_tool("get_callback_slots", payload, session_id="sess_tool_1")

        assert result["success"] is True
        assert len(result["available_slots"]) >= 5
        assert "instruction_for_llm" in result

    @pytest.mark.asyncio
    async def test_book_human_callback_tool_success(self, monkeypatch, temp_store):
        import backend.agent.tools
        monkeypatch.setattr(backend.agent.tools, "callback_store", temp_store)

        today_str = date.today().strftime("%Y-%m-%d")
        payload = json.dumps({
            "date": today_str,
            "time_slot": "03:30 PM",
            "language": "hi-IN",
        })
        result = await dispatch_tool("book_human_callback", payload, session_id="sess_book_1")

        assert result["success"] is True
        assert result["status"] == "CONFIRMED"
        assert result["booking_reference"].startswith("CBK-")
        assert "Booking is CONFIRMED" in result["instruction_for_llm"]

    @pytest.mark.asyncio
    async def test_book_human_callback_tool_failure_guardrail(self, monkeypatch, temp_store):
        import backend.agent.tools
        monkeypatch.setattr(backend.agent.tools, "callback_store", temp_store)

        today_str = date.today().strftime("%Y-%m-%d")
        payload_1 = json.dumps({"date": today_str, "time_slot": "05:00 PM", "language": "en-IN"})
        await dispatch_tool("book_human_callback", payload_1, session_id="sess_user_1")

        # Second user booking same slot
        payload_2 = json.dumps({"date": today_str, "time_slot": "05:00 PM", "language": "en-IN"})
        result_2 = await dispatch_tool("book_human_callback", payload_2, session_id="sess_user_2")

        assert result_2["success"] is False
        assert result_2["booking_reference"] is None
        assert "Booking FAILED" in result_2["instruction_for_llm"]
        assert "Do NOT confirm a booking" in result_2["instruction_for_llm"]


class TestCallbackStateAndManager:
    def test_manager_updates_workflow_to_callback(self):
        manager = ConversationManager()
        state = ConversationState(session_id="sess_cbk_flow")

        # Calling get_callback_slots transitions workflow
        raw_args_1 = json.dumps({})
        tool_res_1 = {"success": True, "available_slots": []}
        manager._update_state_from_tool("get_callback_slots", raw_args_1, tool_res_1, state)
        assert state.current_workflow == WorkflowState.CALLBACK

        # Calling book_human_callback keeps workflow as CALLBACK
        raw_args_2 = json.dumps({"date": "2025-05-01", "time_slot": "10:00 AM", "language": "en-IN"})
        tool_res_2 = {"success": True, "booking_reference": "CBK-123456"}
        manager._update_state_from_tool("book_human_callback", raw_args_2, tool_res_2, state)
        assert state.current_workflow == WorkflowState.CALLBACK

    @pytest.mark.asyncio
    async def test_full_turn_booking_orchestration(self, monkeypatch, temp_store):
        import backend.agent.tools
        monkeypatch.setattr(backend.agent.tools, "callback_store", temp_store)

        manager = ConversationManager()
        today_str = date.today().strftime("%Y-%m-%d")
        state = ConversationState(session_id="sess_e2e_cbk", response_language="hi-IN")

        mock_tool_request = ToolCallRequest(
            tool_call_id="call_book_777",
            tool_name="book_human_callback",
            raw_arguments=json.dumps({
                "date": today_str,
                "time_slot": "10:00 AM",
                "language": "hi-IN",
            }),
        )

        async def mock_stream_chunks(messages):
            yield "आपका कॉलबैक सफलतापूर्वक बुक हो गया है।"
            yield " आपका बुकिंग रेफरेंस नंबर CBK-XXXX है।"

        with patch.object(manager._llm, "orchestrate", new_callable=AsyncMock) as mock_orch, \
             patch.object(manager._llm, "stream_after_tool", side_effect=mock_stream_chunks):
            mock_orch.return_value = mock_tool_request

            chunks = []
            async for chunk in manager.run_turn(
                user_text="कृपया आज सुबह 10 बजे कॉलबैक बुक करें",
                detected_lang="hi-IN",
                state=state,
                generation_id=1,
            ):
                chunks.append(chunk)

            full_resp = "".join(chunks)
            assert "सफलतापूर्वक" in full_resp
            assert state.current_workflow == WorkflowState.CALLBACK
