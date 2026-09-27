"""
tests/test_language_switch.py

Phase 11: Tests for explicit language switching tool (`set_language`) and state preservation.
"""

from __future__ import annotations

import json
import os
import pytest
from unittest.mock import AsyncMock, patch

# Ensure SARVAM_API_KEY is set for test environment
os.environ.setdefault("SARVAM_API_KEY", "dummy_test_key")

from backend.agent.tools import SetLanguageArgs, dispatch_tool
from backend.agent.manager import ConversationManager
from backend.agent.state import ConversationState, WorkflowState
from backend.agent.prompts import build_system_message
from backend.sarvam.llm import ToolCallRequest


class TestSetLanguageValidation:
    def test_valid_language_codes(self):
        for code in ["en-IN", "hi-IN", "kn-IN"]:
            args = SetLanguageArgs(language_code=code)
            assert args.language_code == code

    def test_invalid_language_codes_rejected(self):
        invalid_codes = ["fr-FR", "es-ES", "english", "hindi", "te-IN", ""]
        for code in invalid_codes:
            with pytest.raises(Exception):
                SetLanguageArgs(language_code=code)


class TestSetLanguageToolExecution:
    @pytest.mark.asyncio
    async def test_dispatch_set_language_valid(self):
        payload = json.dumps({"language_code": "hi-IN"})
        result = await dispatch_tool("set_language", payload)

        assert result["success"] is True
        assert result["language_code"] == "hi-IN"
        assert "instruction_for_llm" in result
        assert "Response language successfully updated to hi-IN" in result["instruction_for_llm"]

    @pytest.mark.asyncio
    async def test_dispatch_set_language_kannada(self):
        payload = json.dumps({"language_code": "kn-IN"})
        result = await dispatch_tool("set_language", payload)

        assert result["success"] is True
        assert result["language_code"] == "kn-IN"

    @pytest.mark.asyncio
    async def test_dispatch_set_language_invalid(self):
        payload = json.dumps({"language_code": "invalid-lang"})
        result = await dispatch_tool("set_language", payload)

        assert result["success"] is False
        assert result["error"] == "invalid_arguments"

    @pytest.mark.asyncio
    async def test_dispatch_set_language_malformed_json(self):
        result = await dispatch_tool("set_language", "NOT_A_JSON")
        assert result["success"] is False
        assert result["error"] == "invalid_arguments"


class TestLanguageSwitchStatePreservation:
    def test_manager_updates_response_language(self):
        manager = ConversationManager()
        state = ConversationState(session_id="test_sess_lang", response_language="hi-IN")

        raw_args = json.dumps({"language_code": "kn-IN"})
        tool_result = {"success": True, "language_code": "kn-IN"}
        manager._update_state_from_tool("set_language", raw_args, tool_result, state)

        assert state.response_language == "kn-IN"

    def test_multi_turn_scheme_and_workflow_retention_on_language_switch(self):
        """
        Spec §90: Switching language must NOT clear current_scheme, current_workflow,
        or collected_slots.
        """
        manager = ConversationManager()
        state = ConversationState(
            session_id="test_multi_turn",
            response_language="hi-IN",
            current_scheme="pm_kisan",
            current_workflow=WorkflowState.ELIGIBILITY,
            collected_slots={"land_ownership_hectares": 1.5},
            required_slots=["is_institutional_landholder", "pays_income_tax"],
        )

        # User explicitly asks to speak in English
        raw_args = json.dumps({"language_code": "en-IN"})
        tool_result = {
            "success": True,
            "language_code": "en-IN",
            "message": "Response language successfully updated to en-IN.",
            "instruction_for_llm": "Response language successfully updated to en-IN. Please acknowledge this and answer the user's overarching request in this new language.",
        }
        manager._update_state_from_tool("set_language", raw_args, tool_result, state)

        # Verify language changed
        assert state.response_language == "en-IN"

        # Verify all business context is retained
        assert state.current_scheme == "pm_kisan"
        assert state.current_workflow == WorkflowState.ELIGIBILITY
        assert state.collected_slots == {"land_ownership_hectares": 1.5}
        assert state.required_slots == ["is_institutional_landholder", "pays_income_tax"]

        # Verify system prompt generated contains updated language & retained state
        sys_msg = build_system_message(state)
        assert "response_language: en-IN" in sys_msg["content"]
        assert "current_scheme:    pm_kisan" in sys_msg["content"]
        assert "current_workflow:  ELIGIBILITY" in sys_msg["content"]
        assert "land_ownership_hectares: 1.5" in sys_msg["content"]

    @pytest.mark.asyncio
    async def test_full_turn_language_switch_orchestration(self):
        """
        Simulate a full turn where LLM decides to call set_language.
        """
        manager = ConversationManager()
        state = ConversationState(
            session_id="sess_full_turn",
            response_language="hi-IN",
            current_scheme="pm_kisan",
            current_workflow=WorkflowState.ELIGIBILITY,
        )

        # Mock LLM orchestrate to return a set_language ToolCallRequest
        mock_tool_request = ToolCallRequest(
            tool_call_id="call_lang_123",
            tool_name="set_language",
            raw_arguments=json.dumps({"language_code": "en-IN"}),
        )

        async def mock_stream_chunks(messages):
            yield "Sure, I have switched to English. "
            yield "Regarding PM-KISAN, you are eligible for ₹6,000 annually."

        with patch.object(manager._llm, "orchestrate", new_callable=AsyncMock) as mock_orch, \
             patch.object(manager._llm, "stream_after_tool", side_effect=mock_stream_chunks):
            mock_orch.return_value = mock_tool_request

            chunks = []
            async for chunk in manager.run_turn(
                user_text="Please speak in English",
                detected_lang="en-IN",
                state=state,
                generation_id=1,
            ):
                chunks.append(chunk)

            full_response = "".join(chunks)
            assert "switched to English" in full_response
            assert state.response_language == "en-IN"
            assert state.current_scheme == "pm_kisan"
            assert state.current_workflow == WorkflowState.ELIGIBILITY
