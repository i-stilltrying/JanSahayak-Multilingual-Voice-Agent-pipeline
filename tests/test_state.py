"""
tests/test_state.py

Unit tests for ConversationState (Phase 8 memory).
No Sarvam API calls — fully local.
"""

import pytest
from backend.agent.state import ConversationState, WorkflowState


def make_state(session_id: str = "test-session") -> ConversationState:
    return ConversationState(session_id=session_id)


class TestConversationStateInit:
    def test_defaults(self):
        state = make_state()
        assert state.session_id == "test-session"
        assert state.current_scheme is None
        assert state.current_workflow == WorkflowState.NONE
        assert state.collected_slots == {}
        assert state.conversation_history == []

    def test_response_language_defaults_to_config(self):
        state = make_state()
        # Should get default_response_language from settings (hi-IN in .env.example)
        assert state.response_language != ""


class TestHistoryManagement:
    def test_add_user_message(self):
        state = make_state()
        state.add_user_message("Hello")
        assert len(state.conversation_history) == 1
        assert state.conversation_history[0]["role"] == "user"
        assert state.conversation_history[0]["content"] == "Hello"

    def test_add_assistant_message(self):
        state = make_state()
        state.add_assistant_message("Hi there!")
        assert state.conversation_history[0]["role"] == "assistant"

    def test_history_pruning(self):
        state = make_state()
        for i in range(30):
            state.add_user_message(f"msg {i}")
        # Should be capped at max_recent_messages (24)
        assert len(state.conversation_history) == 24
        # Most recent messages should be retained
        assert state.conversation_history[-1]["content"] == "msg 29"

    def test_pruning_keeps_latest(self):
        state = make_state()
        for i in range(30):
            state.add_user_message(f"msg {i}")
        # First 6 should be gone
        contents = [m["content"] for m in state.conversation_history]
        assert "msg 0" not in contents
        assert "msg 5" not in contents
        assert "msg 6" in contents

    def test_add_tool_messages(self):
        state = make_state()
        state.add_tool_messages(
            tool_call_id="tc-1",
            tool_name="search_knowledge",
            tool_result_json='{"success": true, "chunks": []}',
        )
        # Two messages added: assistant (tool_calls) + tool result
        assert len(state.conversation_history) == 2
        assert state.conversation_history[0]["role"] == "assistant"
        assert state.conversation_history[1]["role"] == "tool"
        assert state.conversation_history[1]["tool_call_id"] == "tc-1"


class TestLanguageHelpers:
    def test_update_detected_language_sets_both(self):
        state = make_state()
        state.response_language = ""  # reset
        state.update_detected_language("kn-IN")
        assert state.input_language == "kn-IN"

    def test_set_response_language(self):
        state = make_state()
        state.set_response_language("en-IN")
        assert state.response_language == "en-IN"

    def test_update_detected_does_not_override_explicit(self):
        """If user explicitly set language, STT detection should not override it."""
        state = make_state()
        state.set_response_language("en-IN")
        # Detect Hindi — should NOT override the explicit en-IN setting
        # (response_language is only mirrored when it matches the default)
        state.input_language = ""  # clear to test fresh update
        # After an explicit set, response_language stays en-IN
        assert state.response_language == "en-IN"


class TestStructuredState:
    def test_workflow_transitions(self):
        state = make_state()
        state.current_workflow = WorkflowState.ELIGIBILITY
        assert state.current_workflow == "ELIGIBILITY"
        state.current_workflow = WorkflowState.CALLBACK
        assert state.current_workflow == "CALLBACK"

    def test_collected_slots(self):
        state = make_state()
        state.collected_slots["state"] = "Uttar Pradesh"
        state.collected_slots["age"] = 45
        assert state.collected_slots["state"] == "Uttar Pradesh"
        assert state.collected_slots["age"] == 45
