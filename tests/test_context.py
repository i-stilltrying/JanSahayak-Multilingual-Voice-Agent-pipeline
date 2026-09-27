"""
tests/test_context.py

Unit tests for conversation context / memory behaviour.
Verifies that the agent manager correctly wires state between turns.
No Sarvam API calls — exercises only local state management.
"""

import pytest
from backend.agent.state import ConversationState, WorkflowState
from backend.agent.prompts import build_system_message


class TestContextFollowUp:
    """
    Spec §17 — context follow-up behaviour.
    The structured state must carry scheme/workflow across turns.
    """

    def test_scheme_persists_after_second_turn(self):
        """If user asks about PM-KISAN then 'documents', state retains pm_kisan."""
        state = ConversationState(session_id="ctx-test")
        state.current_scheme = "pm_kisan"
        state.current_workflow = WorkflowState.FAQ
        state.add_user_message("PM Kisan kya hai?")
        state.add_assistant_message("PM-KISAN is a farmer income support scheme.")

        # Second turn without repeating scheme
        state.current_workflow = WorkflowState.DOCUMENTS
        # scheme should still be pm_kisan
        assert state.current_scheme == "pm_kisan"

    def test_collected_slots_persist(self):
        state = ConversationState(session_id="ctx-test")
        state.collected_slots["state"] = "Uttar Pradesh"
        state.add_user_message("I'm from UP.")
        state.add_assistant_message("Got it, you're from Uttar Pradesh.")

        state.collected_slots["age"] = 45
        state.add_user_message("My age is 45.")
        state.add_assistant_message("Great.")

        # Both slots should still be there
        assert state.collected_slots["state"] == "Uttar Pradesh"
        assert state.collected_slots["age"] == 45

    def test_language_switch_persists(self):
        state = ConversationState(session_id="ctx-test")
        state.set_response_language("en-IN")
        state.current_scheme = "pm_kisan"

        # Language should not reset on scheme change
        assert state.response_language == "en-IN"
        assert state.current_scheme == "pm_kisan"


class TestSystemMessageContext:
    """Verify that the system message correctly reflects state."""

    def test_current_scheme_in_prompt(self):
        state = ConversationState(session_id="sys-test")
        state.current_scheme = "ayushman_bharat"
        msg = build_system_message(state)
        assert "ayushman_bharat" in msg["content"]

    def test_current_workflow_in_prompt(self):
        state = ConversationState(session_id="sys-test")
        state.current_workflow = WorkflowState.ELIGIBILITY
        msg = build_system_message(state)
        assert "ELIGIBILITY" in msg["content"]

    def test_collected_slots_in_prompt(self):
        state = ConversationState(session_id="sys-test")
        state.collected_slots = {"state": "Kerala", "age": 30}
        msg = build_system_message(state)
        assert "Kerala" in msg["content"]
        assert "30" in msg["content"]

    def test_response_language_in_prompt(self):
        state = ConversationState(session_id="sys-test")
        state.response_language = "kn-IN"
        msg = build_system_message(state)
        assert "kn-IN" in msg["content"]

    def test_workflow_hint_eligibility(self):
        state = ConversationState(session_id="sys-test")
        state.current_workflow = WorkflowState.ELIGIBILITY
        msg = build_system_message(state)
        # Eligibility workflow hint should appear
        assert "check_eligibility" in msg["content"] or "eligibility" in msg["content"].lower()

    def test_workflow_hint_callback(self):
        state = ConversationState(session_id="sys-test")
        state.current_workflow = WorkflowState.CALLBACK
        msg = build_system_message(state)
        assert "callback" in msg["content"].lower() or "book_human_callback" in msg["content"]

    def test_no_scheme_shows_none(self):
        state = ConversationState(session_id="sys-test")
        msg = build_system_message(state)
        assert "(none)" in msg["content"]


class TestSessionIsolation:
    """Spec §162 — state must never leak across sessions."""

    def test_two_sessions_independent(self):
        state_a = ConversationState(session_id="session-A")
        state_b = ConversationState(session_id="session-B")

        state_a.current_scheme = "pm_kisan"
        state_a.add_user_message("PM Kisan question")

        state_b.current_scheme = "ayushman_bharat"
        state_b.add_user_message("Ayushman question")

        # Neither session should see the other's data
        assert state_a.current_scheme == "pm_kisan"
        assert state_b.current_scheme == "ayushman_bharat"
        assert len(state_a.conversation_history) == 1
        assert len(state_b.conversation_history) == 1
        assert state_a.conversation_history[0]["content"] != state_b.conversation_history[0]["content"]
