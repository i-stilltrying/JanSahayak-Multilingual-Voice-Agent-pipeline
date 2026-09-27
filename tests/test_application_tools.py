"""
tests/test_application_tools.py

Phase 10: Tests for documents, application steps, and demo application status tools.
"""

from __future__ import annotations

import json
import os
import pytest

# Ensure SARVAM_API_KEY is set for test environment
os.environ.setdefault("SARVAM_API_KEY", "dummy_test_key")

from backend.knowledge.retriever import retriever
from backend.agent.tools import dispatch_tool
from backend.agent.manager import ConversationManager
from backend.agent.state import ConversationState, WorkflowState


# ---------------------------------------------------------------------------
# Retriever Helper Method Tests
# ---------------------------------------------------------------------------

class TestRetrieverHelpers:
    def test_get_scheme_documents_valid(self):
        docs = retriever.get_scheme_documents("pm_kisan")
        assert docs is not None
        assert docs["scheme"] == "PM-KISAN"
        assert len(docs["documents"]) > 0
        assert any("Aadhaar" in d for d in docs["documents"])
        assert docs["source_url"] == "https://pmkisan.gov.in"

    def test_get_scheme_documents_invalid(self):
        docs = retriever.get_scheme_documents("invalid_scheme")
        assert docs is None

    def test_get_scheme_application_steps_valid(self):
        steps = retriever.get_scheme_application_steps("ayushman_bharat")
        assert steps is not None
        assert steps["scheme"] == "Ayushman Bharat PM-JAY"
        assert len(steps["steps"]) > 0
        assert steps["official_source"] == "https://pmjay.gov.in"

    def test_get_scheme_application_steps_invalid(self):
        steps = retriever.get_scheme_application_steps("invalid_scheme")
        assert steps is None


# ---------------------------------------------------------------------------
# Tool: get_required_documents Tests (Spec §35)
# ---------------------------------------------------------------------------

class TestGetRequiredDocumentsTool:
    @pytest.mark.asyncio
    async def test_get_required_documents_success(self):
        payload = json.dumps({"scheme_id": "pm_kisan"})
        result = await dispatch_tool("get_required_documents", payload)
        assert result["success"] is True
        assert result["scheme"] == "PM-KISAN"
        assert isinstance(result["documents"], list)
        assert len(result["documents"]) > 0
        assert "source_url" in result

    @pytest.mark.asyncio
    async def test_get_required_documents_alias_scheme(self):
        # Test backward-compatibility alias "scheme"
        payload = json.dumps({"scheme": "ayushman_bharat"})
        result = await dispatch_tool("get_required_documents", payload)
        assert result["success"] is True
        assert result["scheme"] == "Ayushman Bharat PM-JAY"

    @pytest.mark.asyncio
    async def test_get_required_documents_invalid_scheme(self):
        payload = json.dumps({"scheme_id": "invalid_scheme"})
        result = await dispatch_tool("get_required_documents", payload)
        assert result["success"] is False
        assert result["error"] == "invalid_arguments"

    @pytest.mark.asyncio
    async def test_get_required_documents_all_schemes(self):
        for sid in ["pm_kisan", "ayushman_bharat", "pm_awas", "atal_pension", "mgnrega"]:
            payload = json.dumps({"scheme_id": sid})
            result = await dispatch_tool("get_required_documents", payload)
            assert result["success"] is True
            assert len(result["documents"]) > 0


# ---------------------------------------------------------------------------
# Tool: get_application_steps Tests (Spec §36)
# ---------------------------------------------------------------------------

class TestGetApplicationStepsTool:
    @pytest.mark.asyncio
    async def test_get_application_steps_success(self):
        payload = json.dumps({"scheme_id": "pm_awas"})
        result = await dispatch_tool("get_application_steps", payload)
        assert result["success"] is True
        assert result["scheme"] == "PM Awas Yojana"
        assert isinstance(result["steps"], list)
        assert len(result["steps"]) > 0
        assert "official_source" in result
        assert "pmaymis.gov.in" in result["official_source"]

    @pytest.mark.asyncio
    async def test_get_application_steps_invalid_scheme(self):
        payload = json.dumps({"scheme_id": "nonexistent"})
        result = await dispatch_tool("get_application_steps", payload)
        assert result["success"] is False
        assert result["error"] == "invalid_arguments"

    @pytest.mark.asyncio
    async def test_get_application_steps_all_schemes(self):
        for sid in ["pm_kisan", "ayushman_bharat", "pm_awas", "atal_pension", "mgnrega"]:
            payload = json.dumps({"scheme_id": sid})
            result = await dispatch_tool("get_application_steps", payload)
            assert result["success"] is True
            assert len(result["steps"]) > 0
            assert result["official_source"] != ""


# ---------------------------------------------------------------------------
# Tool: get_application_status Tests (Spec §37, §89)
# ---------------------------------------------------------------------------

class TestGetApplicationStatusTool:
    @pytest.mark.asyncio
    async def test_get_application_status_found(self):
        payload = json.dumps({"application_id": "APP-12345"})
        result = await dispatch_tool("get_application_status", payload)
        assert result["success"] is True
        assert result["found"] is True
        assert result["application_id"] == "APP-12345"
        assert result["scheme"] == "PM-KISAN"
        assert result["status"] == "Under Review"
        assert "warning" in result
        assert "Demo application lookup" in result["warning"]

    @pytest.mark.asyncio
    async def test_get_application_status_case_insensitive(self):
        payload = json.dumps({"application_id": "app-67890"})
        result = await dispatch_tool("get_application_status", payload)
        assert result["success"] is True
        assert result["found"] is True
        assert result["status"] == "Approved"

    @pytest.mark.asyncio
    async def test_get_application_status_not_found(self):
        payload = json.dumps({"application_id": "APP-99999"})
        result = await dispatch_tool("get_application_status", payload)
        assert result["success"] is True
        assert result["found"] is False
        assert "No application record found" in result["message"]
        assert "warning" in result

    @pytest.mark.asyncio
    async def test_get_application_status_empty_id(self):
        payload = json.dumps({"application_id": ""})
        result = await dispatch_tool("get_application_status", payload)
        assert result["success"] is False
        assert result["error"] == "invalid_arguments"


# ---------------------------------------------------------------------------
# Workflow and State Transitions in ConversationManager
# ---------------------------------------------------------------------------

class TestManagerWorkflowTransitions:
    def test_workflow_transition_to_documents(self):
        manager = ConversationManager()
        state = ConversationState(session_id="sess_doc")

        raw_args = json.dumps({"scheme_id": "pm_kisan"})
        tool_result = {"success": True, "scheme": "PM-KISAN", "documents": ["Aadhaar"]}
        manager._update_state_from_tool("get_required_documents", raw_args, tool_result, state)

        assert state.current_workflow == WorkflowState.DOCUMENTS
        assert state.current_scheme == "pm_kisan"

    def test_workflow_transition_to_application(self):
        manager = ConversationManager()
        state = ConversationState(session_id="sess_app")

        raw_args = json.dumps({"scheme_id": "atal_pension"})
        tool_result = {"success": True, "scheme": "Atal Pension Yojana", "steps": ["Step 1"]}
        manager._update_state_from_tool("get_application_steps", raw_args, tool_result, state)

        assert state.current_workflow == WorkflowState.APPLICATION
        assert state.current_scheme == "atal_pension"

    def test_workflow_transition_to_application_status(self):
        manager = ConversationManager()
        state = ConversationState(session_id="sess_status")

        raw_args = json.dumps({"application_id": "APP-12345"})
        tool_result = {
            "success": True,
            "found": True,
            "application_id": "APP-12345",
            "scheme": "PM-KISAN",
            "status": "Under Review",
        }
        manager._update_state_from_tool("get_application_status", raw_args, tool_result, state)

        assert state.current_workflow == WorkflowState.APPLICATION_STATUS
        assert state.current_scheme == "PM-KISAN"
