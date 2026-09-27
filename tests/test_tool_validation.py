"""
tests/test_tool_validation.py

Unit tests for tool argument validation (Phase 7 tools.py).
Validates that Pydantic models reject bad arguments and accept good ones.
No Sarvam API calls.
"""

import json
import pytest
import pytest_asyncio
import asyncio

from backend.agent.tools import (
    SearchKnowledgeArgs,
    GetDocumentsArgs,
    GetApplicationStepsArgs,
    dispatch_tool,
    execute_search_knowledge,
    execute_get_required_documents,
    execute_get_application_steps,
    TOOL_DEFINITIONS,
)


class TestToolDefinitions:
    def test_tool_definitions_is_list(self):
        assert isinstance(TOOL_DEFINITIONS, list)
        assert len(TOOL_DEFINITIONS) == 8

    def test_tool_names_present(self):
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        expected = {
            "search_knowledge",
            "check_eligibility",
            "get_required_documents",
            "get_application_steps",
            "get_application_status",
            "get_callback_slots",
            "book_human_callback",
            "set_language",
        }
        assert names == expected

    def test_each_tool_has_type_function(self):
        for t in TOOL_DEFINITIONS:
            assert t["type"] == "function"

    def test_search_knowledge_has_required_query(self):
        tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "search_knowledge")
        params = tool["function"]["parameters"]
        assert "query" in params["required"]


class TestSearchKnowledgeArgs:
    def test_valid_args(self):
        args = SearchKnowledgeArgs(query="PM Kisan kya hai")
        assert args.query == "PM Kisan kya hai"
        assert args.scheme is None

    def test_valid_with_scheme(self):
        args = SearchKnowledgeArgs(query="eligibility", scheme="pm_kisan")
        assert args.scheme == "pm_kisan"

    def test_valid_with_topic(self):
        args = SearchKnowledgeArgs(query="docs", scheme="pm_kisan", topic="documents")
        assert args.topic == "documents"

    def test_invalid_scheme_rejected(self):
        with pytest.raises(Exception):
            SearchKnowledgeArgs(query="test", scheme="unknown_scheme_xyz")

    def test_invalid_topic_rejected(self):
        with pytest.raises(Exception):
            SearchKnowledgeArgs(query="test", topic="invalid_topic")

    def test_empty_query_rejected(self):
        with pytest.raises(Exception):
            SearchKnowledgeArgs(query="")

    def test_all_valid_schemes(self):
        for scheme in ["pm_kisan", "ayushman_bharat", "pm_awas", "atal_pension", "mgnrega"]:
            args = SearchKnowledgeArgs(query="test", scheme=scheme)
            assert args.scheme == scheme

    def test_all_valid_topics(self):
        for topic in ["overview", "key_facts", "eligibility", "documents", "application", "faq"]:
            args = SearchKnowledgeArgs(query="test", topic=topic)
            assert args.topic == topic


class TestGetDocumentsArgs:
    def test_valid(self):
        args = GetDocumentsArgs(scheme="pm_kisan")
        assert args.scheme == "pm_kisan"

    def test_invalid_scheme(self):
        with pytest.raises(Exception):
            GetDocumentsArgs(scheme="bad_scheme")


class TestDispatchTool:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        result = await dispatch_tool("nonexistent_tool", '{}')
        assert result["success"] is False
        assert result["error"] == "unknown_tool"

    @pytest.mark.asyncio
    async def test_search_knowledge_bad_json(self):
        result = await dispatch_tool("search_knowledge", "NOT_JSON")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_search_knowledge_valid(self):
        args = json.dumps({"query": "PM Kisan scheme overview"})
        result = await dispatch_tool("search_knowledge", args)
        assert "success" in result
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_search_knowledge_with_scheme(self):
        args = json.dumps({"query": "eligibility", "scheme": "pm_kisan"})
        result = await dispatch_tool("search_knowledge", args)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_get_required_documents_valid(self):
        args = json.dumps({"scheme": "pm_kisan"})
        result = await dispatch_tool("get_required_documents", args)
        assert result["success"] is True
        assert "documents" in result

    @pytest.mark.asyncio
    async def test_get_application_steps_valid(self):
        args = json.dumps({"scheme": "ayushman_bharat"})
        result = await dispatch_tool("get_application_steps", args)
        assert result["success"] is True
        assert "steps" in result

    @pytest.mark.asyncio
    async def test_set_language_valid(self):
        args = json.dumps({"language_code": "en-IN"})
        result = await dispatch_tool("set_language", args)
        assert result["success"] is True
        assert result["language_code"] == "en-IN"

    @pytest.mark.asyncio
    async def test_check_eligibility_dispatch(self):
        """check_eligibility is now implemented in Phase 9."""
        args = json.dumps({"scheme_id": "pm_kisan", "provided_slots": {"land_ownership_hectares": 1.5, "is_institutional_landholder": False, "pays_income_tax": False}})
        result = await dispatch_tool("check_eligibility", args)
        assert result["success"] is True
        assert result["status"] == "ELIGIBLE"
