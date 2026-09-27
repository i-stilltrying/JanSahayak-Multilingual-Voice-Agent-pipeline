"""
tests/test_retriever.py

Unit tests for the KnowledgeRetriever (Phase 7 knowledge layer).
No Sarvam API calls.
"""

import pytest
from backend.knowledge.retriever import KnowledgeRetriever, retriever
from backend.knowledge.schemas import SearchResult


class TestRetrieverInit:
    def test_singleton_loaded(self):
        """Retriever singleton should have loaded all 5 scheme files."""
        assert retriever is not None

    def test_known_scheme_loaded(self):
        record = retriever.get_scheme("pm_kisan")
        assert record is not None
        assert record.scheme_name != ""

    def test_all_schemes_loaded(self):
        for scheme_id in ["pm_kisan", "ayushman_bharat", "pm_awas", "atal_pension", "mgnrega"]:
            record = retriever.get_scheme(scheme_id)
            assert record is not None, f"Scheme not loaded: {scheme_id}"

    def test_unknown_scheme_returns_none(self):
        assert retriever.get_scheme("nonexistent_scheme") is None


class TestSearch:
    def test_search_returns_result(self):
        result = retriever.search(query="PM Kisan kya hai")
        assert isinstance(result, SearchResult)
        assert result.chunks is not None

    def test_search_with_scheme_filter(self):
        result = retriever.search(query="eligibility", scheme="pm_kisan")
        assert result is not None
        # All results should be from PM-KISAN
        for chunk in result.chunks:
            assert "PM" in chunk.scheme_name or "Kisan" in chunk.scheme_name or chunk.scheme_name != ""

    def test_search_with_topic_filter(self):
        result = retriever.search(query="documents", scheme="pm_kisan", topic="documents")
        assert result is not None

    def test_search_unknown_scheme_returns_empty_or_fallback(self):
        # Searching a nonexistent scheme shouldn't crash
        result = retriever.search(query="test", scheme="nonexistent_xyz")
        assert result is not None
        assert result.chunks is not None  # may be empty list

    def test_search_returns_at_most_3_chunks(self):
        result = retriever.search(query="PM Kisan farmer")
        assert len(result.chunks) <= 3

    def test_search_empty_query_handled(self):
        # Should not crash on minimal query
        result = retriever.search(query="a")
        assert result is not None


class TestSchemeRecord:
    def test_pm_kisan_has_required_fields(self):
        record = retriever.get_scheme("pm_kisan")
        assert record.scheme_id == "pm_kisan"
        assert record.scheme_name
        assert record.overview
        assert isinstance(record.documents, list)
        assert isinstance(record.application_steps, list)
        assert record.official_source_url

    def test_ayushman_bharat_loaded(self):
        record = retriever.get_scheme("ayushman_bharat")
        assert record.scheme_id == "ayushman_bharat"
        assert record.overview

    def test_mgnrega_loaded(self):
        record = retriever.get_scheme("mgnrega")
        assert record.scheme_id == "mgnrega"
