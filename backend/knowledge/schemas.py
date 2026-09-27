"""
backend/knowledge/schemas.py

Pydantic models for the JanSahayak knowledge base.

Each scheme JSON file is loaded into a SchemeRecord.
The retriever returns KnowledgeChunk objects — small, focused excerpts
sized for the LLM context (spec §80: return top 2–3 chunks, not the full file).
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Stored schema — mirrors the JSON file structure
# ---------------------------------------------------------------------------


class EligibilityRule(BaseModel):
    rule_id: str
    description: str
    field: str
    required_value: str | None = None
    min_value: float | None = None
    max_value: float | None = None


class EligibilityInfo(BaseModel):
    description: str = ""
    required_slots: list[str] = Field(default_factory=list)
    rules: list[EligibilityRule] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    notes: str = ""
    how_to_check: str = ""
    income_categories: dict[str, str] = Field(default_factory=dict)


class FaqEntry(BaseModel):
    question: str
    answer: str


class SchemeRecord(BaseModel):
    """One complete scheme loaded from a JSON file in knowledge/."""

    scheme_id: str
    scheme_name: str
    full_name: str = ""
    ministry: str = ""
    overview: str = ""
    key_facts: list[str] = Field(default_factory=list)
    eligibility: EligibilityInfo = Field(default_factory=EligibilityInfo)
    documents: list[str] = Field(default_factory=list)
    application_steps: list[str] = Field(default_factory=list)
    faqs: list[FaqEntry] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    official_source_url: str = ""
    source_name: str = ""
    last_verified: str = ""

    # Allow extra fields — forward-compatible with future JSON additions.
    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Retrieval output — what the tool returns to the LLM
# ---------------------------------------------------------------------------


class KnowledgeChunk(BaseModel):
    """
    One focused excerpt returned by search_knowledge.

    Sized for LLM consumption: concise content, explicit source metadata.
    The LLM must use this as its factual basis — spec §81, §100.
    """

    scheme_id: str
    scheme_name: str
    topic: str           # "overview" | "key_facts" | "eligibility" | "documents" | "application" | "faq"
    content: str         # The actual text the LLM should use
    source_url: str = ""
    source_name: str = ""
    last_verified: str = ""


class SearchResult(BaseModel):
    """Return value of search_knowledge tool."""

    query: str
    chunks: list[KnowledgeChunk]
    total_found: int
    scheme_matched: str | None = None   # Which scheme was matched, if any
