"""
backend/knowledge/retriever.py

In-memory keyword-matching knowledge retriever.

Design (spec §78–80):
  - Loads all 5 JSON scheme files into memory at startup.
  - Retrieval is a deterministic weighted scoring function — no embeddings,
    no vector DB. For ~5 schemes and ~50 chunks this is instant (< 1 ms).
  - Returns the top 2–3 KnowledgeChunk objects.

Scoring weights:
  Exact scheme_id match          : 100 pts
  Exact scheme_name match        : 80  pts
  Keyword exact match            : 20  pts each (capped at 60)
  Keyword partial match          : 8   pts each (capped at 24)
  Topic match (if topic given)   : 30  pts
  Query term in content          : 5   pts each (capped at 20)

The KB is loaded once at module import and reused across all sessions.
A module-level singleton `retriever` is exported for use by the tool layer.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import ClassVar

from backend.knowledge.schemas import (
    FaqEntry,
    KnowledgeChunk,
    SchemeRecord,
    SearchResult,
)

logger = logging.getLogger(__name__)

# Path to the knowledge JSON files (relative to project root)
_KB_DIR = Path(__file__).resolve().parents[2] / "knowledge"

# Maximum chunks returned per search (spec §80)
MAX_CHUNKS = 3

# Maximum characters per chunk content (keeps LLM context small)
MAX_CONTENT_CHARS = 600


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _truncate(text: str, max_chars: int = MAX_CONTENT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


class KnowledgeRetriever:
    """
    Singleton in-memory knowledge retriever.

    Usage:
        from backend.knowledge.retriever import retriever
        result = retriever.search("PM Kisan kya hai", scheme="pm_kisan")
    """

    # Class-level cache so multiple sessions share one loaded copy.
    _schemes: ClassVar[dict[str, SchemeRecord]] = {}
    _loaded: ClassVar[bool] = False

    def __init__(self) -> None:
        if not KnowledgeRetriever._loaded:
            self._load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load every *.json file in the knowledge directory."""
        count = 0
        for path in sorted(_KB_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                record = SchemeRecord.model_validate(data)
                KnowledgeRetriever._schemes[record.scheme_id] = record
                count += 1
                logger.info("[KB] Loaded scheme: %s (%s)", record.scheme_id, record.scheme_name)
            except Exception as exc:
                logger.error("[KB] Failed to load %s: %s", path.name, exc)

        KnowledgeRetriever._loaded = True
        logger.info("[KB] Knowledge base ready — %d schemes loaded", count)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        scheme: str | None = None,
        topic: str | None = None,
        max_chunks: int = MAX_CHUNKS,
    ) -> SearchResult:
        """
        Return the top `max_chunks` KnowledgeChunks for the given query.

        Args:
            query  : The user's question (in any language — we match on keywords).
            scheme : Optional scheme_id hint (e.g. "pm_kisan").
            topic  : Optional topic filter ("overview"|"eligibility"|"documents"|
                     "application"|"faq"|"key_facts").
            max_chunks: Maximum chunks to return (default 3).

        Returns:
            SearchResult with ranked chunks.
        """
        norm_query = _normalize(query)
        query_terms = set(norm_query.split())

        # Determine candidate schemes
        if scheme and scheme in self._schemes:
            candidates = {scheme: self._schemes[scheme]}
            matched_scheme = scheme
        else:
            candidates = self._schemes
            # Try to detect scheme from query
            matched_scheme = self._detect_scheme(norm_query, query_terms)

        scored: list[tuple[float, KnowledgeChunk]] = []

        for sid, record in candidates.items():
            base_score = self._scheme_score(sid, record, norm_query, query_terms)
            chunks = self._build_chunks(record, topic)
            for chunk in chunks:
                topic_bonus = 30.0 if (topic and chunk.topic == topic) else 0.0
                content_score = self._content_score(chunk.content, query_terms)
                total = base_score + topic_bonus + content_score
                scored.append((total, chunk))

        # Sort descending by score, take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [chunk for _, chunk in scored[:max_chunks] if _ > 0]

        return SearchResult(
            query=query,
            chunks=top,
            total_found=len(top),
            scheme_matched=matched_scheme,
        )

    def get_scheme(self, scheme_id: str) -> SchemeRecord | None:
        """Direct lookup by scheme_id — used by eligibility/documents tools."""
        return self._schemes.get(scheme_id)

    def get_scheme_documents(self, scheme_id: str) -> dict[str, Any] | None:
        """
        Direct document lookup for a scheme.
        Returns scheme name, list of required documents, and official source URL.
        """
        record = self.get_scheme(scheme_id)
        if not record:
            return None
        return {
            "scheme": record.scheme_name,
            "documents": record.documents,
            "source": record.source_name,
            "source_url": record.official_source_url,
            "last_verified": record.last_verified,
        }

    def get_scheme_application_steps(self, scheme_id: str) -> dict[str, Any] | None:
        """
        Direct application process lookup for a scheme.
        Returns scheme name, application steps, and official source URL.
        """
        record = self.get_scheme(scheme_id)
        if not record:
            return None
        return {
            "scheme": record.scheme_name,
            "steps": record.application_steps,
            "official_source": record.official_source_url,
            "source": record.source_name,
            "last_verified": record.last_verified,
        }

    def list_scheme_ids(self) -> list[str]:
        return list(self._schemes.keys())

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _detect_scheme(self, norm_query: str, query_terms: set[str]) -> str | None:
        """Return scheme_id with highest keyword overlap, or None."""
        best_id: str | None = None
        best_score = 0.0
        for sid, record in self._schemes.items():
            score = self._scheme_score(sid, record, norm_query, query_terms)
            if score > best_score:
                best_score = score
                best_id = sid
        return best_id if best_score > 5 else None

    def _scheme_score(
        self,
        scheme_id: str,
        record: SchemeRecord,
        norm_query: str,
        query_terms: set[str],
    ) -> float:
        score = 0.0

        # Exact scheme_id match
        if scheme_id == _normalize(norm_query.replace(" ", "_")):
            score += 100

        # Scheme name in query
        norm_name = _normalize(record.scheme_name)
        if norm_name and norm_name in norm_query:
            score += 80
        elif norm_name and any(t in norm_query for t in norm_name.split()):
            score += 30

        # Keyword matches (capped)
        kw_exact = 0
        kw_partial = 0
        for kw in record.keywords:
            norm_kw = _normalize(kw)
            if norm_kw in norm_query:
                kw_exact += 1
            elif any(t in norm_kw for t in query_terms if len(t) > 3):
                kw_partial += 1
        score += min(kw_exact * 20, 60)
        score += min(kw_partial * 8, 24)

        return score

    def _content_score(self, content: str, query_terms: set[str]) -> float:
        norm_content = _normalize(content)
        hits = sum(1 for t in query_terms if len(t) > 3 and t in norm_content)
        return min(hits * 5, 20)

    # ------------------------------------------------------------------
    # Chunk builders
    # ------------------------------------------------------------------

    def _build_chunks(
        self, record: SchemeRecord, topic_filter: str | None
    ) -> list[KnowledgeChunk]:
        """
        Build all retrievable chunks for one scheme.
        Each chunk is a focused excerpt on one topic.
        """
        meta = dict(
            scheme_id=record.scheme_id,
            scheme_name=record.scheme_name,
            source_url=record.official_source_url,
            source_name=record.source_name,
            last_verified=record.last_verified,
        )
        chunks: list[KnowledgeChunk] = []

        def _add(topic: str, content: str) -> None:
            if topic_filter and topic != topic_filter:
                return
            content = content.strip()
            if content:
                chunks.append(KnowledgeChunk(
                    topic=topic,
                    content=_truncate(content),
                    **meta,
                ))

        # Overview
        _add("overview", record.overview)

        # Key facts — join into a readable paragraph
        if record.key_facts:
            facts_text = record.full_name + " — Key Facts: " + " | ".join(record.key_facts)
            _add("key_facts", facts_text)

        # Eligibility
        elig = record.eligibility
        if elig.description:
            excl = ""
            if elig.exclusions:
                excl = " Not eligible: " + "; ".join(elig.exclusions[:4]) + "."
            notes = (" " + elig.notes) if elig.notes else ""
            _add("eligibility", elig.description + excl + notes)

        # Documents
        if record.documents:
            _add("documents", "Required documents: " + ", ".join(record.documents))

        # Application steps
        if record.application_steps:
            steps_text = " ".join(record.application_steps[:4])
            _add("application", steps_text)

        # FAQs — merge top 2 as a single chunk
        if record.faqs:
            faq_text = " ".join(
                f"Q: {f.question} A: {f.answer}"
                for f in record.faqs[:2]
            )
            _add("faq", faq_text)

        return chunks


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------

retriever = KnowledgeRetriever()
