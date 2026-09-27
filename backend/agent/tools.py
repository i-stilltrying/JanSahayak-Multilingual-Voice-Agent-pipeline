"""
backend/agent/tools.py

Phase 7: Tool definitions and Python executors for JanSahayak.

Architecture (spec §32, §23, §101):
  - TOOL_DEFINITIONS: the JSON schema list passed to the LLM as `tools=`.
  - Each tool has a corresponding async Python executor function.
  - The executor validates arguments using Pydantic (spec §31), executes
    deterministic Python business logic, and returns a JSON-serialisable dict.
  - The LLM must never independently decide eligibility or fabricate facts.

Phase 7 tools:
  search_knowledge   — query the in-memory KB (spec §33)

Implemented tools:
  search_knowledge       — query the in-memory KB (spec §33)
  check_eligibility      — deterministic rules engine (spec §34, Phase 9)
  get_required_documents — direct verified document requirements (spec §35, Phase 10)
  get_application_steps  — official application process steps (spec §36, Phase 10)
  get_application_status — mock demo application status lookup (spec §37, §89, Phase 10)
  set_language           — explicit response language switching (spec §44–47, Phase 11)
  get_callback_slots     — fetch available human callback slots (spec §38, Phase 12)
  book_human_callback    — atomic & idempotent callback booking (spec §39–42, Phase 12)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.agent.eligibility import check_scheme_eligibility
from backend.knowledge.retriever import retriever
from backend.persistence.callback_store import callback_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool JSON schemas — passed directly to the Sarvam LLM `tools=` parameter
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search the JanSahayak government scheme knowledge base for factual information. "
                "Use this for scheme overviews, FAQs, key facts, eligibility summaries, "
                "document requirements, and application process questions. "
                "Always call this before answering factual questions about government schemes. "
                "Do NOT call for greetings, goodbyes, or language requests."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question or search phrase (in any language)",
                    },
                    "scheme": {
                        "type": "string",
                        "description": (
                            "Optional scheme_id to narrow the search. "
                            "One of: pm_kisan, ayushman_bharat, pm_awas, atal_pension, mgnrega. "
                            "Omit if the scheme is unclear."
                        ),
                        "enum": ["pm_kisan", "ayushman_bharat", "pm_awas", "atal_pension", "mgnrega"],
                    },
                    "topic": {
                        "type": "string",
                        "description": (
                            "Optional topic filter to retrieve a specific aspect of the scheme."
                        ),
                        "enum": ["overview", "key_facts", "eligibility", "documents", "application", "faq"],
                    },
                },
                "required": ["query"],
            },
        },
    },
    # -----------------------------------------------------------------------
    # Stub placeholders — registered so the LLM sees them (prevents
    # hallucination about capabilities). Executors added in later phases.
    # -----------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "check_eligibility",
            "description": (
                "Evaluate whether the user is eligible for a specific scheme based on their "
                "provided information using deterministic business rules. If information is missing, "
                "the tool will indicate which specific slots to ask the user for."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scheme_id": {
                        "type": "string",
                        "description": "Scheme identifier (e.g. 'pm_kisan', 'ayushman_bharat', 'pm_awas', 'atal_pension', 'mgnrega')",
                        "enum": ["pm_kisan", "ayushman_bharat", "pm_awas", "atal_pension", "mgnrega"],
                    },
                    "provided_slots": {
                        "type": "object",
                        "description": "Key-value pairs of currently collected user slots/entities",
                    },
                },
                "required": ["scheme_id", "provided_slots"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_required_documents",
            "description": "Return the official list of documents required for a specific scheme.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scheme_id": {
                        "type": "string",
                        "description": "Scheme identifier (e.g. 'pm_kisan', 'ayushman_bharat', 'pm_awas', 'atal_pension', 'mgnrega')",
                        "enum": ["pm_kisan", "ayushman_bharat", "pm_awas", "atal_pension", "mgnrega"],
                    },
                },
                "required": ["scheme_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_application_steps",
            "description": "Return the official step-by-step application process for a specific scheme.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scheme_id": {
                        "type": "string",
                        "description": "Scheme identifier (e.g. 'pm_kisan', 'ayushman_bharat', 'pm_awas', 'atal_pension', 'mgnrega')",
                        "enum": ["pm_kisan", "ayushman_bharat", "pm_awas", "atal_pension", "mgnrega"],
                    },
                },
                "required": ["scheme_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_application_status",
            "description": "Look up the demo status of an existing scheme application using an application ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "application_id": {
                        "type": "string",
                        "description": "Application tracking reference ID (e.g. 'APP-12345')",
                    },
                },
                "required": ["application_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_callback_slots",
            "description": (
                "Return available dates and time slots for scheduling a human agent callback. "
                "Use this to show the user available options when they request to speak with a representative."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Optional specific date in YYYY-MM-DD format. If omitted, returns upcoming available slots.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_human_callback",
            "description": (
                "Atomically book an available callback slot with a human agent. "
                "Call this ONLY after the user has explicitly selected a specific date and time slot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date of callback in YYYY-MM-DD format",
                    },
                    "time_slot": {
                        "type": "string",
                        "description": "Time slot string (e.g. '10:00 AM' or '10:00 AM - 10:30 AM')",
                    },
                    "language": {
                        "type": "string",
                        "description": "Preferred callback language code (e.g. 'en-IN', 'hi-IN', 'kn-IN')",
                    },
                },
                "required": ["date", "time_slot", "language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_language",
            "description": "Switch the response language to the user's requested language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language_code": {
                        "type": "string",
                        "description": "BCP-47 language code: 'en-IN' (English), 'hi-IN' (Hindi), or 'kn-IN' (Kannada)",
                        "enum": ["en-IN", "hi-IN", "kn-IN"],
                    },
                },
                "required": ["language_code"],
            },
        },
    },
]

# Quick lookup by name for the dispatcher
_TOOL_MAP = {t["function"]["name"]: t for t in TOOL_DEFINITIONS}


# ---------------------------------------------------------------------------
# Pydantic argument models — validated before execution (spec §31)
# ---------------------------------------------------------------------------


class SearchKnowledgeArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    scheme: str | None = Field(default=None)
    topic: str | None = Field(default=None)

    @field_validator("scheme")
    @classmethod
    def valid_scheme(cls, v: str | None) -> str | None:
        valid = {"pm_kisan", "ayushman_bharat", "pm_awas", "atal_pension", "mgnrega", None}
        if v not in valid:
            raise ValueError(f"Unknown scheme: {v!r}")
        return v

    @field_validator("topic")
    @classmethod
    def valid_topic(cls, v: str | None) -> str | None:
        valid = {"overview", "key_facts", "eligibility", "documents", "application", "faq", None}
        if v not in valid:
            raise ValueError(f"Unknown topic: {v!r}")
        return v


class GetDocumentsArgs(BaseModel):
    scheme_id: str = Field(...)

    @property
    def scheme(self) -> str:
        return self.scheme_id

    @model_validator(mode="before")
    @classmethod
    def normalize_scheme(cls, data: Any) -> Any:
        if isinstance(data, dict):
            scheme = data.get("scheme_id") or data.get("scheme")
            return {"scheme_id": scheme}
        return data

    @field_validator("scheme_id")
    @classmethod
    def valid_scheme(cls, v: str) -> str:
        valid = {"pm_kisan", "ayushman_bharat", "pm_awas", "atal_pension", "mgnrega"}
        if v not in valid:
            raise ValueError(f"Unknown scheme: {v!r}")
        return v


class GetApplicationStepsArgs(BaseModel):
    scheme_id: str = Field(...)

    @property
    def scheme(self) -> str:
        return self.scheme_id

    @model_validator(mode="before")
    @classmethod
    def normalize_scheme(cls, data: Any) -> Any:
        if isinstance(data, dict):
            scheme = data.get("scheme_id") or data.get("scheme")
            return {"scheme_id": scheme}
        return data

    @field_validator("scheme_id")
    @classmethod
    def valid_scheme(cls, v: str) -> str:
        valid = {"pm_kisan", "ayushman_bharat", "pm_awas", "atal_pension", "mgnrega"}
        if v not in valid:
            raise ValueError(f"Unknown scheme: {v!r}")
        return v


class GetApplicationStatusArgs(BaseModel):
    application_id: str = Field(..., min_length=1, max_length=100)


class SetLanguageArgs(BaseModel):
    language_code: str = Field(...)

    @field_validator("language_code")
    @classmethod
    def valid_language(cls, v: str) -> str:
        valid = {"en-IN", "hi-IN", "kn-IN"}
        if v not in valid:
            raise ValueError(f"Unsupported language code: {v!r}. Supported: {sorted(valid)}")
        return v


class GetCallbackSlotsArgs(BaseModel):
    date: str | None = Field(default=None)


class BookCallbackArgs(BaseModel):
    date: str = Field(..., min_length=4)
    time_slot: str = Field(..., min_length=1)
    language: str = Field(default="en-IN")


class CheckEligibilityArgs(BaseModel):
    scheme_id: str = Field(...)
    provided_slots: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Support both scheme / scheme_id and user_data / provided_slots
            scheme = data.get("scheme_id") or data.get("scheme")
            slots = data.get("provided_slots") if "provided_slots" in data else data.get("user_data", {})
            if not isinstance(slots, dict):
                slots = {}
            return {"scheme_id": scheme, "provided_slots": slots}
        return data

    @field_validator("scheme_id")
    @classmethod
    def valid_scheme(cls, v: str) -> str:
        valid = {"pm_kisan", "ayushman_bharat", "pm_awas", "atal_pension", "mgnrega"}
        if v not in valid:
            raise ValueError(f"Unknown scheme: {v!r}")
        return v


# ---------------------------------------------------------------------------
# Executors — one async function per tool
# ---------------------------------------------------------------------------


async def execute_search_knowledge(raw_args: str) -> dict[str, Any]:
    """
    Execute the search_knowledge tool.
    Returns a dict ready for JSON serialisation.
    """
    try:
        parsed = json.loads(raw_args)
        args = SearchKnowledgeArgs.model_validate(parsed)
    except Exception as exc:
        logger.warning("[Tool] search_knowledge bad args %r: %s", raw_args[:200], exc)
        return {"success": False, "error": "invalid_arguments", "detail": str(exc)}

    result = retriever.search(
        query=args.query,
        scheme=args.scheme,
        topic=args.topic,
    )

    if not result.chunks:
        return {
            "success": True,
            "found": False,
            "message": "No verified information found for this query in the knowledge base.",
            "chunks": [],
        }

    return {
        "success": True,
        "found": True,
        "scheme_matched": result.scheme_matched,
        "chunks": [
            {
                "scheme": c.scheme_name,
                "topic": c.topic,
                "content": c.content,
                "source": c.source_name,
                "source_url": c.source_url,
                "last_verified": c.last_verified,
            }
            for c in result.chunks
        ],
    }


# ---------------------------------------------------------------------------
# Demo Application Status Store (Spec §37, §89)
# ---------------------------------------------------------------------------

_MOCK_APPLICATIONS: dict[str, dict[str, str]] = {
    "APP-12345": {
        "scheme": "PM-KISAN",
        "status": "Under Review",
        "last_updated": "2025-02-15",
        "remarks": "Document verification in progress at state nodal office.",
    },
    "APP-67890": {
        "scheme": "Ayushman Bharat PM-JAY",
        "status": "Approved",
        "last_updated": "2025-01-20",
        "remarks": "Ayushman Card is generated and active.",
    },
    "APP1": {
        "scheme": "PM-KISAN",
        "status": "Under Review",
        "last_updated": "2025-02-15",
        "remarks": "Document verification in progress at state nodal office.",
    },
    "APP2": {
        "scheme": "Ayushman Bharat PM-JAY",
        "status": "Approved",
        "last_updated": "2025-01-20",
        "remarks": "Ayushman Card is generated and active.",
    },
    "APP3": {
        "scheme": "PM Awas Yojana",
        "status": "Sanctioned",
        "last_updated": "2025-02-01",
        "remarks": "First instalment released via DBT.",
    },
    "APP4": {
        "scheme": "Atal Pension Yojana",
        "status": "Active",
        "last_updated": "2024-11-10",
        "remarks": "Auto-debit mandate active on savings bank account.",
    },
    "APP5": {
        "scheme": "MGNREGA",
        "status": "Job Card Issued",
        "last_updated": "2025-02-18",
        "remarks": "Job Card active at Gram Panchayat office.",
    },
}


async def execute_get_required_documents(raw_args: str) -> dict[str, Any]:
    """Phase 10 — direct document lookup (Spec §35)."""
    try:
        parsed = json.loads(raw_args)
        args = GetDocumentsArgs.model_validate(parsed)
    except Exception as exc:
        return {"success": False, "error": "invalid_arguments", "detail": str(exc)}

    doc_data = retriever.get_scheme_documents(args.scheme_id)
    if not doc_data:
        return {"success": False, "error": "scheme_not_found", "scheme_id": args.scheme_id}

    return {
        "success": True,
        "scheme_id": args.scheme_id,
        "scheme": doc_data["scheme"],
        "documents": doc_data["documents"],
        "source": doc_data["source"],
        "source_url": doc_data["source_url"],
        "last_verified": doc_data["last_verified"],
    }


async def execute_get_application_steps(raw_args: str) -> dict[str, Any]:
    """Phase 10 — direct application steps lookup (Spec §36)."""
    try:
        parsed = json.loads(raw_args)
        args = GetApplicationStepsArgs.model_validate(parsed)
    except Exception as exc:
        return {"success": False, "error": "invalid_arguments", "detail": str(exc)}

    step_data = retriever.get_scheme_application_steps(args.scheme_id)
    if not step_data:
        return {"success": False, "error": "scheme_not_found", "scheme_id": args.scheme_id}

    return {
        "success": True,
        "scheme_id": args.scheme_id,
        "scheme": step_data["scheme"],
        "steps": step_data["steps"],
        "official_source": step_data["official_source"],
        "source": step_data["source"],
        "last_verified": step_data["last_verified"],
    }


async def execute_get_application_status(raw_args: str) -> dict[str, Any]:
    """
    Phase 10 — mock demo application status lookup (Spec §37, §89).
    Includes explicit warning that this is demo mock data to prevent hallucination.
    """
    try:
        parsed = json.loads(raw_args)
        args = GetApplicationStatusArgs.model_validate(parsed)
    except Exception as exc:
        return {"success": False, "error": "invalid_arguments", "detail": str(exc)}

    app_id = args.application_id.strip().upper()
    app_info = _MOCK_APPLICATIONS.get(app_id)

    if not app_info:
        return {
            "success": True,
            "found": False,
            "application_id": app_id,
            "message": f"No application record found for ID '{app_id}'.",
            "warning": "Demo application lookup: In production, this connects to official state/central API portals.",
        }

    return {
        "success": True,
        "found": True,
        "application_id": app_id,
        "scheme": app_info["scheme"],
        "status": app_info["status"],
        "last_updated": app_info["last_updated"],
        "remarks": app_info["remarks"],
        "warning": "Demo application lookup: In production, this connects to official state/central API portals.",
    }


async def execute_check_eligibility(raw_args: str) -> dict[str, Any]:
    """
    Phase 9 — Deterministic eligibility check.
    Walls off the LLM from hallucinating eligibility decisions (Spec §34, §83, §85).
    """
    try:
        parsed = json.loads(raw_args)
        args = CheckEligibilityArgs.model_validate(parsed)
    except Exception as exc:
        logger.warning("[Tool] check_eligibility bad args %r: %s", raw_args[:200], exc)
        return {"success": False, "error": "invalid_arguments", "detail": str(exc)}

    result = check_scheme_eligibility(args.scheme_id, args.provided_slots)
    return {
        "success": True,
        "status": result.status.value,
        "scheme_id": result.scheme_id,
        "scheme_name": result.scheme_name,
        "message": result.message,
        "missing_fields": result.missing_fields,
        "reason": result.reason,
        "instruction_for_llm": result.instruction_for_llm,
        "slots_evaluated": args.provided_slots,
    }


async def execute_get_callback_slots(raw_args: str, session_id: str = "") -> dict[str, Any]:
    """
    Phase 12 — fetch available callback slots from SQLite (Spec §38).
    """
    try:
        parsed = json.loads(raw_args) if raw_args.strip() else {}
        args = GetCallbackSlotsArgs.model_validate(parsed)
    except Exception as exc:
        logger.warning("[Tool] get_callback_slots bad args %r: %s", raw_args[:200], exc)
        return {"success": False, "error": "invalid_arguments", "detail": str(exc)}

    slots = callback_store.get_available_slots(args.date)
    return {
        "success": True,
        "date_queried": args.date,
        "available_slots": slots,
        "count": len(slots),
        "instruction_for_llm": (
            "Present available slots clearly to the user and ask which time works best for them."
            if slots
            else "Inform the user that no slots are available for this date and suggest checking another date."
        ),
    }


async def execute_book_human_callback(raw_args: str, session_id: str = "") -> dict[str, Any]:
    """
    Phase 12 — atomic & idempotent callback booking in SQLite (Spec §39–42, §88).
    Guards against hallucinated booking confirmations.
    """
    try:
        parsed = json.loads(raw_args)
        args = BookCallbackArgs.model_validate(parsed)
    except Exception as exc:
        logger.warning("[Tool] book_human_callback bad args %r: %s", raw_args[:200], exc)
        return {"success": False, "error": "invalid_arguments", "detail": str(exc)}

    # Construct idempotency key from session_id + date + time_slot
    sess = session_id or "default_session"
    idempotency_key = f"{sess}:{args.date}:{args.time_slot.strip()}"

    booking_res = callback_store.book_slot(
        date_str=args.date,
        time_slot=args.time_slot,
        idempotency_key=idempotency_key,
        language=args.language,
    )

    if booking_res["success"]:
        return {
            "success": True,
            "status": "CONFIRMED",
            "booking_reference": booking_res["booking_reference"],
            "date": booking_res["date"],
            "time_slot": booking_res["time_slot"],
            "language": booking_res["language"],
            "message": booking_res["message"],
            "instruction_for_llm": (
                f"Booking is CONFIRMED. Reference ID: {booking_res['booking_reference']}. "
                f"Confirm this reference ID, date, and time to the user clearly."
            ),
        }
    else:
        return {
            "success": False,
            "status": booking_res.get("status", "FAILED"),
            "booking_reference": None,
            "message": booking_res["message"],
            "reason": booking_res.get("reason", "Slot unavailable"),
            "instruction_for_llm": (
                f"Booking FAILED: {booking_res['message']} "
                f"Do NOT confirm a booking. Politely explain that this slot is unavailable and ask the user to pick another slot."
            ),
        }


async def execute_set_language(raw_args: str) -> dict[str, Any]:
    """
    Phase 11 — Explicit language switching (Spec §44–47).
    Validates the requested language code and instructs the LLM to transition gracefully.
    """
    try:
        parsed = json.loads(raw_args)
        args = SetLanguageArgs.model_validate(parsed)
    except Exception as exc:
        logger.warning("[Tool] set_language bad args %r: %s", raw_args[:200], exc)
        return {"success": False, "error": "invalid_arguments", "detail": str(exc)}

    return {
        "success": True,
        "language_code": args.language_code,
        "message": f"Response language successfully updated to {args.language_code}.",
        "instruction_for_llm": (
            f"Response language successfully updated to {args.language_code}. "
            f"Please acknowledge this and answer the user's overarching request in this new language."
        ),
    }


# ---------------------------------------------------------------------------
# Dispatcher — maps tool name → executor
# ---------------------------------------------------------------------------

_EXECUTORS = {
    "search_knowledge":        execute_search_knowledge,
    "check_eligibility":       execute_check_eligibility,
    "get_required_documents":  execute_get_required_documents,
    "get_application_steps":   execute_get_application_steps,
    "get_application_status":  execute_get_application_status,
    "get_callback_slots":      execute_get_callback_slots,
    "book_human_callback":     execute_book_human_callback,
    "set_language":            execute_set_language,
}


async def dispatch_tool(tool_name: str, raw_args: str, session_id: str = "") -> dict[str, Any]:
    """
    Validate tool_name is in the allowlist, then execute.
    Never allows arbitrary Python execution (spec §101).
    """
    executor = _EXECUTORS.get(tool_name)
    if executor is None:
        logger.warning("[Tool] Unknown tool requested: %r", tool_name)
        return {"success": False, "error": "unknown_tool", "tool": tool_name}

    logger.info("[Tool] Executing %s  args=%r", tool_name, raw_args[:120])
    try:
        # Pass session_id to callback tools if supported, else call with raw_args
        if tool_name in {"book_human_callback", "get_callback_slots"}:
            result = await executor(raw_args, session_id=session_id)
        else:
            result = await executor(raw_args)
        logger.info("[Tool] %s result: success=%s", tool_name, result.get("success"))
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("[Tool] %s raised: %s", tool_name, exc)
        return {"success": False, "error": "execution_error", "detail": str(exc)}
