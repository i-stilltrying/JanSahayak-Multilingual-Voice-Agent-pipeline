"""
backend/agent/eligibility.py

Phase 9: Deterministic Eligibility Engine.

Critical Architectural Constraint (Spec §34, §83, §85):
  - The LLM must NEVER be the final eligibility decision-maker.
  - The LLM's only job is to extract entities (slots) and pass them to Python.
  - Python evaluates deterministic business rules and returns an exact result.
  - The LLM then translates that result into natural, empathetic speech.

Return states:
  - ELIGIBLE: Confirmation message and benefit details.
  - INELIGIBLE: Specific deterministic reason for ineligibility.
  - MISSING_DATA: List of exact missing fields required from user, plus guidance instruction.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Status Enum and Result Model
# ---------------------------------------------------------------------------

class EligibilityStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    MISSING_DATA = "MISSING_DATA"


class EligibilityResult(BaseModel):
    status: EligibilityStatus
    scheme_id: str
    scheme_name: str
    message: str
    missing_fields: list[str] = Field(default_factory=list)
    reason: str | None = None
    instruction_for_llm: str | None = None


# ---------------------------------------------------------------------------
# Scheme Slots Models (Pydantic)
# ---------------------------------------------------------------------------

class PMKisanSlots(BaseModel):
    land_ownership_hectares: float | None = None
    is_institutional_landholder: bool | None = None
    pays_income_tax: bool | None = None
    is_government_employee: bool | None = None


class AyushmanBharatSlots(BaseModel):
    secc_deprivation_listed: bool | None = None
    has_existing_government_health_insurance: bool | None = None


class PMAwasSlots(BaseModel):
    owns_pucca_house: bool | None = None
    annual_income_inr: float | None = None


class AtalPensionSlots(BaseModel):
    age: int | None = None
    has_savings_bank_account: bool | None = None
    is_income_tax_payer: bool | None = None


class MGNREGASlots(BaseModel):
    is_rural_household: bool | None = None
    age: int | None = None
    willing_for_manual_work: bool | None = None


# ---------------------------------------------------------------------------
# Helper type coercion utilities for flexible slot inputs
# ---------------------------------------------------------------------------

def _to_bool(val: Any) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"true", "yes", "y", "1", "हाँ", "हा", "ಹೌದು"}:
            return True
        if v in {"false", "no", "n", "0", "नहीं", "ना", "ಇಲ್ಲ"}:
            return False
    return None


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            cleaned = val.replace(",", "").strip()
            return float(cleaned)
        except ValueError:
            return None
    return None


def _to_int(val: Any) -> int | None:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        try:
            cleaned = val.replace(",", "").strip()
            return int(float(cleaned))
        except ValueError:
            return None
    return None


def _get_first(slots: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in slots and slots[k] is not None:
            return slots[k]
    return None


# ---------------------------------------------------------------------------
# Scheme Rule Evaluators
# ---------------------------------------------------------------------------

def evaluate_pm_kisan(slots: dict[str, Any]) -> EligibilityResult:
    """
    Deterministic rule engine for PM-KISAN (Pradhan Mantri Kisan Samman Nidhi).

    Required slots:
      - land_ownership_hectares (float > 0)
      - is_institutional_landholder (bool, must be False)
      - pays_income_tax (bool, must be False)

    Optional / Exclusion check:
      - is_government_employee (bool, must be False if provided)
    """
    scheme_id = "pm_kisan"
    scheme_name = "PM-KISAN"

    raw_land = _get_first(slots, "land_ownership_hectares", "land_ownership", "land_hectares")
    raw_inst = _get_first(slots, "is_institutional_landholder", "institutional_landholder")
    raw_tax = _get_first(slots, "pays_income_tax", "income_tax_payer", "pays_tax")
    raw_govt = _get_first(slots, "is_government_employee", "government_employee")

    land = _to_float(raw_land)
    inst = _to_bool(raw_inst)
    tax = _to_bool(raw_tax)
    govt = _to_bool(raw_govt)

    # Check for missing required slots
    missing = []
    if land is None:
        missing.append("land_ownership_hectares")
    if inst is None:
        missing.append("is_institutional_landholder")
    if tax is None:
        missing.append("pays_income_tax")

    if missing:
        missing_str = ", ".join(missing)
        return EligibilityResult(
            status=EligibilityStatus.MISSING_DATA,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="Additional information is required to verify PM-KISAN eligibility.",
            missing_fields=missing,
            instruction_for_llm=f"Ask the user for the following missing information: {missing_str}",
        )

    # Deterministic Ineligibility Checks
    if land <= 0:
        return EligibilityResult(
            status=EligibilityStatus.INELIGIBLE,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="You are not eligible for PM-KISAN.",
            reason="PM-KISAN is exclusively for farmer families who own cultivable agricultural land.",
        )

    if inst is True:
        return EligibilityResult(
            status=EligibilityStatus.INELIGIBLE,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="You are not eligible for PM-KISAN.",
            reason="Institutional landholders are excluded from PM-KISAN benefits.",
        )

    if tax is True:
        return EligibilityResult(
            status=EligibilityStatus.INELIGIBLE,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="You are not eligible for PM-KISAN.",
            reason="Individuals who pay income tax are excluded from PM-KISAN.",
        )

    if govt is True:
        return EligibilityResult(
            status=EligibilityStatus.INELIGIBLE,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="You are not eligible for PM-KISAN.",
            reason="Serving or retired government employees (excluding Class IV/Group D) are excluded from PM-KISAN.",
        )

    # Eligible
    return EligibilityResult(
        status=EligibilityStatus.ELIGIBLE,
        scheme_id=scheme_id,
        scheme_name=scheme_name,
        message="You are eligible for PM-KISAN! You are entitled to ₹6,000 per year paid in three equal instalments of ₹2,000 directly into your bank account via Direct Benefit Transfer.",
    )


def evaluate_ayushman_bharat(slots: dict[str, Any]) -> EligibilityResult:
    """
    Deterministic rule engine for Ayushman Bharat (PM-JAY).

    Required slots:
      - secc_deprivation_listed (bool, must be True)
    """
    scheme_id = "ayushman_bharat"
    scheme_name = "Ayushman Bharat PM-JAY"

    raw_secc = _get_first(slots, "secc_deprivation_listed", "secc_listed", "is_secc_listed")
    secc = _to_bool(raw_secc)
    missing = []
    if secc is None:
        missing.append("secc_deprivation_listed")

    if missing:
        missing_str = ", ".join(missing)
        return EligibilityResult(
            status=EligibilityStatus.MISSING_DATA,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="Additional information is required to verify Ayushman Bharat eligibility.",
            missing_fields=missing,
            instruction_for_llm=f"Ask the user for the following missing information: {missing_str}",
        )

    if secc is False:
        return EligibilityResult(
            status=EligibilityStatus.INELIGIBLE,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="You are not eligible for Ayushman Bharat PM-JAY under the SECC criteria.",
            reason="Ayushman Bharat PM-JAY eligibility requires the beneficiary family to be listed in the Socio-Economic Caste Census (SECC 2011) deprivation database or relevant state target categories.",
        )

    return EligibilityResult(
        status=EligibilityStatus.ELIGIBLE,
        scheme_id=scheme_id,
        scheme_name=scheme_name,
        message="You are eligible for Ayushman Bharat PM-JAY! Your family is entitled to free, cashless hospitalisation cover of up to ₹5 lakh per year at empanelled government and private hospitals across India.",
    )


def evaluate_pm_awas(slots: dict[str, Any]) -> EligibilityResult:
    """
    Deterministic rule engine for PM Awas Yojana (PMAY).

    Required slots:
      - owns_pucca_house (bool, must be False)
      - annual_income_inr (float, max ₹18,00,000 for CLSS/PMAY)
    """
    scheme_id = "pm_awas"
    scheme_name = "PM Awas Yojana"

    raw_house = _get_first(slots, "owns_pucca_house", "house_ownership")
    raw_income = _get_first(slots, "annual_income_inr", "annual_income", "income")

    owns_house = _to_bool(raw_house)
    income = _to_float(raw_income)

    missing = []
    if owns_house is None:
        missing.append("owns_pucca_house")
    if income is None:
        missing.append("annual_income_inr")

    if missing:
        missing_str = ", ".join(missing)
        return EligibilityResult(
            status=EligibilityStatus.MISSING_DATA,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="Additional information is required to verify PM Awas Yojana eligibility.",
            missing_fields=missing,
            instruction_for_llm=f"Ask the user for the following missing information: {missing_str}",
        )

    if owns_house is True:
        return EligibilityResult(
            status=EligibilityStatus.INELIGIBLE,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="You are not eligible for PM Awas Yojana.",
            reason="Applicants or family members who already own a pucca house anywhere in India are not eligible for PMAY housing assistance.",
        )

    if income > 1800000:
        return EligibilityResult(
            status=EligibilityStatus.INELIGIBLE,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="You are not eligible for PM Awas Yojana subsidy.",
            reason="Annual household income exceeds the maximum threshold of ₹18,00,000 per year for PMAY interest subsidies.",
        )

    return EligibilityResult(
        status=EligibilityStatus.ELIGIBLE,
        scheme_id=scheme_id,
        scheme_name=scheme_name,
        message="You are eligible for PM Awas Yojana! You qualify for financial assistance or interest subsidy on housing loans based on your income group.",
    )


def evaluate_atal_pension(slots: dict[str, Any]) -> EligibilityResult:
    """
    Deterministic rule engine for Atal Pension Yojana (APY).

    Required slots:
      - age (int, between 18 and 40)
      - has_savings_bank_account (bool, must be True)
    """
    scheme_id = "atal_pension"
    scheme_name = "Atal Pension Yojana"

    raw_age = _get_first(slots, "age")
    raw_bank = _get_first(slots, "has_savings_bank_account", "bank_account", "has_bank_account")

    age = _to_int(raw_age)
    bank = _to_bool(raw_bank)

    missing = []
    if age is None:
        missing.append("age")
    if bank is None:
        missing.append("has_savings_bank_account")

    if missing:
        missing_str = ", ".join(missing)
        return EligibilityResult(
            status=EligibilityStatus.MISSING_DATA,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="Additional information is required to verify Atal Pension Yojana eligibility.",
            missing_fields=missing,
            instruction_for_llm=f"Ask the user for the following missing information: {missing_str}",
        )

    if age < 18 or age > 40:
        return EligibilityResult(
            status=EligibilityStatus.INELIGIBLE,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="You are not eligible to join Atal Pension Yojana.",
            reason=f"Atal Pension Yojana requires subscribers to be between 18 and 40 years of age (current age provided: {age}).",
        )

    if bank is False:
        return EligibilityResult(
            status=EligibilityStatus.INELIGIBLE,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="You cannot enrol in Atal Pension Yojana without a savings bank account.",
            reason="A savings bank account is mandatory for setting up monthly auto-debit pension contributions.",
        )

    return EligibilityResult(
        status=EligibilityStatus.ELIGIBLE,
        scheme_id=scheme_id,
        scheme_name=scheme_name,
        message="You are eligible for Atal Pension Yojana! You can choose a guaranteed monthly pension between ₹1,000 and ₹5,000 starting from age 60.",
    )


def evaluate_mgnrega(slots: dict[str, Any]) -> EligibilityResult:
    """
    Deterministic rule engine for MGNREGA.

    Required slots:
      - is_rural_household (bool, must be True)
      - age (int, must be >= 18)
      - willing_for_manual_work (bool, must be True)
    """
    scheme_id = "mgnrega"
    scheme_name = "MGNREGA"

    raw_rural = _get_first(slots, "is_rural_household", "rural_household")
    if raw_rural is None and "residence_type" in slots:
        raw_rural = slots["residence_type"] == "rural"

    raw_age = _get_first(slots, "age")
    raw_work = _get_first(slots, "willing_for_manual_work", "willingness_for_manual_work", "manual_work")

    rural = _to_bool(raw_rural)
    age = _to_int(raw_age)
    work = _to_bool(raw_work)

    missing = []
    if rural is None:
        missing.append("is_rural_household")
    if age is None:
        missing.append("age")
    if work is None:
        missing.append("willing_for_manual_work")

    if missing:
        missing_str = ", ".join(missing)
        return EligibilityResult(
            status=EligibilityStatus.MISSING_DATA,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="Additional information is required to verify MGNREGA eligibility.",
            missing_fields=missing,
            instruction_for_llm=f"Ask the user for the following missing information: {missing_str}",
        )

    if rural is False:
        return EligibilityResult(
            status=EligibilityStatus.INELIGIBLE,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="You are not eligible for MGNREGA.",
            reason="MGNREGA is strictly for rural households. Urban households are not eligible.",
        )

    if age < 18:
        return EligibilityResult(
            status=EligibilityStatus.INELIGIBLE,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="You are not eligible for MGNREGA.",
            reason=f"MGNREGA requires the applicant to be an adult of at least 18 years of age (current age provided: {age}).",
        )

    if work is False:
        return EligibilityResult(
            status=EligibilityStatus.INELIGIBLE,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            message="You are not eligible for MGNREGA.",
            reason="MGNREGA provides guaranteed wage employment for unskilled manual labour. Willingness to do manual work is mandatory.",
        )

    return EligibilityResult(
        status=EligibilityStatus.ELIGIBLE,
        scheme_id=scheme_id,
        scheme_name=scheme_name,
        message="You are eligible for MGNREGA! Your household is entitled to at least 100 days of guaranteed wage employment per financial year.",
    )


# ---------------------------------------------------------------------------
# Scheme Registry & Main Entry Point
# ---------------------------------------------------------------------------

_EVALUATORS = {
    "pm_kisan": evaluate_pm_kisan,
    "ayushman_bharat": evaluate_ayushman_bharat,
    "pm_awas": evaluate_pm_awas,
    "atal_pension": evaluate_atal_pension,
    "mgnrega": evaluate_mgnrega,
}


def check_scheme_eligibility(scheme_id: str, slots: dict[str, Any]) -> EligibilityResult:
    """
    Evaluate deterministic eligibility for a scheme given currently known slots.
    """
    normalized_id = scheme_id.strip().lower().replace("-", "_")
    evaluator = _EVALUATORS.get(normalized_id)
    if not evaluator:
        return EligibilityResult(
            status=EligibilityStatus.INELIGIBLE,
            scheme_id=scheme_id,
            scheme_name=scheme_id,
            message=f"Unknown scheme: {scheme_id}",
            reason=f"Scheme '{scheme_id}' is not supported by the eligibility engine.",
        )

    return evaluator(slots)
