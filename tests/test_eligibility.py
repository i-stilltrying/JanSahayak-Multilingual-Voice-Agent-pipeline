"""
tests/test_eligibility.py

Phase 9: Tests for deterministic eligibility rules engine and tool integration.
"""

from __future__ import annotations

import json
import os
import pytest

# Ensure SARVAM_API_KEY is set for test environment if not present
os.environ.setdefault("SARVAM_API_KEY", "dummy_test_key")

from backend.agent.eligibility import (
    EligibilityStatus,
    check_scheme_eligibility,
    evaluate_pm_kisan,
    evaluate_ayushman_bharat,
    evaluate_pm_awas,
    evaluate_atal_pension,
    evaluate_mgnrega,
)
from backend.agent.tools import dispatch_tool
from backend.agent.manager import ConversationManager
from backend.agent.state import ConversationState, WorkflowState


# ---------------------------------------------------------------------------
# PM-KISAN Unit Tests
# ---------------------------------------------------------------------------

class TestPMKisanEligibility:
    def test_missing_all_slots(self):
        result = evaluate_pm_kisan({})
        assert result.status == EligibilityStatus.MISSING_DATA
        assert "land_ownership_hectares" in result.missing_fields
        assert "is_institutional_landholder" in result.missing_fields
        assert "pays_income_tax" in result.missing_fields
        assert result.instruction_for_llm is not None
        assert "Ask the user for the following missing information:" in result.instruction_for_llm

    def test_partial_slots_missing(self):
        result = evaluate_pm_kisan({"land_ownership_hectares": 1.5})
        assert result.status == EligibilityStatus.MISSING_DATA
        assert "land_ownership_hectares" not in result.missing_fields
        assert "is_institutional_landholder" in result.missing_fields
        assert "pays_income_tax" in result.missing_fields

    def test_eligible_marginal_farmer(self):
        slots = {
            "land_ownership_hectares": 1.2,
            "is_institutional_landholder": False,
            "pays_income_tax": False,
        }
        result = evaluate_pm_kisan(slots)
        assert result.status == EligibilityStatus.ELIGIBLE
        assert "₹6,000" in result.message

    def test_eligible_string_coercion(self):
        slots = {
            "land_ownership_hectares": "2.0",
            "is_institutional_landholder": "no",
            "pays_income_tax": "false",
        }
        result = evaluate_pm_kisan(slots)
        assert result.status == EligibilityStatus.ELIGIBLE

    def test_ineligible_no_land(self):
        slots = {
            "land_ownership_hectares": 0.0,
            "is_institutional_landholder": False,
            "pays_income_tax": False,
        }
        result = evaluate_pm_kisan(slots)
        assert result.status == EligibilityStatus.INELIGIBLE
        assert "cultivable agricultural land" in (result.reason or "")

    def test_ineligible_institutional_landholder(self):
        slots = {
            "land_ownership_hectares": 5.0,
            "is_institutional_landholder": True,
            "pays_income_tax": False,
        }
        result = evaluate_pm_kisan(slots)
        assert result.status == EligibilityStatus.INELIGIBLE
        assert "Institutional landholders" in (result.reason or "")

    def test_ineligible_pays_income_tax(self):
        slots = {
            "land_ownership_hectares": 2.5,
            "is_institutional_landholder": False,
            "pays_income_tax": True,
        }
        result = evaluate_pm_kisan(slots)
        assert result.status == EligibilityStatus.INELIGIBLE
        assert "income tax" in (result.reason or "")

    def test_ineligible_government_employee(self):
        slots = {
            "land_ownership_hectares": 1.0,
            "is_institutional_landholder": False,
            "pays_income_tax": False,
            "is_government_employee": True,
        }
        result = evaluate_pm_kisan(slots)
        assert result.status == EligibilityStatus.INELIGIBLE
        assert "government employees" in (result.reason or "")


# ---------------------------------------------------------------------------
# Ayushman Bharat Unit Tests
# ---------------------------------------------------------------------------

class TestAyushmanBharatEligibility:
    def test_missing_slots(self):
        result = evaluate_ayushman_bharat({})
        assert result.status == EligibilityStatus.MISSING_DATA
        assert "secc_deprivation_listed" in result.missing_fields
        assert "Ask the user" in (result.instruction_for_llm or "")

    def test_eligible_secc_listed(self):
        result = evaluate_ayushman_bharat({"secc_deprivation_listed": True})
        assert result.status == EligibilityStatus.ELIGIBLE
        assert "₹5 lakh" in result.message

    def test_ineligible_not_secc_listed(self):
        result = evaluate_ayushman_bharat({"secc_deprivation_listed": False})
        assert result.status == EligibilityStatus.INELIGIBLE
        assert "SECC" in (result.reason or "")


# ---------------------------------------------------------------------------
# Other Schemes (PM Awas, APY, MGNREGA) Tests
# ---------------------------------------------------------------------------

class TestOtherSchemesEligibility:
    def test_pm_awas_missing_slots(self):
        result = evaluate_pm_awas({})
        assert result.status == EligibilityStatus.MISSING_DATA
        assert "owns_pucca_house" in result.missing_fields
        assert "annual_income_inr" in result.missing_fields

    def test_pm_awas_eligible(self):
        result = evaluate_pm_awas({"owns_pucca_house": False, "annual_income_inr": 250000})
        assert result.status == EligibilityStatus.ELIGIBLE

    def test_pm_awas_ineligible_owns_house(self):
        result = evaluate_pm_awas({"owns_pucca_house": True, "annual_income_inr": 250000})
        assert result.status == EligibilityStatus.INELIGIBLE
        assert "pucca house" in (result.reason or "")

    def test_atal_pension_eligible(self):
        result = evaluate_atal_pension({"age": 28, "has_savings_bank_account": True})
        assert result.status == EligibilityStatus.ELIGIBLE

    def test_atal_pension_ineligible_age(self):
        result = evaluate_atal_pension({"age": 45, "has_savings_bank_account": True})
        assert result.status == EligibilityStatus.INELIGIBLE
        assert "between 18 and 40" in (result.reason or "")

    def test_mgnrega_eligible(self):
        result = evaluate_mgnrega({
            "is_rural_household": True,
            "age": 22,
            "willing_for_manual_work": True,
        })
        assert result.status == EligibilityStatus.ELIGIBLE
        assert "100 days" in result.message

    def test_mgnrega_ineligible_urban(self):
        result = evaluate_mgnrega({
            "is_rural_household": False,
            "age": 22,
            "willing_for_manual_work": True,
        })
        assert result.status == EligibilityStatus.INELIGIBLE
        assert "rural households" in (result.reason or "")


# ---------------------------------------------------------------------------
# Tool Dispatcher & State Merging Integration Tests
# ---------------------------------------------------------------------------

class TestToolAndStateIntegration:
    @pytest.mark.asyncio
    async def test_dispatch_check_eligibility_missing(self):
        payload = json.dumps({
            "scheme_id": "pm_kisan",
            "provided_slots": {"land_ownership_hectares": 1.5},
        })
        result = await dispatch_tool("check_eligibility", payload)
        assert result["success"] is True
        assert result["status"] == "MISSING_DATA"
        assert "is_institutional_landholder" in result["missing_fields"]
        assert "instruction_for_llm" in result
        assert "Ask the user" in result["instruction_for_llm"]

    @pytest.mark.asyncio
    async def test_dispatch_check_eligibility_aliases(self):
        # Using legacy "scheme" and "user_data" aliases
        payload = json.dumps({
            "scheme": "ayushman_bharat",
            "user_data": {"secc_listed": True},
        })
        result = await dispatch_tool("check_eligibility", payload)
        assert result["success"] is True
        assert result["status"] == "ELIGIBLE"

    def test_manager_state_slot_merging(self):
        manager = ConversationManager()
        state = ConversationState(session_id="test_sess")

        # Simulate Turn 1: user provides land ownership
        raw_args_1 = json.dumps({
            "scheme_id": "pm_kisan",
            "provided_slots": {"land_ownership_hectares": 2.0},
        })
        tool_result_1 = {
            "success": True,
            "status": "MISSING_DATA",
            "missing_fields": ["is_institutional_landholder", "pays_income_tax"],
        }
        manager._update_state_from_tool("check_eligibility", raw_args_1, tool_result_1, state)

        assert state.current_scheme == "pm_kisan"
        assert state.current_workflow == WorkflowState.ELIGIBILITY
        assert state.collected_slots == {"land_ownership_hectares": 2.0}
        assert state.required_slots == ["is_institutional_landholder", "pays_income_tax"]

        # Simulate Turn 2: user provides remaining slots
        raw_args_2 = json.dumps({
            "scheme_id": "pm_kisan",
            "provided_slots": {
                "is_institutional_landholder": False,
                "pays_income_tax": False,
            },
        })
        tool_result_2 = {
            "success": True,
            "status": "ELIGIBLE",
            "missing_fields": [],
        }
        manager._update_state_from_tool("check_eligibility", raw_args_2, tool_result_2, state)

        assert state.collected_slots == {
            "land_ownership_hectares": 2.0,
            "is_institutional_landholder": False,
            "pays_income_tax": False,
        }
        assert state.required_slots == []
