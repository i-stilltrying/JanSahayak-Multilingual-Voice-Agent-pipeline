"""
tests/test_evaluation_harness.py

Phase 15: Unit tests for automated evaluation harness, argument matching,
and metric calculation without external API dependencies.
"""

from __future__ import annotations

import json
import os
import pytest
from unittest.mock import AsyncMock, patch

# Ensure SARVAM_API_KEY is set for test environment
os.environ.setdefault("SARVAM_API_KEY", "dummy_test_key")

from evaluation.metrics import MetricAccumulator, TurnEvaluationResult, ScenarioEvaluationResult
from evaluation.runner import _compare_args, run_scenario, run_turn_eval
from backend.agent.manager import ConversationManager
from backend.agent.state import ConversationState, WorkflowState
from backend.sarvam.llm import ToolCallRequest


class TestEvaluationArgumentMatcher:
    def test_compare_args_exact_and_subset(self):
        expected = {"scheme": "pm_kisan"}
        actual = {"scheme_id": "pm_kisan", "query": "kisan money"}
        assert _compare_args(expected, actual) is True

    def test_compare_args_nested_dict(self):
        expected = {"provided_slots": {"age": 25, "is_rural_household": True}}
        actual = {
            "scheme_id": "mgnrega",
            "provided_slots": {"age": 25, "is_rural_household": True, "extra": "val"},
        }
        assert _compare_args(expected, actual) is True

    def test_compare_args_mismatch(self):
        expected = {"scheme_id": "pm_kisan"}
        actual = {"scheme_id": "ayushman_bharat"}
        assert _compare_args(expected, actual) is False

    def test_compare_args_empty_expected(self):
        assert _compare_args({}, {"any": "key"}) is True
        assert _compare_args(None, {"any": "key"}) is True


class TestEvaluationMetricsAccumulator:
    def test_metrics_calculation(self):
        acc = MetricAccumulator(
            total_scenarios=2,
            passed_scenarios=2,
            total_turns=3,
            tool_selection_correct=3,
            tool_turns_total=2,
            tool_args_correct=2,
            workflow_correct=3,
            language_correct=3,
            followup_turns_total=1,
            followup_turns_correct=1,
        )
        summary = acc.to_summary([])

        assert summary.total_cases == 2
        assert summary.passed_cases == 2
        assert summary.tool_selection_accuracy == 1.0
        assert summary.tool_argument_accuracy == 1.0
        assert summary.workflow_completion_rate == 1.0
        assert summary.language_adherence == 1.0
        assert summary.context_followup_accuracy == 1.0


class TestEvaluationRunnerExecution:
    @pytest.mark.asyncio
    async def test_run_scenario_mocked(self):
        scenario_data = {
            "scenario_id": "TEST_SCEN_01",
            "name": "Mocked test scenario",
            "turns": [
                {
                    "turn": 1,
                    "user_input": "Tell me about PM Kisan",
                    "input_language": "en-IN",
                    "expected_tool": "search_knowledge",
                    "expected_tool_args": {"scheme": "pm_kisan"},
                    "expected_workflow": "SCHEME_DISCOVERY",
                    "expected_keywords": ["₹6,000"],
                }
            ],
        }

        mock_tool_request = ToolCallRequest(
            tool_call_id="call_eval_1",
            tool_name="search_knowledge",
            raw_arguments=json.dumps({"query": "PM Kisan", "scheme": "pm_kisan"}),
        )

        async def mock_stream(messages):
            yield "PM-KISAN provides ₹6,000 annually to eligible farmers."

        with patch("backend.sarvam.llm.SarvamLLM.orchestrate", new_callable=AsyncMock) as mock_orch, \
             patch("backend.sarvam.llm.SarvamLLM.stream_after_tool", side_effect=mock_stream):
            mock_orch.return_value = mock_tool_request

            result = await run_scenario(scenario_data)

            assert result.scenario_id == "TEST_SCEN_01"
            assert result.success is True
            assert len(result.turn_results) == 1
            assert result.turn_results[0].tool_matched is True
            assert result.turn_results[0].args_matched is True
            assert result.turn_results[0].workflow_matched is True
            assert result.turn_results[0].keywords_matched is True
