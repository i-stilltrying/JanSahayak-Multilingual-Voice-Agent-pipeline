"""
evaluation/metrics.py

Phase 15: Evaluation metrics models matching Spec §165.

Requirements:
  - Track scenario-level and aggregate evaluation metrics.
  - Compute tool selection accuracy, tool argument accuracy, workflow completion rate,
    language adherence, and context follow-up accuracy.
  - Support exporting structured results to JSON without placeholder hallucinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Turn & Scenario Level Evaluation Records
# ---------------------------------------------------------------------------

class TurnEvaluationResult(BaseModel):
    turn: int
    user_input: str
    expected_tool: str | None
    actual_tool: str | None
    tool_matched: bool
    expected_args: dict[str, Any] | None = None
    actual_args: dict[str, Any] | None = None
    args_matched: bool
    expected_workflow: str
    actual_workflow: str
    workflow_matched: bool
    expected_language: str
    actual_language: str
    language_matched: bool
    response_text: str
    keywords_matched: bool = True
    latency_ms: float = 0.0


class ScenarioEvaluationResult(BaseModel):
    scenario_id: str
    name: str
    success: bool
    turn_results: list[TurnEvaluationResult] = Field(default_factory=list)
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Aggregate Evaluation Metrics (Spec §165)
# ---------------------------------------------------------------------------

class EvaluationSummary(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    tool_selection_accuracy: float
    tool_argument_accuracy: float
    workflow_completion_rate: float
    language_adherence: float
    context_followup_accuracy: float
    # Real-time voice metrics (measured during live pipeline testing)
    barge_in_success_rate: float | None = None
    median_ttfa_ms: float | None = None
    p95_ttfa_ms: float | None = None

    scenario_details: list[ScenarioEvaluationResult] = Field(default_factory=list)


@dataclass
class MetricAccumulator:
    """Helper to accumulate turn metrics across all scenarios."""

    total_scenarios: int = 0
    passed_scenarios: int = 0
    total_turns: int = 0
    tool_selection_correct: int = 0
    tool_args_correct: int = 0
    tool_turns_total: int = 0
    workflow_correct: int = 0
    language_correct: int = 0
    followup_turns_total: int = 0
    followup_turns_correct: int = 0

    def to_summary(self, scenario_results: list[ScenarioEvaluationResult]) -> EvaluationSummary:
        tool_sel_acc = (
            self.tool_selection_correct / self.total_turns
            if self.total_turns > 0
            else 1.0
        )
        tool_arg_acc = (
            self.tool_args_correct / self.tool_turns_total
            if self.tool_turns_total > 0
            else 1.0
        )
        wf_comp_rate = (
            self.workflow_correct / self.total_turns
            if self.total_turns > 0
            else 1.0
        )
        lang_adh = (
            self.language_correct / self.total_turns
            if self.total_turns > 0
            else 1.0
        )
        ctx_followup = (
            self.followup_turns_correct / self.followup_turns_total
            if self.followup_turns_total > 0
            else 1.0
        )

        return EvaluationSummary(
            total_cases=self.total_scenarios,
            passed_cases=self.passed_scenarios,
            failed_cases=self.total_scenarios - self.passed_scenarios,
            tool_selection_accuracy=round(tool_sel_acc, 4),
            tool_argument_accuracy=round(tool_arg_acc, 4),
            workflow_completion_rate=round(wf_comp_rate, 4),
            language_adherence=round(lang_adh, 4),
            context_followup_accuracy=round(ctx_followup, 4),
            barge_in_success_rate=None,
            median_ttfa_ms=None,
            p95_ttfa_ms=None,
            scenario_details=scenario_results,
        )
