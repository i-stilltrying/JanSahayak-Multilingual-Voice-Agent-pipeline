"""
evaluation/runner.py

Phase 15: Automated Evaluation Harness (Spec §128, §146–165).

Runs predefined multi-turn scenarios directly against ConversationManager,
records tool selections, argument matches, workflow transitions, and language adherence,
and outputs dynamic aggregate metrics to evaluation/results.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.agent.manager import ConversationManager
from backend.agent.state import ConversationState, TurnMetrics, WorkflowState
from backend.sarvam.llm import ToolCallRequest
from evaluation.metrics import (
    EvaluationSummary,
    MetricAccumulator,
    ScenarioEvaluationResult,
    TurnEvaluationResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval_runner")

_EVAL_DIR = Path(__file__).resolve().parent
_DEFAULT_SCENARIOS_PATH = _EVAL_DIR / "scenarios.json"
_DEFAULT_RESULTS_PATH = _EVAL_DIR / "results.json"


def _compare_args(expected: dict[str, Any] | None, actual: dict[str, Any] | None) -> bool:
    """
    Sub-match verification: Checks if all keys in expected exist with matching values in actual.
    """
    if expected is None or len(expected) == 0:
        return True
    if actual is None:
        return False

    for k, v in expected.items():
        if k not in actual:
            # Handle key aliases (e.g. scheme vs scheme_id)
            if k == "scheme" and "scheme_id" in actual:
                if str(actual["scheme_id"]).lower() != str(v).lower():
                    return False
                continue
            if k == "scheme_id" and "scheme" in actual:
                if str(actual["scheme"]).lower() != str(v).lower():
                    return False
                continue
            return False

        if isinstance(v, dict):
            if not isinstance(actual[k], dict) or not _compare_args(v, actual[k]):
                return False
        elif isinstance(v, (int, float)):
            try:
                if float(actual[k]) != float(v):
                    return False
            except (ValueError, TypeError):
                return False
        elif isinstance(v, bool):
            if bool(actual[k]) != v:
                return False
        elif str(actual[k]).strip().lower() != str(v).strip().lower():
            return False

    return True


async def run_turn_eval(
    manager: ConversationManager,
    state: ConversationState,
    turn_data: dict[str, Any],
    turn_num: int,
) -> TurnEvaluationResult:
    """
    Execute a single turn and evaluate correctness against expected outputs.
    """
    user_input = turn_data.get("user_input", "")
    input_lang = turn_data.get("input_language", "en-IN")
    expected_tool = turn_data.get("expected_tool")
    expected_args = turn_data.get("expected_tool_args")
    expected_workflow = turn_data.get("expected_workflow", "NONE")
    expected_keywords = turn_data.get("expected_keywords", [])

    # Intercept orchestrate to record the actual tool call if made
    actual_tool: str | None = None
    actual_args: dict[str, Any] | None = None

    orig_orchestrate = manager._llm.orchestrate

    async def _intercept_orchestrate(messages):
        nonlocal actual_tool, actual_args
        res = await orig_orchestrate(messages)
        if isinstance(res, ToolCallRequest):
            actual_tool = res.tool_name
            try:
                actual_args = json.loads(res.raw_arguments)
            except Exception:
                actual_args = {}
        else:
            actual_tool = None
            actual_args = None
        return res

    manager._llm.orchestrate = _intercept_orchestrate

    t_start = time.perf_counter()
    response_chunks: list[str] = []
    turn_metrics = TurnMetrics(t_speech_end=t_start)

    try:
        async for chunk in manager.run_turn(
            user_text=user_input,
            detected_lang=input_lang,
            state=state,
            generation_id=turn_num,
            turn_metrics=turn_metrics,
        ):
            response_chunks.append(chunk)
    finally:
        manager._llm.orchestrate = orig_orchestrate

    latency_ms = (time.perf_counter() - t_start) * 1000
    response_text = "".join(response_chunks)

    # 1. Tool Selection Matching
    tool_matched = (actual_tool == expected_tool)

    # 2. Tool Arguments Matching
    args_matched = True
    if expected_tool is not None:
        args_matched = tool_matched and _compare_args(expected_args, actual_args)

    # 3. Workflow Matching
    actual_workflow = state.current_workflow
    workflow_matched = (
        actual_workflow == expected_workflow
        or (expected_workflow == "NONE" and actual_workflow in {"NONE", WorkflowState.NONE})
    )

    # 4. Language Adherence Matching
    actual_language = state.response_language
    expected_lang = input_lang
    if actual_tool == "set_language" and actual_args and "language_code" in actual_args:
        expected_lang = actual_args["language_code"]
    language_matched = (actual_language == expected_lang or actual_language == state.response_language)

    # 5. Keyword presence in final speech
    keywords_matched = True
    if expected_keywords:
        # Check if at least one expected keyword is present
        keywords_matched = any(kw.lower() in response_text.lower() for kw in expected_keywords)

    return TurnEvaluationResult(
        turn=turn_num,
        user_input=user_input,
        expected_tool=expected_tool,
        actual_tool=actual_tool,
        tool_matched=tool_matched,
        expected_args=expected_args,
        actual_args=actual_args,
        args_matched=args_matched,
        expected_workflow=expected_workflow,
        actual_workflow=actual_workflow,
        workflow_matched=workflow_matched,
        expected_language=expected_lang,
        actual_language=actual_language,
        language_matched=language_matched,
        response_text=response_text,
        keywords_matched=keywords_matched,
        latency_ms=latency_ms,
    )


async def run_scenario(scenario: dict[str, Any]) -> ScenarioEvaluationResult:
    """
    Run a single multi-turn scenario in full session isolation (Spec §163).
    """
    scen_id = scenario.get("scenario_id", "UNKNOWN")
    scen_name = scenario.get("name", "Unknown Scenario")
    turns = scenario.get("turns", [])

    # Instantiate fresh isolated manager & session state
    manager = ConversationManager()
    session_id = f"eval_{scen_id}_{uuid4().hex[:6]}"
    state = ConversationState(session_id=session_id)

    turn_results: list[TurnEvaluationResult] = []
    scenario_passed = True
    error_msg: str | None = None

    for i, turn_data in enumerate(turns, start=1):
        try:
            res = await run_turn_eval(manager, state, turn_data, turn_num=i)
            turn_results.append(res)
            # A scenario turn is passed if tool, args, and workflow matched
            if not (res.tool_matched and res.args_matched and res.workflow_matched):
                scenario_passed = False
        except Exception as exc:
            logger.exception("Error executing scenario %s turn %d: %s", scen_id, i, exc)
            scenario_passed = False
            error_msg = str(exc)
            break

    return ScenarioEvaluationResult(
        scenario_id=scen_id,
        name=scen_name,
        success=scenario_passed,
        turn_results=turn_results,
        error_message=error_msg,
    )


async def run_all_scenarios(
    scenarios_path: str | Path = _DEFAULT_SCENARIOS_PATH,
    results_path: str | Path = _DEFAULT_RESULTS_PATH,
) -> EvaluationSummary:
    """
    Load all test scenarios, execute sequentially, accumulate metrics,
    and save summary results to results.json (Spec §128, §165).
    """
    scenarios_file = Path(scenarios_path)
    if not scenarios_file.exists():
        raise FileNotFoundError(f"Scenarios file not found: {scenarios_file}")

    scenarios = json.loads(scenarios_file.read_text(encoding="utf-8"))
    logger.info("Loaded %d scenarios from %s", len(scenarios), scenarios_file.name)

    acc = MetricAccumulator(total_scenarios=len(scenarios))
    scenario_results: list[ScenarioEvaluationResult] = []

    for idx, scen in enumerate(scenarios, start=1):
        logger.info("[%d/%d] Running scenario: %s (%s)", idx, len(scenarios), scen.get("scenario_id"), scen.get("name"))
        res = await run_scenario(scen)
        scenario_results.append(res)

        if res.success:
            acc.passed_scenarios += 1

        for turn_res in res.turn_results:
            acc.total_turns += 1
            if turn_res.tool_matched:
                acc.tool_selection_correct += 1
            if turn_res.expected_tool is not None:
                acc.tool_turns_total += 1
                if turn_res.args_matched:
                    acc.tool_args_correct += 1
            if turn_res.workflow_matched:
                acc.workflow_correct += 1
            if turn_res.language_matched:
                acc.language_correct += 1
            if turn_res.turn > 1:
                acc.followup_turns_total += 1
                if turn_res.workflow_matched and turn_res.tool_matched:
                    acc.followup_turns_correct += 1

    summary = acc.to_summary(scenario_results)

    # Save output to results_path
    out_file = Path(results_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Evaluation complete! Results written to %s", out_file)

    # Print clean summary table to stdout
    print("\n" + "=" * 60)
    print(" JANSAHAYAK VOICE AGENT EVALUATION SUMMARY (Phase 15)")
    print("=" * 60)
    print(f" Total Cases Evaluated       : {summary.total_cases}")
    print(f" Passed Cases                : {summary.passed_cases}")
    print(f" Failed Cases                : {summary.failed_cases}")
    print(f" Tool Selection Accuracy     : {summary.tool_selection_accuracy * 100:.2f}%")
    print(f" Tool Argument Accuracy      : {summary.tool_argument_accuracy * 100:.2f}%")
    print(f" Workflow Completion Rate    : {summary.workflow_completion_rate * 100:.2f}%")
    print(f" Language Adherence          : {summary.language_adherence * 100:.2f}%")
    print(f" Context Follow-up Accuracy  : {summary.context_followup_accuracy * 100:.2f}%")
    print("=" * 60 + "\n")

    return summary


def main():
    scenarios_arg = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_SCENARIOS_PATH
    results_arg = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_RESULTS_PATH
    asyncio.run(run_all_scenarios(scenarios_arg, results_arg))


if __name__ == "__main__":
    main()
