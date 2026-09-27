"""
tests/test_metrics.py

Phase 14: Tests for TurnMetrics latency calculations, breakdown logic,
and WebSocket protocol serialization (Spec §130–132, §144).
"""

from __future__ import annotations

import json
import os
import pytest

# Ensure SARVAM_API_KEY is set for test environment
os.environ.setdefault("SARVAM_API_KEY", "dummy_test_key")

from backend.agent.state import TurnMetrics
from backend.websocket.protocol import LatencyMetricsMessage, MessageType


class TestTurnMetricsCalculations:
    def test_complete_turn_with_tool(self):
        """
        Test a full turn where STT, Tool, LLM, and TTS all occur.
        """
        metrics = TurnMetrics(
            t_speech_end=10.0,
            t_stt_final=10.25,        # STT = 250 ms
            t_llm_start=10.26,
            t_tool_start=10.40,
            t_tool_end=10.50,         # Tool = 100 ms
            t_llm_first_token=10.66,   # Total LLM raw = 400 ms -> LLM net = 300 ms
            t_tts_start=10.70,
            t_tts_first_audio=10.85,   # TTS = 150 ms (Total E2E = 850 ms from speech_end)
        )

        breakdown = metrics.calculate_breakdown()

        assert pytest.approx(breakdown["stt_ms"], 0.1) == 250.0
        assert pytest.approx(breakdown["tool_ms"], 0.1) == 100.0
        assert pytest.approx(breakdown["llm_ms"], 0.1) == 300.0
        assert pytest.approx(breakdown["tts_ms"], 0.1) == 150.0
        assert pytest.approx(breakdown["e2e_ms"], 0.1) == 850.0

    def test_direct_answer_non_tool_turn(self):
        """
        Test a conversational turn with no tool execution.
        """
        metrics = TurnMetrics(
            t_speech_end=1.0,
            t_stt_final=1.20,        # STT = 200 ms
            t_llm_start=1.21,
            t_llm_first_token=1.41,  # LLM = 200 ms
            t_tts_start=1.45,
            t_tts_first_audio=1.60,  # TTS = 150 ms (Total E2E = 600 ms)
        )

        breakdown = metrics.calculate_breakdown()

        assert "tool_ms" not in breakdown
        assert pytest.approx(breakdown["stt_ms"], 0.1) == 200.0
        assert pytest.approx(breakdown["llm_ms"], 0.1) == 200.0
        assert pytest.approx(breakdown["tts_ms"], 0.1) == 150.0
        assert pytest.approx(breakdown["e2e_ms"], 0.1) == 600.0

    def test_partial_metrics_missing_speech_end(self):
        """
        Test fallback when t_speech_end is not available (uses t_stt_final as origin).
        """
        metrics = TurnMetrics(
            t_stt_final=5.0,
            t_llm_start=5.05,
            t_llm_first_token=5.25,
            t_tts_start=5.30,
            t_tts_first_audio=5.50,
        )

        breakdown = metrics.calculate_breakdown()

        assert "stt_ms" not in breakdown
        assert pytest.approx(breakdown["llm_ms"], 0.1) == 200.0
        assert pytest.approx(breakdown["tts_ms"], 0.1) == 200.0
        assert pytest.approx(breakdown["e2e_ms"], 0.1) == 500.0

    def test_empty_metrics(self):
        """
        Test calculating breakdown on an uninitialized TurnMetrics object.
        """
        metrics = TurnMetrics()
        breakdown = metrics.calculate_breakdown()
        assert breakdown == {}


class TestLatencyProtocolSerialization:
    def test_latency_metrics_message_serialization(self):
        breakdown = {
            "stt_ms": 180.5,
            "llm_ms": 220.0,
            "tool_ms": 45.2,
            "tts_ms": 130.0,
            "e2e_ms": 575.7,
        }

        msg = LatencyMetricsMessage(
            session_id="sess_metrics_123",
            turn_id=3,
            generation_id=1,
            timestamp_ms=1710000000000.0,
            metrics=breakdown,
        )

        dump = msg.model_dump()
        assert dump["type"] == MessageType.LATENCY
        assert dump["session_id"] == "sess_metrics_123"
        assert dump["turn_id"] == 3
        assert dump["metrics"]["stt_ms"] == 180.5
        assert dump["metrics"]["llm_ms"] == 220.0

        # Verify JSON roundtrip
        json_str = msg.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["type"] == "latency"
        assert parsed["metrics"]["e2e_ms"] == 575.7
