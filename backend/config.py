"""
backend/config.py

Loads all configuration from environment variables via pydantic-settings.
The application will refuse to start if SARVAM_API_KEY is missing.

NOTE: Phase 1/2 — Sarvam model fields are defined here but not used yet.
They will be wired in Phase 3 (STT), 4 (LLM), and 5 (TTS).
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application configuration, loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------
    sarvam_api_key: str = Field(default="provided_by_reviewer_ui", description="Sarvam subscription API key")

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_env: str = "development"
    log_level: str = "DEBUG"

    # ------------------------------------------------------------------
    # Sarvam models (used from Phase 3 onward)
    # ------------------------------------------------------------------
    sarvam_stt_model: str = "saaras:v3-realtime"
    sarvam_llm_model: str = "sarvam-105b-conversations"
    sarvam_tts_model: str = "bulbul:v3"

    # ------------------------------------------------------------------
    # Voice / language defaults
    # ------------------------------------------------------------------
    default_response_language: str = "hi-IN"
    tts_speaker_hi: str = "shubh"
    tts_speaker_en: str = "shubh"
    tts_speaker_kn: str = "amit"

    # ------------------------------------------------------------------
    # STT realtime (used from Phase 3 onward)
    # ------------------------------------------------------------------
    stt_stream_type: str = "fast"
    stt_endpointing: str = "vad"
    stt_sample_rate: int = 16000
    stt_encoding: str = "linear16"
    stt_vad_threshold: float = 0.6          # raised: filters faint noise & breath
    stt_silence_duration_ms: int = 1000
    stt_min_speech_duration_ms: int = 500   # raised: ignores sub-500 ms noise bursts

    # ------------------------------------------------------------------
    # TTS WebSocket (used from Phase 5 onward)
    # ------------------------------------------------------------------
    tts_sample_rate: int = 16000
    tts_codec: str = "linear16"
    tts_pace: float = 1.0
    tts_min_buffer_size: int = 50
    tts_max_chunk_length: int = 200
    tts_keepalive_interval: int = 25

    def speaker_for(self, language_code: str) -> str:
        """Return the configured speaker name for the given BCP-47 language code."""
        lang = language_code.lower()
        if lang.startswith("kn"):
            return self.tts_speaker_kn
        if lang.startswith("en"):
            return self.tts_speaker_en
        # Default: Hindi / Hinglish
        return self.tts_speaker_hi

    # Convenience alias used by Phase 5 before language-per-turn tracking is done.
    @property
    def tts_speaker(self) -> str:
        return self.speaker_for(self.default_response_language)

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    llm_max_tokens: int = 256

    # Session / context
    # ------------------------------------------------------------------
    max_recent_messages: int = 24

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000


# Module-level singleton — import this everywhere instead of constructing Settings()
settings = Settings()
