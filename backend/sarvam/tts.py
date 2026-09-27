"""
backend/sarvam/tts.py

Wraps Sarvam bulbul:v3 WebSocket TTS for one agent response stream.

Architecture (spec §51–53, §63–65):
  - One SarvamTTS instance per active LLM turn (created in _run_llm_turn).
  - Opened via `async with SarvamTTS(...) as tts:` which establishes the
    WebSocket and sends the configure message.
  - Caller feeds text via tts.send_text(chunk) / tts.flush_and_finish().
  - Audio arrives as base64-encoded LINEAR16 PCM in AudioOutput.data.audio.
  - We decode it and yield raw bytes to the caller (handler → browser WS).
  - On barge-in: caller simply closes the context manager — the WS closes
    immediately, discarding any buffered audio. No in-band cancel exists
    (spec §63, §65).

SDK call surface (confirmed from sarvamai==0.1.34 source):
  async with client.text_to_speech_streaming.connect(...) as tts_socket:
      await tts_socket.configure(
          target_language_code=...,
          speaker=...,
          pace=...,
          speech_sample_rate=16000,
          output_audio_codec="linear16",
          enable_preprocessing=True,    # required for bulbul:v3
          min_buffer_size=50,
          max_chunk_length=200,
      )
      await tts_socket.convert(text)    # feed text
      await tts_socket.flush()          # force flush after final chunk
      async for event in tts_socket:    # iterate for audio
          if isinstance(event, AudioOutput):
              yield base64.b64decode(event.data.audio)
          elif isinstance(event, EventResponse):  # "final" completion event
              break

Note — `configure()` in the SDK socket client does NOT accept `model` as a
keyword argument at this level; the model is passed to `.connect()` instead.

Keepalive:
  Long-lived idle TTS connections are kept alive by periodic `ping()` calls.
  This class does NOT manage keepalive because each instance is short-lived
  (one per LLM response). Keepalive is only needed for persistent connections.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from types import TracebackType
from typing import TYPE_CHECKING, AsyncIterator

from sarvamai import AsyncSarvamAI
from sarvamai.types.audio_output import AudioOutput
from sarvamai.types.event_response import EventResponse
from sarvamai.types.error_response import ErrorResponse

from backend.config import settings

if TYPE_CHECKING:
    from sarvamai.text_to_speech_streaming.socket_client import (
        AsyncTextToSpeechStreamingSocketClient,
    )

logger = logging.getLogger(__name__)


class SarvamTTS:
    """
    Async context manager wrapping one Sarvam bulbul:v3 TTS WebSocket session.

    Usage:
        async with SarvamTTS(language_code="hi-IN", generation_id=gen) as tts:
            async for pcm_bytes in tts.stream("Your text here"):
                await handler.send_bytes(pcm_bytes)

    The `generation_id` parameter is used only for logging — the caller is
    responsible for the stale-generation check (spec §64).
    """

    def __init__(
        self,
        *,
        language_code: str,
        generation_id: int,
        session_id: str = "",
        api_key: str | None = None,
    ) -> None:
        self._language_code = language_code or settings.default_response_language
        self._generation_id = generation_id
        self._session_id = session_id

        _key = api_key or settings.sarvam_api_key
        self._client = AsyncSarvamAI(
            api_subscription_key=_key
        )
        # Set by __aenter__, used by stream() and __aexit__.
        self._cm = None          # the async context manager from .connect()
        self._socket: "AsyncTextToSpeechStreamingSocketClient | None" = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "SarvamTTS":
        self._cm = self._client.text_to_speech_streaming.connect(
            model=settings.sarvam_tts_model,          # "bulbul:v3"
            send_completion_event=True,                # emit EventResponse on finish
        )
        self._socket = await self._cm.__aenter__()

        # Dynamically determine the correct speaker for the requested language code
        speaker_name = settings.speaker_for(self._language_code)

        # Send configure — MUST be the first message after connect.
        await self._socket.configure(
            target_language_code=self._language_code,
            speaker=speaker_name,
            pace=settings.tts_pace,                   # default 1.0
            speech_sample_rate=settings.tts_sample_rate,   # 16000
            enable_preprocessing=True,                # always enabled for v3
            output_audio_codec=settings.tts_codec,    # "linear16"
            min_buffer_size=settings.tts_min_buffer_size,  # 50
            max_chunk_length=settings.tts_max_chunk_length,# 200
        )

        logger.info(
            "[%s] TTS socket opened  gen=%d  lang=%s  speaker=%s",
            self._session_id,
            self._generation_id,
            self._language_code,
            speaker_name,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if self._cm is not None:
            try:
                await self._cm.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[%s] TTS socket close error (expected on barge-in): %s",
                    self._session_id,
                    exc,
                )
        self._socket = None
        self._cm = None
        return False  # never suppress exceptions

    # ------------------------------------------------------------------
    # Public streaming API
    # ------------------------------------------------------------------

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """
        Send `text` to the TTS socket and yield decoded PCM bytes as
        audio chunks arrive.

        Sends: convert(text) → flush() → iterates socket events until
        the EventResponse("final") completion signal arrives.

        Yields:
            bytes — raw LINEAR16 PCM audio at tts_sample_rate Hz, mono.

        Raises:
            RuntimeError — if called outside the async context manager.
            Exception   — propagated from the WebSocket on fatal error.
        """
        if self._socket is None:
            raise RuntimeError("SarvamTTS.stream() called outside async with block")

        t_start = time.monotonic()
        first_audio_logged = False
        chunks_yielded = 0

        # Feed text and flush to trigger synthesis.
        await self._socket.convert(text)
        await self._socket.flush()

        logger.debug(
            "[%s] TTS convert+flush sent  gen=%d  len=%d chars",
            self._session_id,
            self._generation_id,
            len(text),
        )

        async for event in self._socket:
            if isinstance(event, AudioOutput):
                raw: bytes = base64.b64decode(event.data.audio)
                if not raw:
                    continue

                if not first_audio_logged:
                    ttfa = (time.monotonic() - t_start) * 1000
                    logger.info(
                        "[%s] TTS first audio chunk  gen=%d  %.0f ms  %d bytes",
                        self._session_id,
                        self._generation_id,
                        ttfa,
                        len(raw),
                    )
                    first_audio_logged = True

                chunks_yielded += 1
                yield raw

            elif isinstance(event, EventResponse):
                # "final" completion event — synthesis is done.
                elapsed = (time.monotonic() - t_start) * 1000
                logger.info(
                    "[%s] TTS stream complete  gen=%d  %d chunks  %.0f ms total",
                    self._session_id,
                    self._generation_id,
                    chunks_yielded,
                    elapsed,
                )
                return

            elif isinstance(event, ErrorResponse):
                logger.error(
                    "[%s] TTS error event  gen=%d: %s",
                    self._session_id,
                    self._generation_id,
                    getattr(event, "message", str(event)),
                )
                return

            else:
                # Unknown event type — log and continue.
                logger.debug(
                    "[%s] TTS unknown event type: %s", self._session_id, type(event)
                )
