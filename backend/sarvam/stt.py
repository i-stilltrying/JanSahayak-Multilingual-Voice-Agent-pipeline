"""
backend/sarvam/stt.py

Manages the Sarvam saaras:v3-realtime WebSocket STT connection for one session.

Architecture (spec §116, §178):
  Two asyncio tasks run concurrently per session:
    1. _sender_task  — pulls raw PCM bytes from an asyncio.Queue and forwards
                       them to Sarvam as base64-encoded RealtimeAudioInput JSON.
    2. _receiver_task — iterates incoming SDK events and invokes the caller's
                       callback for each event type.

Audio encoding note:
  The Sarvam realtime API requires audio as base64-encoded strings inside a
  JSON envelope (RealtimeAudioInput), NOT as raw binary WebSocket frames.
  We encode each PCM chunk at the point of sending.

Keepalive:
  A RealtimePing is sent every STT_KEEPALIVE_INTERVAL seconds while the
  connection is active (spec §67).

Reconnect behaviour:
  Non-fatal STT errors trigger a reconnect with exponential back-off up to
  MAX_RECONNECT_ATTEMPTS times. Fatal errors surface to the handler immediately.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Awaitable, Callable

from sarvamai import AsyncSarvamAI
from sarvamai.types.realtime_audio_input import RealtimeAudioInput
from sarvamai.types.realtime_error import RealtimeError
from sarvamai.types.realtime_ping import RealtimePing
from sarvamai.types.realtime_session_begin import RealtimeSessionBegin
from sarvamai.types.realtime_session_end import RealtimeSessionEnd
from sarvamai.types.realtime_transcript_final import RealtimeTranscriptFinal
from sarvamai.types.realtime_transcript_partial import RealtimeTranscriptPartial
from sarvamai.types.realtime_vad_speech_end import RealtimeVadSpeechEnd
from sarvamai.types.realtime_vad_speech_start import RealtimeVadSpeechStart

from backend.config import settings

logger = logging.getLogger(__name__)

# Maximum audio queue depth — prevents unbounded growth if STT falls behind.
_AUDIO_QUEUE_MAX = 512
# Base reconnect delay in seconds.
_RECONNECT_BASE_DELAY = 1.0
# Max reconnect attempts before giving up.
MAX_RECONNECT_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Event callback type alias
# ---------------------------------------------------------------------------

# Called by SarvamSTT when a meaningful STT event occurs.
# Signature: async def callback(event) -> None
STTEventCallback = Callable[..., Awaitable[None]]


# ---------------------------------------------------------------------------
# SarvamSTT
# ---------------------------------------------------------------------------


class SarvamSTT:
    """
    Wraps the Sarvam realtime STT WebSocket for a single session.

    Usage:
        stt = SarvamSTT(session_id, on_event_callback)
        await stt.start()          # open connection, start tasks
        await stt.send_audio(pcm)  # enqueue audio bytes
        await stt.stop()           # graceful shutdown
    """

    def __init__(
        self,
        session_id: str,
        on_event: STTEventCallback,
        api_key: str | None = None,
    ) -> None:
        self._session_id = session_id
        self._on_event = on_event
        self._api_key: str = api_key or settings.sarvam_api_key

        # Bounded queue: WebSocketHandler pushes PCM bytes; _sender_task pulls.
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=_AUDIO_QUEUE_MAX
        )

        self._sender_task: asyncio.Task[None] | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None

        self._stopped = False
        self._reconnect_attempts = 0

        # NOTE: _client is intentionally NOT created here.
        # A fresh AsyncSarvamAI instance is created inside _connect_and_run on
        # every connection attempt so that a TTS barge-in that poisons the shared
        # httpx connection pool cannot corrupt the STT reconnect path.

        # Timestamp of last received final transcript (for latency measurement).
        self.t_stt_final: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the STT WebSocket and launch background tasks."""
        logger.info("[%s][STT] Opening saaras:v3-realtime connection.", self._session_id)
        await self._connect_and_run()

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """
        Enqueue a raw LINEAR16 PCM frame for forwarding to Sarvam STT.

        This is non-blocking: if the queue is full (STT has fallen behind)
        the oldest frame is dropped to prevent unbounded memory growth.
        """
        if self._stopped:
            return
        try:
            self._audio_queue.put_nowait(pcm_bytes)
        except asyncio.QueueFull:
            # Drop oldest frame and insert new one — prefer freshness.
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._audio_queue.put_nowait(pcm_bytes)
            except asyncio.QueueFull:
                pass

    async def stop(self) -> None:
        """Signal shutdown and cancel all background tasks."""
        if self._stopped:
            return
        self._stopped = True
        # Sentinel to unblock the sender task.
        try:
            self._audio_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        for task in (self._sender_task, self._receiver_task, self._keepalive_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        logger.info("[%s][STT] Connection stopped.", self._session_id)

    # ------------------------------------------------------------------
    # Internal connection management
    # ------------------------------------------------------------------

    async def _connect_and_run(self) -> None:
        """
        Open the STT socket via the SDK async context manager and run the
        sender + receiver tasks. Creates a fresh client per connection to 
        prevent httpx pool exhaustion during barge-in deadlocks.
        """
        import websockets.exceptions

        while not self._stopped:
            # CRITICAL FIX 1: Fresh client per loop to isolate the connection pool
            client = AsyncSarvamAI(api_subscription_key=self._api_key)
            
            try:
                async with client.speech_to_text_realtime_streaming.connect(
                    language_code="auto",
                    model=settings.sarvam_stt_model,
                    stream_type="fast",
                    mode="transcribe",
                    endpointing="vad",
                    encoding="linear16",
                    sample_rate=str(settings.stt_sample_rate),
                    threshold=str(settings.stt_vad_threshold),
                    silence_duration_ms=str(settings.stt_silence_duration_ms),
                    min_speech_duration_ms=str(settings.stt_min_speech_duration_ms),
                ) as stt_socket:
                    
                    self._reconnect_attempts = 0
                    logger.info("[%s][STT] Connected to Sarvam realtime STT.", self._session_id)

                    self._sender_task = asyncio.create_task(self._sender_loop(stt_socket))
                    self._receiver_task = asyncio.create_task(self._receiver_loop(stt_socket))
                    self._keepalive_task = asyncio.create_task(self._keepalive_loop(stt_socket))

                    # CRITICAL FIX 2: Wait for the FIRST task to fail/exit
                    done, pending = await asyncio.wait(
                        [self._sender_task, self._receiver_task, self._keepalive_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Cancel pending tasks, then wait up to 1 s for them to exit.
                    # asyncio.wait with a timeout — never gather — so a stuck SDK
                    # task cannot block the reconnect loop indefinitely.
                    for task in pending:
                        task.cancel()
                    if pending:
                        _, _ = await asyncio.wait(pending, timeout=1.0)

                    for task in done:
                        exc = task.exception()
                        if exc is not None and not isinstance(exc, asyncio.CancelledError):
                            raise exc

                if self._stopped:
                    break
                
                await asyncio.sleep(0.05)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._stopped:
                    break

                is_normal_close = isinstance(exc, websockets.exceptions.ConnectionClosedOK) or (
                    isinstance(exc, websockets.exceptions.ConnectionClosed) and exc.rcvd and exc.rcvd.code == 1000
                )

                if is_normal_close:
                    await asyncio.sleep(0.05)
                    continue

                self._reconnect_attempts += 1
                if self._reconnect_attempts > MAX_RECONNECT_ATTEMPTS:
                    logger.error("[%s][STT] Exceeded max reconnect attempts.", self._session_id)
                    await self._on_event(_STTFatalError(message=str(exc)))
                    break
                
                delay = _RECONNECT_BASE_DELAY * (2 ** (self._reconnect_attempts - 1))
                logger.warning("[%s][STT] Connection error: %s — reconnecting in %.1fs", self._session_id, exc, delay)
                await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Sender task
    # ------------------------------------------------------------------

    async def _sender_loop(self, stt_socket) -> None:
        """
        Pull PCM frames from the audio queue and send them to Sarvam STT
        as base64-encoded RealtimeAudioInput JSON messages.
        """
        import websockets.exceptions

        while not self._stopped:
            chunk = await self._audio_queue.get()

            if chunk is None:
                # Sentinel value — session is stopping.
                break

            # Encode bytes to base64 string as required by the Sarvam API.
            audio_b64 = base64.b64encode(chunk).decode("utf-8")
            msg = RealtimeAudioInput(audio=audio_b64)
            try:
                await stt_socket.send_realtime_audio_input(msg)
            except websockets.exceptions.ConnectionClosed as exc:
                logger.info("[%s][STT] Sender socket closed (%s) — breaking loop for reconnect", self._session_id, exc)
                raise
            except Exception as exc:
                logger.warning("[%s][STT] send_audio error: %s", self._session_id, exc)
                raise  # Trigger reconnect in _connect_and_run.

    # ------------------------------------------------------------------
    # Receiver task
    # ------------------------------------------------------------------

    async def _receiver_loop(self, stt_socket) -> None:
        """
        Iterate incoming events from the STT socket and forward them
        to the session callback.
        """
        import websockets.exceptions

        try:
            async for event in stt_socket:
                if self._stopped:
                    break
                await self._dispatch(event)
        except websockets.exceptions.ConnectionClosed as exc:
            logger.info("[%s][STT] Receiver socket closed (%s) — breaking loop for reconnect", self._session_id, exc)
            raise

    async def _dispatch(self, event: object) -> None:
        """Map SDK event types to session callback invocations."""
        if isinstance(event, RealtimeSessionBegin):
            logger.debug(
                "[%s][STT] Session begin — request_id=%s",
                self._session_id,
                event.request_id,
            )

        elif isinstance(event, RealtimeTranscriptPartial):
            logger.debug(
                "[%s][STT] Partial: %r  lang=%s",
                self._session_id,
                event.text,
                event.language,
            )
            await self._on_event(event)

        elif isinstance(event, RealtimeTranscriptFinal):
            self.t_stt_final = time.monotonic()
            logger.info(
                "[%s][STT] Final: %r  lang=%s",
                self._session_id,
                event.text,
                event.language,
            )
            await self._on_event(event)

        elif isinstance(event, RealtimeVadSpeechStart):
            logger.debug(
                "[%s][STT] VAD speech_start utterance=%s",
                self._session_id,
                event.utterance_idx,
            )
            await self._on_event(event)

        elif isinstance(event, RealtimeVadSpeechEnd):
            logger.debug(
                "[%s][STT] VAD speech_end utterance=%s",
                self._session_id,
                event.utterance_idx,
            )
            await self._on_event(event)

        elif isinstance(event, RealtimeError):
            logger.error(
                "[%s][STT] Error code=%s fatal=%s message=%s",
                self._session_id,
                event.code,
                event.is_fatal,
                event.message,
            )
            await self._on_event(event)
            if event.is_fatal:
                raise RuntimeError(
                    f"Fatal STT error: code={event.code} message={event.message}"
                )

        elif isinstance(event, RealtimeSessionEnd):
            logger.info("[%s][STT] Session end received.", self._session_id)

        else:
            logger.debug(
                "[%s][STT] Unhandled event type: %s", self._session_id, type(event).__name__
            )

    # ------------------------------------------------------------------
    # Keepalive task
    # ------------------------------------------------------------------

    async def _keepalive_loop(self, stt_socket) -> None:
        """
        Send a RealtimePing every STT_KEEPALIVE_INTERVAL seconds to keep
        the socket alive on idle connections (spec §67).
        """
        import websockets.exceptions

        interval = settings.tts_keepalive_interval  # reuse same env var
        while not self._stopped:
            await asyncio.sleep(interval)
            if self._stopped:
                break
            try:
                await stt_socket.send_realtime_ping(RealtimePing())
                logger.debug("[%s][STT] Keepalive ping sent.", self._session_id)
            except websockets.exceptions.ConnectionClosed as exc:
                logger.info(
                    "[%s][STT] Keepalive ping socket closed (%s) — exiting loop cleanly for reconnect",
                    self._session_id,
                    exc,
                )
                break
            except Exception as exc:
                logger.warning(
                    "[%s][STT] Keepalive ping failed: %s", self._session_id, exc
                )
                raise  # Trigger reconnect.


# ---------------------------------------------------------------------------
# Internal sentinel for fatal errors surfaced via callback
# ---------------------------------------------------------------------------


class _STTFatalError:
    """Synthetic event emitted when we exhaust reconnect attempts."""

    def __init__(self, message: str) -> None:
        self.message = message
