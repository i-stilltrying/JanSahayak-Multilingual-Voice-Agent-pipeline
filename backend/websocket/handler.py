"""
backend/websocket/handler.py

Per-session WebSocket handler.

Phase 7+8 additions:
  - Creates a ConversationManager + ConversationState per session.
  - _run_llm_turn() calls manager.run_turn() which handles the two-call
    tool-calling pattern and updates session memory (spec §23, §30).
  - _stream_tts() uses state.response_language for correct TTS voice.
  - agent_status events include "searching_knowledge" and "responding"
    so the browser shows tool-execution indicators.

Concurrency model (spec §178):
  Each session has independent asyncio tasks:
    browser_ws_receive  → forwards audio + control messages
    stt_sender          → inside SarvamSTT
    stt_receiver        → inside SarvamSTT
    stt_keepalive       → inside SarvamSTT
    llm_turn (per turn) → drives ConversationManager, streams TTS
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from backend.agent.manager import ConversationManager
from backend.agent.state import ConversationState, TurnMetrics
from backend.audio.generation import stream_sentences
from backend.config import settings
from backend.sarvam.stt import (
    RealtimeError,
    RealtimeTranscriptFinal,
    RealtimeTranscriptPartial,
    RealtimeVadSpeechEnd,
    RealtimeVadSpeechStart,
    SarvamSTT,
    _STTFatalError,
)
from backend.sarvam.tts import SarvamTTS
from backend.websocket.protocol import (
    AgentStatusMessage,
    AgentStatusValue,
    ClientEventMessage,
    ClientEventType,
    ErrorMessage,
    LanguageChangedMessage,
    LatencyMetricsMessage,
    LlmChunkMessage,
    LlmCompleteMessage,
    MessageType,
    SessionReadyMessage,
    StartSessionMessage,
    ToolCallEventMessage,
    TranscriptFinalMessage,
    TranscriptPartialMessage,
)

# Import SDK types for event dispatch — only used in _on_stt_event.
from sarvamai.types.realtime_error import RealtimeError
from sarvamai.types.realtime_transcript_final import RealtimeTranscriptFinal
from sarvamai.types.realtime_transcript_partial import RealtimeTranscriptPartial
from sarvamai.types.realtime_vad_speech_end import RealtimeVadSpeechEnd
from sarvamai.types.realtime_vad_speech_start import RealtimeVadSpeechStart

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """
    Manages the full lifecycle of one browser session over WebSocket.

    Attach points for later phases:
      # [PHASE-4-LLM]   — invoke ConversationManager on transcript_final
      # [PHASE-5-TTS]   — open TTS socket, send PCM to browser
      # [PHASE-7-TOOL]  — wire tool calling into ConversationManager
      # [PHASE-13-BARGEIN] — full generation_id handling
    """

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self.session_id: str = uuid4().hex
        self.turn_id: int = 0
        self.generation_id: int = 0

        # API key supplied by the reviewer via start_session (falls back to
        # settings.sarvam_api_key when None — standard server-side key).
        self.session_api_key: str | None = None

        # STT component — created here with the default key; rebuilt in
        # _on_start_session if the reviewer supplies their own key.
        self._stt: SarvamSTT = SarvamSTT(
            session_id=self.session_id,
            on_event=self._on_stt_event,
        )
        self._stt_started: bool = False

        # Phase 7+8: ConversationManager + ConversationState per session.
        self._manager: ConversationManager = ConversationManager()
        self._state: ConversationState = ConversationState(
            session_id=self.session_id
        )

        # Phase 14: Turn metrics tracker
        self._current_metrics: TurnMetrics | None = None

        # At most one LLM task runs per session at a time.
        self._active_llm_task: asyncio.Task[None] | None = None

        # Dedicated TTS queue and consumer worker task per turn / session.
        self._tts_queue: asyncio.Queue[str | None] | None = None
        self._tts_worker_task: asyncio.Task[None] | None = None

        # Asyncio tasks tracked for clean cancellation on disconnect.
        self._tasks: list[asyncio.Task[None]] = []

        # Last tool executed this turn — forwarded to LatencyMetricsMessage so
        # the browser can label the Tool latency row with the tool name.
        self._last_tool_name: str | None = None
        self._last_tool_display: str | None = None

        # Audio byte stats for logging.
        self._audio_frames_received: int = 0
        self._audio_bytes_received: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main entry point — called once per WebSocket connection.
        Sends session_ready then enters the receive loop.
        """
        await self._send_session_ready()
        # Start STT immediately so it's ready before the first audio arrives.
        await self._start_stt()
        logger.info(
            "[%s] Session started — awaiting audio.", self.session_id
        )
        try:
            await self._receive_loop()
        except WebSocketDisconnect:
            logger.info("[%s] Client disconnected.", self.session_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] Unexpected handler error: %s", self.session_id, exc)
        finally:
            await self._cleanup()

    # ------------------------------------------------------------------
    # STT startup
    # ------------------------------------------------------------------

    async def _start_stt(self) -> None:
        """
        Launch the STT connection in a background task so it doesn't block
        the WebSocket receive loop.
        """
        if self._stt_started:
            return
        self._stt_started = True
        self._spawn_task(self._stt.start())
        logger.info("[%s] STT start task spawned.", self.session_id)

    # ------------------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """
        Continuously receive messages from the browser.
        """
        while True:
            message = await self._ws.receive()

            if "bytes" in message and message["bytes"] is not None:
                await self._handle_audio(message["bytes"])

            elif "text" in message and message["text"] is not None:
                await self._handle_text(message["text"])

            elif message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(code=1000)

    # ------------------------------------------------------------------
    # Audio frame handling
    # ------------------------------------------------------------------

    async def _handle_audio(self, data: bytes) -> None:
        """
        Receive a raw PCM audio frame from the browser AudioWorklet
        and forward it to the STT sender queue.
        """
        self._audio_frames_received += 1
        self._audio_bytes_received += len(data)

        if self._audio_frames_received % 200 == 0:
            logger.debug(
                "[%s] Audio transport — frames=%d  bytes=%d",
                self.session_id,
                self._audio_frames_received,
                self._audio_bytes_received,
            )

        # [PHASE-3-STT] Forward audio to Sarvam STT.
        await self._stt.send_audio(data)

        # [PHASE-5-TTS] No audio back to browser yet.

    # ------------------------------------------------------------------
    # JSON control message handling
    # ------------------------------------------------------------------

    async def _handle_text(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[%s] Malformed JSON: %.120s", self.session_id, raw)
            return

        msg_type = payload.get("type")

        if msg_type == MessageType.START_SESSION:
            await self._on_start_session(StartSessionMessage(**payload))
        elif msg_type == MessageType.CLIENT_EVENT:
            await self._on_client_event(ClientEventMessage(**payload))
        else:
            logger.warning("[%s] Unknown message type: %s", self.session_id, msg_type)

    async def _on_start_session(self, _msg: StartSessionMessage) -> None:
        """Browser sent explicit start_session — trigger proactive greeting."""
        logger.info("[%s] Received start_session. Triggering proactive greeting.", self.session_id)

        # Capture reviewer-supplied API key (may be None — falls back to settings).
        if _msg.api_key:
            self.session_api_key = _msg.api_key
            logger.info("[%s] Reviewer API key received — rebuilding STT/LLM clients.", self.session_id)

            # Rebuild STT with the reviewer key so every reconnect uses it.
            # Stop the existing STT first (it hasn't started streaming yet at
            # this point because start_session is sent before the first audio).
            await self._stt.stop()
            self._stt = SarvamSTT(
                session_id=self.session_id,
                on_event=self._on_stt_event,
                api_key=self.session_api_key,
            )
            self._stt_started = False

            # Rebuild ConversationManager so SarvamLLM uses the reviewer key.
            self._manager = ConversationManager(api_key=self.session_api_key)

        # Advance turn ID for the initial greeting
        self.turn_id += 1

        # Cancel any stray tasks just in case
        if self._active_llm_task and not self._active_llm_task.done():
            self._active_llm_task.cancel()

        # Re-start STT now that the correct client is in place.
        await self._start_stt()

        # Use the default language for the first proactive greeting
        start_lang = settings.default_response_language or "en-IN"
        self._state.input_language = start_lang
        self._state.response_language = start_lang

        # Spawn an invisible first turn to make the agent speak immediately
        self._active_llm_task = self._spawn_task(
            self._run_llm_turn(
                user_text="Hello! Please introduce yourself and list the government schemes you can help with.",
                detected_lang=start_lang,
                turn_id=self.turn_id,
                generation_id=self.generation_id,
                turn_metrics=None,
            )
        )

    async def _on_client_event(self, msg: ClientEventMessage) -> None:
        logger.info(
            "[%s] Client event: %s  data=%s",
            self.session_id,
            msg.event,
            msg.data,
        )
        if msg.event == ClientEventType.BARGE_IN:
            self.generation_id += 1
            self._state.generation_id = self.generation_id
            logger.info(
                "[%s] Barge-in received — incremented generation_id=%d",
                self.session_id,
                self.generation_id,
            )

            # Cancel currently active LLM task if any (Spec §63)
            if self._active_llm_task and not self._active_llm_task.done():
                self._active_llm_task.cancel()
                self._active_llm_task = None
                logger.info("[%s] Cancelled active LLM turn on barge-in.", self.session_id)

            # CRITICAL FIX: Cancel ONLY the TTS worker task.
            # Do NOT iterate over self._tasks — that kills the STT listening task too,
            # permanently deafening the agent after the first barge-in.
            if getattr(self, "_tts_worker_task", None) and not self._tts_worker_task.done():
                self._tts_worker_task.cancel()
                logger.info("[%s] TTS worker successfully cancelled on barge-in.", self.session_id)

    # ------------------------------------------------------------------
    # STT event callback
    # ------------------------------------------------------------------

    async def _on_stt_event(self, event: object) -> None:
        """
        Called by SarvamSTT when it receives a meaningful event.
        Maps SDK types to protocol messages and sends them to the browser.
        """
        ts = time.time() * 1000

        if isinstance(event, RealtimeTranscriptPartial):
            msg = TranscriptPartialMessage(
                session_id=self.session_id,
                turn_id=self.turn_id,
                generation_id=self.generation_id,
                timestamp_ms=ts,
                text=event.text or "",
                language=event.language or "",
            )
            await self._send_json(msg.model_dump())

        elif isinstance(event, RealtimeTranscriptFinal):
            # Finalise the current turn.
            self.turn_id += 1
            if self._current_metrics:
                self._current_metrics.t_stt_final = time.perf_counter()

            msg = TranscriptFinalMessage(
                session_id=self.session_id,
                turn_id=self.turn_id,
                generation_id=self.generation_id,
                timestamp_ms=ts,
                text=event.text or "",
                language=event.language or "",
            )
            await self._send_json(msg.model_dump())
            logger.info(
                "[%s] turn_id=%d  transcript_final: %r",
                self.session_id,
                self.turn_id,
                event.text,
            )

            # Phase 4: kick off a streaming LLM turn.
            # Capture current IDs before awaiting anything.
            snapshot_turn = self.turn_id
            snapshot_gen  = self.generation_id
            user_text     = event.text or ""
            detected_lang = event.language or ""
            metrics_ref   = self._current_metrics

            if detected_lang:
                self._state.input_language = detected_lang
                # CRITICAL FIX: Do NOT auto-sync response_language. STT misclassifies Hinglish.
                # Let the set_language tool handle explicit language switches.

            if user_text.strip():
                # Cancel any in-flight LLM task from a previous turn.
                if self._active_llm_task and not self._active_llm_task.done():
                    self._active_llm_task.cancel()

                self._active_llm_task = self._spawn_task(
                    self._run_llm_turn(
                        user_text,
                        detected_lang,
                        snapshot_turn,
                        snapshot_gen,
                        turn_metrics=metrics_ref,
                    )
                )

        elif isinstance(event, RealtimeVadSpeechStart):
            status_msg = AgentStatusMessage(
                session_id=self.session_id,
                turn_id=self.turn_id,
                generation_id=self.generation_id,
                timestamp_ms=ts,
                status=AgentStatusValue.LISTENING,
            )
            await self._send_json(status_msg.model_dump())

        elif isinstance(event, RealtimeVadSpeechEnd):
            # Start tracking a new turn's metrics from speech end
            self._current_metrics = TurnMetrics(
                t_speech_end=time.perf_counter()
            )
            status_msg = AgentStatusMessage(
                session_id=self.session_id,
                turn_id=self.turn_id,
                generation_id=self.generation_id,
                timestamp_ms=ts,
                status=AgentStatusValue.THINKING,
            )
            await self._send_json(status_msg.model_dump())

        elif isinstance(event, RealtimeError):
            err_msg = ErrorMessage(
                session_id=self.session_id,
                turn_id=self.turn_id,
                generation_id=self.generation_id,
                timestamp_ms=ts,
                code=str(event.code or "stt_error"),
                message=event.message or "Speech recognition error. Please try again.",
            )
            await self._send_json(err_msg.model_dump())

        elif isinstance(event, _STTFatalError):
            err_msg = ErrorMessage(
                session_id=self.session_id,
                turn_id=self.turn_id,
                generation_id=self.generation_id,
                timestamp_ms=ts,
                code="stt_fatal",
                message="Speech recognition is unavailable. Please refresh and try again.",
            )
            await self._send_json(err_msg.model_dump())

    # ------------------------------------------------------------------
    # LLM turn
    # ------------------------------------------------------------------

    async def _run_llm_turn(
        self,
        user_text: str,
        detected_lang: str,
        turn_id: int,
        generation_id: int,
        turn_metrics: TurnMetrics | None = None,
    ) -> None:
        """
        Drive one complete user turn through ConversationManager.

        Phase 7+8 flow:
          1. Build status callback so manager can push agent_status events.
          2. Call manager.run_turn() which returns an async generator of
             final-answer text chunks (after optional tool execution).
          3. Forward chunks to stream_sentences() → _stream_tts() for
             concurrent LLM + TTS + playback (Phase 6 pattern unchanged).
          4. Send llm_chunk/llm_complete events to browser.

        Stale-generation guard: checks self.generation_id at each chunk.
        """
        t_start = time.monotonic()
        ts = time.time() * 1000

        # Reset tool identity at the start of each turn so the latency row
        # never shows a stale name from a previous turn.
        self._last_tool_name = None
        self._last_tool_display = None

        logger.info(
            "[%s] LLM turn start — turn_id=%d  gen=%d  user=%r",
            self.session_id, turn_id, generation_id, user_text[:80],
        )

        async def _status_callback(status_name: str) -> None:
            """Forward manager status updates to browser as agent_status events."""
            _map = {
                "thinking":           AgentStatusValue.THINKING,
                "searching_knowledge": AgentStatusValue.SEARCHING_KNOWLEDGE,
                "responding":         AgentStatusValue.RESPONDING,
            }
            status_val = _map.get(status_name, AgentStatusValue.RESPONDING)
            if self.generation_id == generation_id:
                await self._send_json(
                    AgentStatusMessage(
                        session_id=self.session_id,
                        turn_id=turn_id,
                        generation_id=generation_id,
                        timestamp_ms=time.time() * 1000,
                        status=status_val,
                    ).model_dump()
                )

        full_text: list[str] = []
        had_speech: bool = False

        async def _tts_worker() -> None:
            """
            Dedicated sequential TTS consumer worker.
            Pulls sentences one-by-one from self._tts_queue and streams audio without
            interleaving multiple TTS requests on the WebSocket.
            """
            try:
                while True:
                    if self._tts_queue is None:
                        break
                    sentence = await self._tts_queue.get()
                    if sentence is None or self.generation_id != generation_id:
                        logger.debug(
                            "[%s] TTS worker reached sentinel or stale gen (%d != %d)",
                            self.session_id, generation_id, self.generation_id,
                        )
                        self._tts_queue.task_done()
                        break
                    try:
                        if sentence.strip() and self.generation_id == generation_id:
                            logger.debug(
                                "[%s] TTS worker consuming sentence: %r (turn_id=%d, gen=%d)",
                                self.session_id, sentence[:60], turn_id, generation_id,
                            )
                            if turn_metrics and turn_metrics.t_tts_start is None:
                                turn_metrics.t_tts_start = time.perf_counter()
                            await self._stream_tts(
                                sentence,
                                turn_id,
                                generation_id,
                                turn_metrics=turn_metrics,
                            )
                    finally:
                        self._tts_queue.task_done()
            except asyncio.CancelledError:
                logger.debug(
                    "[%s] TTS worker cancelled (turn_id=%d  gen=%d)",
                    self.session_id, turn_id, generation_id,
                )
                raise

        # Ensure the TTS worker is alive to consume sentences from this turn.
        # After a barge-in the previous worker task is cancelled and done; we
        # must re-spawn it here so sentences don't pile up in an unread queue.
        # _tts_worker is defined above so it is already in scope for this call.
        if getattr(self, "_tts_worker_task", None) is None or self._tts_worker_task.done():
            self._tts_queue = asyncio.Queue()
            self._tts_worker_task = self._spawn_task(_tts_worker())

        tts_worker = self._tts_worker_task

        async def _tool_event_callback(tool_name: str, display_text: str, status: str) -> None:
            """Forward tool-call visibility events to the browser."""
            if status == "started":
                # Record so _stream_tts can attach the name to the latency message.
                self._last_tool_name = tool_name
                self._last_tool_display = display_text
            if self.generation_id == generation_id:
                await self._send_json(
                    ToolCallEventMessage(
                        session_id=self.session_id,
                        turn_id=turn_id,
                        generation_id=generation_id,
                        timestamp_ms=time.time() * 1000,
                        tool_name=tool_name,
                        display_text=display_text,
                        status=status,
                    ).model_dump()
                )

        async def _token_gen() -> "AsyncIterator[str]":
            """
            Consume ConversationManager chunks, forward llm_chunk to browser,
            and yield each chunk for stream_sentences() to consume.
            """
            async for chunk in self._manager.run_turn(
                user_text=user_text,
                detected_lang=detected_lang,
                state=self._state,
                generation_id=generation_id,
                turn_metrics=turn_metrics,
                on_status=_status_callback,
                on_tool_event=_tool_event_callback,
            ):
                if self.generation_id != generation_id:
                    logger.debug(
                        "[%s] Stale LLM chunk discarded (gen %d != %d)",
                        self.session_id, generation_id, self.generation_id,
                    )
                    return
                full_text.append(chunk)
                await self._send_json(
                    LlmChunkMessage(
                        session_id=self.session_id,
                        turn_id=turn_id,
                        generation_id=generation_id,
                        timestamp_ms=time.time() * 1000,
                        text=chunk,
                    ).model_dump()
                )
                yield chunk

        try:
            async for sentence in stream_sentences(_token_gen()):
                if self.generation_id != generation_id:
                    break
                if sentence.strip():
                    had_speech = True
                    if self._tts_queue:
                        await self._tts_queue.put(sentence)

            # Signal worker that LLM producer is done
            if self.generation_id == generation_id and self._tts_queue:
                await self._tts_queue.put(None)
                await tts_worker

        except asyncio.CancelledError:
            logger.info(
                "[%s] LLM turn cancelled (turn_id=%d  gen=%d)",
                self.session_id, turn_id, generation_id,
            )
            # Flush queue and cancel worker
            if self._tts_queue:
                while not self._tts_queue.empty():
                    try:
                        self._tts_queue.get_nowait()
                        self._tts_queue.task_done()
                    except (asyncio.QueueEmpty, ValueError):
                        break
            if not tts_worker.done():
                tts_worker.cancel()
            raise

        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[%s] LLM error (turn_id=%d): %s", self.session_id, turn_id, exc
            )
            if self._tts_queue:
                while not self._tts_queue.empty():
                    try:
                        self._tts_queue.get_nowait()
                        self._tts_queue.task_done()
                    except (asyncio.QueueEmpty, ValueError):
                        break
            if not tts_worker.done():
                tts_worker.cancel()
            await self._send_json(
                ErrorMessage(
                    session_id=self.session_id,
                    turn_id=turn_id,
                    generation_id=generation_id,
                    timestamp_ms=time.time() * 1000,
                    code="llm_error",
                    message="I'm having trouble processing that right now. Please try again.",
                ).model_dump()
            )
            return

        # Emit completion event.
        if self.generation_id == generation_id:
            elapsed_ms = (time.monotonic() - t_start) * 1000
            assembled = "".join(full_text)
            await self._send_json(
                LlmCompleteMessage(
                    session_id=self.session_id,
                    turn_id=turn_id,
                    generation_id=generation_id,
                    timestamp_ms=time.time() * 1000,
                    full_text=assembled,
                    latency_ms=elapsed_ms,
                ).model_dump()
            )
            logger.info(
                "[%s] LLM+TTS pipeline complete — turn=%d  gen=%d  len=%d  %.0f ms",
                self.session_id, turn_id, generation_id, len(assembled), elapsed_ms,
            )

            if not assembled.strip() and not had_speech:
                await self._send_json(
                    AgentStatusMessage(
                        session_id=self.session_id,
                        turn_id=turn_id,
                        generation_id=generation_id,
                        timestamp_ms=time.time() * 1000,
                        status=AgentStatusValue.IDLE,
                    ).model_dump()
                )

    # ------------------------------------------------------------------
    # TTS streaming (Phase 5)
    # ------------------------------------------------------------------

    async def _stream_tts(
        self,
        text: str,
        turn_id: int,
        generation_id: int,
        turn_metrics: TurnMetrics | None = None,
    ) -> None:
        """
        Synthesize `text` via bulbul:v3, forward PCM chunks to the browser.

        Phase 8: uses state.response_language so the TTS voice matches the
        current conversation language (spec §47, §52).
        Barge-in guard: stops on generation_id mismatch.
        """
        # Phase 8: use per-session response language.
        language_code = self._state.response_language or settings.default_response_language

        t_start = time.monotonic()
        bytes_sent = 0

        try:
            async with SarvamTTS(
                language_code=language_code,
                generation_id=generation_id,
                session_id=self.session_id,
                api_key=self.session_api_key,
            ) as tts:
                async for pcm_chunk in tts.stream(text):
                    # Barge-in guard: stop if a newer generation is active.
                    if self.generation_id != generation_id:
                        logger.info(
                            "[%s] TTS barge-in — dropping audio gen=%d (current=%d)",
                            self.session_id,
                            generation_id,
                            self.generation_id,
                        )
                        return  # __aexit__ closes the WS

                    # Record first audio timestamp and emit latency metrics event
                    if turn_metrics and turn_metrics.t_tts_first_audio is None:
                        turn_metrics.t_tts_first_audio = time.perf_counter()
                        breakdown = turn_metrics.calculate_breakdown()
                        await self._send_json(
                            LatencyMetricsMessage(
                                session_id=self.session_id,
                                turn_id=turn_id,
                                generation_id=generation_id,
                                timestamp_ms=time.time() * 1000,
                                metrics=breakdown,
                                tool_name=self._last_tool_name,
                                tool_display=self._last_tool_display,
                            ).model_dump()
                        )

                    await self.send_bytes(pcm_chunk)
                    bytes_sent += len(pcm_chunk)

        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[%s] TTS error (gen=%d): %s", self.session_id, generation_id, exc
            )
            await self._send_json(
                ErrorMessage(
                    session_id=self.session_id,
                    turn_id=turn_id,
                    generation_id=generation_id,
                    timestamp_ms=time.time() * 1000,
                    code="tts_error",
                    message="Voice synthesis failed. Response text is shown above.",
                ).model_dump()
            )

        elapsed_ms = (time.monotonic() - t_start) * 1000
        logger.info(
            "[%s] TTS stream done — gen=%d  bytes=%d  %.0f ms",
            self.session_id, generation_id, bytes_sent, elapsed_ms,
        )

        # Reset UI to ready state.
        if self.generation_id == generation_id:
            await self._send_json(
                AgentStatusMessage(
                    session_id=self.session_id,
                    turn_id=turn_id,
                    generation_id=generation_id,
                    timestamp_ms=time.time() * 1000,
                    status=AgentStatusValue.IDLE,
                ).model_dump()
            )

    # ------------------------------------------------------------------
    # Outbound helpers
    # ------------------------------------------------------------------

    async def _send_session_ready(self) -> None:
        msg = SessionReadyMessage(
            session_id=self.session_id,
            turn_id=self.turn_id,
            generation_id=self.generation_id,
            timestamp_ms=time.time() * 1000,
        )
        await self._send_json(msg.model_dump())

    async def send_json(self, payload: dict) -> None:
        """Public helper — later phases call this to push events to the browser."""
        await self._send_json(payload)

    async def send_bytes(self, data: bytes) -> None:
        """
        Send raw PCM audio bytes to the browser.
        # [PHASE-5-TTS] — called by TTS receiver task.
        """
        try:
            await self._ws.send_bytes(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] Failed to send audio bytes: %s", self.session_id, exc)

    async def _send_json(self, payload: dict) -> None:
        try:
            await self._ws.send_json(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] Failed to send JSON: %s", self.session_id, exc)

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def _spawn_task(self, coro: "asyncio.Coroutine[None, None, None]") -> asyncio.Task[None]:
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self._tasks.append(task)
        task.add_done_callback(self._tasks.remove)
        return task

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cleanup(self) -> None:
        """Cancel all background tasks and close STT on disconnect."""
        # Stop STT first so it sends RealtimeEnd cleanly.
        try:
            await self._stt.stop()
        except Exception as exc:
            logger.warning("[%s] STT stop error: %s", self.session_id, exc)

        for task in list(self._tasks):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        logger.info(
            "[%s] Session cleaned up — turns=%d  audio_frames=%d.",
            self.session_id,
            self.turn_id,
            self._audio_frames_received,
        )
