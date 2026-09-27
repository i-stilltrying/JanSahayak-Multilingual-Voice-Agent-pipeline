/**
 * frontend/app.js
 *
 * Main application controller for JanSahayak.
 *
 * Responsibilities:
 *   - Manage the frontend state machine.
 *   - On Start: request microphone, resume AudioContext, register AudioWorklet,
 *     open WebSocket.
 *   - Forward binary PCM frames from the AudioWorklet to the WebSocket.
 *   - Dispatch incoming WebSocket messages to UI handlers.
 *   - Phase 5+: receive binary PCM from the WebSocket and forward to
 *     audio-playback.js (PlaybackQueue).
 *
 * State machine (spec §176):
 *   IDLE → CONNECTING → LISTENING → PROCESSING → THINKING →
 *   TOOL_EXECUTION → SPEAKING → INTERRUPTED → ERROR → ENDED
 *
 * The WebSocket URL adapts to the page protocol so ws:// is never used
 * on an https:// page (spec §209).
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WS_PATH = '/ws';
const AUDIO_WORKLET_PATH = '/static/audio-capture.js';
const AUDIO_WORKLET_NAME = 'capture-processor';

// Reconnect budget — if the socket closes unexpectedly we attempt up to this
// many reconnects with exponential back-off before surfacing an error.
const MAX_RECONNECT_ATTEMPTS = 3;
const RECONNECT_BASE_DELAY_MS = 1000;

// ---------------------------------------------------------------------------
// State machine values
// ---------------------------------------------------------------------------

const State = Object.freeze({
  IDLE: 'IDLE',
  CONNECTING: 'CONNECTING',
  LISTENING: 'LISTENING',
  PROCESSING: 'PROCESSING',
  THINKING: 'THINKING',
  TOOL_EXECUTION: 'TOOL_EXECUTION',
  SPEAKING: 'SPEAKING',
  INTERRUPTED: 'INTERRUPTED',
  ERROR: 'ERROR',
  ENDED: 'ENDED',
});

// Human-readable labels shown in the status pill.
const STATE_LABELS = {
  [State.IDLE]:           'Idle',
  [State.CONNECTING]:     'Connecting…',
  [State.LISTENING]:      'Listening',
  [State.PROCESSING]:     'Processing',
  [State.THINKING]:       'Thinking',
  [State.TOOL_EXECUTION]: 'Looking up information…',
  [State.SPEAKING]:       'Speaking',
  [State.INTERRUPTED]:    'Interrupted',
  [State.ERROR]:          'Error',
  [State.ENDED]:          'Session ended',
};

// CSS class applied to the status dot per state.
const STATE_DOT_CLASS = {
  [State.IDLE]:           'dot-idle',
  [State.CONNECTING]:     'dot-connecting',
  [State.LISTENING]:      'dot-listening',
  [State.PROCESSING]:     'dot-processing',
  [State.THINKING]:       'dot-processing',
  [State.TOOL_EXECUTION]: 'dot-processing',
  [State.SPEAKING]:       'dot-speaking',
  [State.INTERRUPTED]:    'dot-listening',
  [State.ERROR]:          'dot-error',
  [State.ENDED]:          'dot-idle',
};

// ---------------------------------------------------------------------------
// Language display names
// ---------------------------------------------------------------------------

const LANG_LABELS = {
  'hi-IN': 'हिंदी',
  'en-IN': 'English',
  'kn-IN': 'ಕನ್ನಡ',
};

// ---------------------------------------------------------------------------
// DOM references (populated after DOMContentLoaded)
// ---------------------------------------------------------------------------

let dom = {};

function cacheDom() {
  dom = {
    apiKeyInput:        document.getElementById('api-key-input'),
    btnStart:           document.getElementById('btn-start'),
    btnStop:            document.getElementById('btn-stop'),
    btnRestart:         document.getElementById('btn-restart'),
    statusPill:         document.getElementById('status-pill'),
    statusDot:          document.getElementById('status-dot'),
    statusText:         document.getElementById('status-text'),
    langBadge:          document.getElementById('lang-badge'),
    activityBar:        document.getElementById('activity-bar'),
    activityIcon:       document.getElementById('activity-icon'),
    activityText:       document.getElementById('activity-text'),
    toolActivity:       document.getElementById('tool-activity'),
    toolActivityBadge:  document.getElementById('tool-activity-badge'),
    toolActivityText:   document.getElementById('tool-activity-text'),
    transcript:         document.getElementById('transcript'),
    conversationBox:    document.getElementById('conversation-box'),
    latStt:             document.getElementById('lat-stt'),
    latLlm:             document.getElementById('lat-llm'),
    latTool:            document.getElementById('lat-tool'),
    latToolLabel:       document.getElementById('lat-tool-label'),
    latTts:             document.getElementById('lat-tts'),
    latE2e:             document.getElementById('lat-e2e'),
    errorBanner:        document.getElementById('error-banner'),
  };
}

// ---------------------------------------------------------------------------
// Application class
// ---------------------------------------------------------------------------

class JanSahayakApp {
  constructor() {
    this._state = State.IDLE;

    /** @type {WebSocket|null} */
    this._ws = null;

    /** @type {AudioContext|null} */
    this._audioCtx = null;

    /** @type {AudioWorkletNode|null} */
    this._workletNode = null;

    /** @type {MediaStreamAudioSourceNode|null} */
    this._micSource = null;

    /** @type {MediaStream|null} */
    this._micStream = null;

    // Session metadata echoed back from the backend.
    this._sessionId = null;
    this._currentLanguage = 'hi-IN';

    // Client-side speech-end timestamp for E2E latency measurement.
    this._clientSpeechEnd = null;

    // Reconnect state.
    this._reconnectAttempts = 0;
    this._isStopping = false;

    // Partial transcript currently displayed (replaced on each partial event).
    this._currentPartialId = null;

    // Phase 4: LLM streaming state.
    // The live DOM row being built by _onLlmChunk (null when no stream is active).
    /** @type {HTMLElement|null} */
    this._streamingAgentRow = null;
    // Most recent generation_id seen from the backend — used to drop stale chunks.
    this._activeGenerationId = 0;

    // Barge-in / Local VAD debounce counter
    this._bargeInConsecutiveFrames = 0;

    // Phase 5: TTS playback queue (created in _buildAudioPipeline).
    /** @type {PlaybackQueue|null} */
    this._playbackQueue = null;
  }

  // --------------------------------------------------------------------------
  // Public API
  // --------------------------------------------------------------------------

  /** Called when the Start button is clicked — must be a user gesture. */
  async start() {
    if (this._state !== State.IDLE && this._state !== State.ERROR && this._state !== State.ENDED) return;
    this._isStopping = false;

    this._setState(State.CONNECTING);
    this._clearError();

    try {
      await this._openMicrophone();
      await this._buildAudioPipeline();
      this._openWebSocket();
    } catch (err) {
      console.error('[JanSahayak] Start failed:', err);
      this._showError(_friendlyError(err));
      this._setState(State.ERROR);
      this._teardown();
    }
  }

  /** Called when the Stop button is clicked. */
  stop() {
    this._isStopping = true;
    this._setState(State.ENDED);
    this._teardown();
  }

  // --------------------------------------------------------------------------
  // Microphone & AudioContext
  // --------------------------------------------------------------------------

  async _openMicrophone() {
    this._micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        // Full-duplex config — mic stays active while agent is speaking (spec §8).
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl:  true,
        channelCount:     1,
        sampleRate:       { ideal: 16000 },
      },
      video: false,
    });
  }

  async _buildAudioPipeline() {
    // Create AudioContext — must be resumed inside a user-gesture handler.
    this._audioCtx = new AudioContext({
      // Request 16 kHz directly if the platform supports it; the worklet will
      // resample regardless, but a native-rate context reduces CPU.
      sampleRate: 16000,
    });

    // Resume the context (autoplay policy requires explicit resume on some browsers).
    if (this._audioCtx.state === 'suspended') {
      await this._audioCtx.resume();
    }

    // Phase 5: create the playback queue now that we have an AudioContext.
    this._playbackQueue = new PlaybackQueue(
      this._audioCtx,
      () => {
        // Playback started — notify backend and update UI.
        this._wsSend({ type: 'client_event', event: 'playback_started' });
        this._setActivity('🔊', 'Speaking…');
        this._setState(State.SPEAKING);
        // Record browser-side playback start for true client E2E TTFA (spec §130).
        this._clientPlaybackStarted = performance.now();
        if (this._clientSpeechEnd != null) {
          const clientE2eMs = Math.max(0, this._clientPlaybackStarted - this._clientSpeechEnd);
          if (dom.latE2e) {
            dom.latE2e.textContent = Math.round(clientE2eMs) + ' ms';
          }
        }
      },
      () => {
        // Queue drained — back to listening.
        if (this._state === State.SPEAKING) {
          this._setState(State.LISTENING);
          this._setActivity('🎙', 'Listening…');
        }
      }
    );

    // Register the AudioWorklet processor module.
    await this._audioCtx.audioWorklet.addModule(AUDIO_WORKLET_PATH);

    // Create the worklet node, passing the source sample rate so the processor
    // can compute the correct resample ratio.
    this._workletNode = new AudioWorkletNode(this._audioCtx, AUDIO_WORKLET_NAME, {
      numberOfInputs:  1,
      numberOfOutputs: 0, // Capture-only; no output needed.
      processorOptions: {
        sourceSampleRate: this._audioCtx.sampleRate,
      },
    });

    // Listen for PCM frames emitted by the worklet processor.
    this._workletNode.port.onmessage = (evt) => {
      if (evt.data.type === 'audio') {
        this._onAudioFrame(evt.data.buffer);
      }
    };

    // Connect mic → worklet.
    this._micSource = this._audioCtx.createMediaStreamSource(this._micStream);
    this._micSource.connect(this._workletNode);
  }

  // --------------------------------------------------------------------------
  // WebSocket
  // --------------------------------------------------------------------------

  _openWebSocket() {
    // Construct URL: wss on https pages, ws on http (localhost dev).
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}${WS_PATH}`;
    console.info('[JanSahayak] Connecting to', url);

    this._ws = new WebSocket(url);
    this._ws.binaryType = 'arraybuffer';

    this._ws.onopen    = () => this._onWsOpen();
    this._ws.onmessage = (evt) => this._onWsMessage(evt);
    this._ws.onerror   = (evt) => this._onWsError(evt);
    this._ws.onclose   = (evt) => this._onWsClose(evt);
  }

  _onWsOpen() {
    this._reconnectAttempts = 0;
    console.info('[JanSahayak] WebSocket open — sending start_session.');

    // Read the reviewer API key from the input field.
    const apiKey = (dom.apiKeyInput?.value ?? '').trim();
    if (!apiKey) {
      this._showError('Please enter your Sarvam API key before starting.');
      this._setState(State.ERROR);
      this._teardown();
      return;
    }
    // Persist for the next page load.
    try { localStorage.setItem('sarvam_api_key', apiKey); } catch (_) {}

    this._wsSend({ type: 'start_session', api_key: apiKey });
  }

  _onWsMessage(evt) {
    if (evt.data instanceof ArrayBuffer) {
      // Binary frame = PCM audio from TTS (Phase 5+).
      this._onInboundAudio(evt.data);
      return;
    }

    let msg;
    try {
      msg = JSON.parse(evt.data);
    } catch {
      console.warn('[JanSahayak] Received non-JSON text frame:', evt.data);
      return;
    }

    switch (msg.type) {
      case 'session_ready':       this._onSessionReady(msg);       break;
      case 'transcript_partial':  this._onTranscriptPartial(msg);  break;
      case 'transcript_final':    this._onTranscriptFinal(msg);    break;
      case 'agent_text':          this._onAgentText(msg);          break;
      case 'agent_status':        this._onAgentStatus(msg);        break;
      case 'language_changed':    this._onLanguageChanged(msg);    break;
      case 'latency':             this._onLatency(msg);            break;
      case 'error':               this._onServerError(msg);        break;
      // Phase 4: streaming LLM text
      case 'llm_chunk':           this._onLlmChunk(msg);           break;
      case 'llm_complete':        this._onLlmComplete(msg);        break;
      // Tool-call visibility
      case 'tool_call':           this._onToolCall(msg);           break;
      default:
        console.debug('[JanSahayak] Unknown message type:', msg.type);
    }
  }

  _onWsError(evt) {
    console.error('[JanSahayak] WebSocket error:', evt);
  }

  _onWsClose(evt) {
    console.info('[JanSahayak] WebSocket closed — code=%d reason=%s', evt.code, evt.reason);
    if (this._isStopping) return;

    if (this._reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
      const delay = RECONNECT_BASE_DELAY_MS * Math.pow(2, this._reconnectAttempts);
      this._reconnectAttempts++;
      console.info('[JanSahayak] Reconnecting in %dms (attempt %d)…', delay, this._reconnectAttempts);
      setTimeout(() => this._openWebSocket(), delay);
    } else {
      this._showError('Connection lost. Please refresh the page and try again.');
      this._setState(State.ERROR);
    }
  }

  // --------------------------------------------------------------------------
  // Audio frame pipeline
  // --------------------------------------------------------------------------

  /**
   * Called by the AudioWorklet message handler for every outbound PCM frame.
   * Forwards the raw Int16 ArrayBuffer directly as a binary WebSocket frame.
   */
  _onAudioFrame(buffer) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;

    // Record approximate speech-end time for E2E latency measurement.
    // This is updated on every frame; the last recorded value before the
    // backend sends a transcript_final will be used as client_speech_end.
    this._clientSpeechEnd = performance.now();

    // Local VAD & Barge-in detection (Spec §21, §61–65)
    // If agent is currently speaking or queue is active, calculate RMS volume
    if (this._state === State.SPEAKING || (this._playbackQueue && this._playbackQueue.isPlaying)) {
      const pcm16 = new Int16Array(buffer);
      let sumSquares = 0;
      for (let i = 0; i < pcm16.length; i++) {
        const norm = pcm16[i] / 32768.0;
        sumSquares += norm * norm;
      }
      const rms = Math.sqrt(sumSquares / (pcm16.length || 1));
      console.debug("RMS Volume:", rms);

      const BARGE_IN_RMS_THRESHOLD = 0.08; // Raised threshold to avoid speaker echo/background noise
      const BARGE_IN_REQUIRED_FRAMES = 12; // ~240-300ms sustained loud speech debounce

      if (rms > BARGE_IN_RMS_THRESHOLD) {
        this._bargeInConsecutiveFrames++;
        if (this._bargeInConsecutiveFrames >= BARGE_IN_REQUIRED_FRAMES) {
          console.info(
            '[JanSahayak] Local VAD triggered barge-in — rms=%.4f consecutive_frames=%d',
            rms,
            this._bargeInConsecutiveFrames
          );
          this._bargeInConsecutiveFrames = 0;
          this._triggerBargeIn();
        }
      } else {
        this._bargeInConsecutiveFrames = 0;
      }
    } else {
      this._bargeInConsecutiveFrames = 0;
    }

    this._ws.send(buffer);
  }

  /**
   * Handle immediate client-side barge-in interruption.
   */
  _triggerBargeIn() {
    if (this._playbackQueue) {
      this._playbackQueue.clear();
    }

    // Immediately stop streaming display indicator if active
    if (this._streamingAgentRow) {
      this._streamingAgentRow.classList.remove('turn-streaming');
      this._streamingAgentRow = null;
    }

    this._clearToolActivity();
    this._setState(State.INTERRUPTED);
    this._setActivity('🎙', 'Listening (interrupted)…');

    // Notify backend to cancel active generation tasks & bump generation ID
    this._wsSend({
      type: 'client_event',
      event: 'barge_in',
      data: { client_timestamp_ms: performance.now() },
    });

    // Switch back to LISTENING
    setTimeout(() => {
      if (this._state === State.INTERRUPTED) {
        this._setState(State.LISTENING);
        this._setActivity('🎙', 'Listening…');
      }
    }, 150);
  }

  /**
   * Receive inbound binary PCM audio from the backend (TTS output).
   * Forward to the PlaybackQueue for immediate scheduling.
   */
  _onInboundAudio(buffer) {
    if (this._playbackQueue) {
      this._playbackQueue.enqueue(buffer);
    } else {
      console.debug('[JanSahayak] Inbound audio — queue not ready yet, bytes=%d', buffer.byteLength);
    }
  }

  // --------------------------------------------------------------------------
  // WebSocket message handlers
  // --------------------------------------------------------------------------

  _onSessionReady(msg) {
    this._sessionId = msg.session_id;
    console.info('[JanSahayak] Session ready — id=%s', this._sessionId);
    this._setState(State.LISTENING);
    this._setActivity('🎙', 'Listening for your voice…');
  }

  _onTranscriptPartial(msg) {
    // Phase 5 barge-in: if the agent is speaking when we receive the first
    // partial transcript, the user has interrupted — stop playback immediately.
    if (this._playbackQueue && this._playbackQueue.isPlaying) {
      console.debug('[JanSahayak] Barge-in detected on transcript_partial — clearing playback');
      this._playbackQueue.clear();
      // Notify backend to increment generation_id and drop TTS stream.
      this._wsSend({ type: 'client_event', event: 'barge_in' });
      this._streamingAgentRow = null;
    }

    this._setState(State.LISTENING);
    if (msg.text) this._upsertPartialTranscript(msg.text);
    // Update language badge from STT-detected language.
    if (msg.language) this._updateLangBadge(msg.language);
  }

  _onTranscriptFinal(msg) {
    this._removePartial();
    if (msg.text && msg.text.trim()) {
      this._appendTranscript('user', msg.text);
    }
    // Update language badge from confirmed detected language.
    if (msg.language) this._updateLangBadge(msg.language);
    this._setState(State.PROCESSING);
    this._setActivity('⚙️', 'Processing your request…');
    // Capture speech-end time for E2E latency measurement (Phase 14).
    this._clientSpeechEnd = performance.now();
  }

  // --------------------------------------------------------------------------
  // Phase 4: streaming LLM text handlers
  // --------------------------------------------------------------------------

  /**
   * Append a streamed LLM chunk to a live "agent" row in the transcript.
   * Creates the row on the first chunk; subsequent chunks extend the same row.
   */
  _onLlmChunk(msg) {
    // Guard: ignore chunks from a stale generation.
    if (msg.generation_id !== undefined && msg.generation_id < this._activeGenerationId) {
      return;
    }
    this._activeGenerationId = msg.generation_id ?? this._activeGenerationId;

    if (!this._streamingAgentRow) {
      const box = dom.conversationBox || dom.transcript;
      // Remove placeholder if still present.
      const ph = box.querySelector('.placeholder');
      if (ph) ph.remove();

      // Create a fresh chat bubble using the agent chat bubble structure.
      this._streamingAgentRow = document.createElement('div');
      this._streamingAgentRow.className = 'chat-bubble agent turn-streaming';
      this._streamingAgentRow.innerHTML =
        '<div class="chat-speaker">JanSahayak</div><div class="chat-content"></div>';
      box.appendChild(this._streamingAgentRow);
      box.scrollTop = box.scrollHeight;
    }

    const contentEl = this._streamingAgentRow.querySelector('.chat-content');
    if (contentEl) {
      contentEl.textContent += msg.text;
      const box = dom.conversationBox || dom.transcript;
      box.scrollTop = box.scrollHeight;
    }
  }

  /**
   * Finalise the streaming agent row once the LLM stream is complete.
   * Updates latency display and returns the UI to LISTENING state.
   */
  _onLlmComplete(msg) {
    if (this._streamingAgentRow) {
      this._streamingAgentRow.classList.remove('turn-streaming');
      this._streamingAgentRow = null;
    }

    // Update LLM latency panel.
    if (msg.latency_ms != null) {
      dom.latLlm.textContent = Math.round(msg.latency_ms) + ' ms';
    }

    this._setActivity('✅', 'Ready');
    if (this._state !== State.IDLE && this._state !== State.STOPPED) {
      this._setState(State.LISTENING);
    }
  }

  _onAgentText(msg) {
    this._appendTranscript('agent', msg.text);
    this._setState(State.SPEAKING);
    this._setActivity('🔊', 'Speaking…');
  }

  _onAgentStatus(msg) {
    const icons = {
      listening:          '🎙',
      thinking:           '🤔',
      searching_knowledge:'🔍',
      responding:         '🔊',
      idle:               '✅',
    };
    const labels = {
      listening:          'Listening…',
      thinking:           'Thinking…',
      searching_knowledge:'Searching knowledge base…',
      responding:         'Responding…',
      idle:               'Ready',
    };
    const icon  = icons[msg.status]  ?? '⏳';
    const label = labels[msg.status] ?? msg.status;

    this._setActivity(icon, label);

    // Clear the tool-activity badge as soon as the agent starts speaking.
    if (msg.status === 'responding' || msg.status === 'idle') {
      this._clearToolActivity();
    }

    const stateMap = {
      listening:          State.LISTENING,
      thinking:           State.THINKING,
      searching_knowledge:State.TOOL_EXECUTION,
      responding:         State.SPEAKING,
      idle:               State.LISTENING,
    };
    if (stateMap[msg.status]) this._setState(stateMap[msg.status]);
  }

  // --------------------------------------------------------------------------
  // Tool-call visibility
  // --------------------------------------------------------------------------

  _onToolCall(msg) {
    if (!dom.toolActivity) return;
    if (msg.status === 'started') {
      dom.toolActivityBadge.textContent = msg.tool_name ?? '';
      dom.toolActivityText.textContent  = msg.display_text ?? '';
      dom.toolActivity.hidden = false;
      dom.toolActivity.classList.remove('tool-complete');
      dom.toolActivity.classList.add('tool-running');
      // Pre-update the latency label so it names the tool even before timing arrives.
      if (dom.latToolLabel && msg.tool_name) {
        dom.latToolLabel.textContent = `Tool (${msg.tool_name})`;
      }
    } else if (msg.status === 'completed') {
      dom.toolActivity.classList.remove('tool-running');
      dom.toolActivity.classList.add('tool-complete');
      // Auto-hide after a short display window; the agent response will follow.
      clearTimeout(this._toolActivityTimer);
      this._toolActivityTimer = setTimeout(() => this._clearToolActivity(), 2500);
    }
  }

  _clearToolActivity() {
    if (!dom.toolActivity) return;
    clearTimeout(this._toolActivityTimer);
    dom.toolActivity.hidden = true;
    dom.toolActivity.classList.remove('tool-running', 'tool-complete');
  }

  _onLanguageChanged(msg) {
    this._updateLangBadge(msg.language);
    console.info('[JanSahayak] Language changed to %s', msg.language);
  }

  /** Update the language badge with a BCP-47 tag from any source. */
  _updateLangBadge(langCode) {
    if (!langCode || langCode === this._currentLanguage) return;
    this._currentLanguage = langCode;
    dom.langBadge.textContent = LANG_LABELS[langCode] ?? langCode;
  }

  _onLatency(msg) {
    const m = msg.metrics || {};
    if (m.stt_ms    != null && dom.latStt)  dom.latStt.textContent  = Math.round(m.stt_ms)    + ' ms';
    if (m.llm_ms    != null && dom.latLlm)  dom.latLlm.textContent  = Math.round(m.llm_ms)    + ' ms';
    if (m.tool_ms   != null && dom.latTool) dom.latTool.textContent = Math.round(m.tool_ms)   + ' ms';
    if (m.tts_ms    != null && dom.latTts)  dom.latTts.textContent  = Math.round(m.tts_ms)    + ' ms';
    // If client E2E TTFA has not already been populated by playback queue start, use backend E2E
    if (m.e2e_ms    != null && dom.latE2e && (!dom.latE2e.textContent || dom.latE2e.textContent === '—')) {
      dom.latE2e.textContent = Math.round(m.e2e_ms) + ' ms';
    }
    // Update the Tool row label with the name of the tool that was executed.
    if (dom.latToolLabel) {
      const name = msg.tool_name ?? null;
      dom.latToolLabel.textContent = name ? `Tool (${name})` : 'Tool';
    }
  }

  _onServerError(msg) {
    console.error('[JanSahayak] Server error code=%s message=%s', msg.code, msg.message);
    this._showError(msg.message || 'Something went wrong. Please try again.');
  }

  // --------------------------------------------------------------------------
  // WebSocket send helper
  // --------------------------------------------------------------------------

  _wsSend(obj) {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(obj));
    }
  }

  // --------------------------------------------------------------------------
  // State machine
  // --------------------------------------------------------------------------

  _setState(newState) {
    if (this._state === newState) return;
    console.debug('[JanSahayak] State: %s → %s', this._state, newState);
    this._state = newState;
    this._updateStatusUI();

    const isActive = newState !== State.IDLE && newState !== State.ENDED && newState !== State.ERROR;
    dom.btnStart.disabled = isActive;
    dom.btnStop.disabled  = !isActive;
  }

  _updateStatusUI() {
    const label   = STATE_LABELS[this._state]   ?? this._state;
    const dotCls  = STATE_DOT_CLASS[this._state] ?? 'dot-idle';

    dom.statusText.textContent = label;
    dom.statusDot.className    = `dot ${dotCls}`;
  }

  // --------------------------------------------------------------------------
  // Activity bar
  // --------------------------------------------------------------------------

  _setActivity(icon, text) {
    dom.activityBar.hidden = false;
    dom.activityIcon.textContent = icon;
    dom.activityText.textContent = text;
  }

  // --------------------------------------------------------------------------
  // Transcript helpers
  // --------------------------------------------------------------------------

  _appendTranscript(role, text) {
    const isUser = role === 'user';
    const speaker = isUser ? 'You' : 'JanSahayak';
    const box = dom.conversationBox || dom.transcript;

    // Remove the placeholder if it's still there.
    const placeholder = box.querySelector('.placeholder');
    if (placeholder) placeholder.remove();

    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${isUser ? 'user' : 'agent'}`;
    bubble.innerHTML = `
      <div class="chat-speaker">${speaker}</div>
      <div class="chat-content">${_escapeHtml(text)}</div>
    `;
    box.appendChild(bubble);
    box.scrollTop = box.scrollHeight;
  }

  _upsertPartialTranscript(text) {
    const box = dom.conversationBox || dom.transcript;
    let bubble = box.querySelector('.turn-partial');
    if (!bubble) {
      // Remove placeholder on first partial.
      const ph = box.querySelector('.placeholder');
      if (ph) ph.remove();

      bubble = document.createElement('div');
      bubble.className = 'chat-bubble user turn-partial';
      box.appendChild(bubble);
    }
    bubble.innerHTML = `
      <div class="chat-speaker">You</div>
      <div class="chat-content"><em>${_escapeHtml(text)}</em></div>
    `;
    box.scrollTop = box.scrollHeight;
  }

  _removePartial() {
    const box = dom.conversationBox || dom.transcript;
    const p = box.querySelector('.turn-partial');
    if (p) p.remove();
  }

  // --------------------------------------------------------------------------
  // Error display
  // --------------------------------------------------------------------------

  _showError(message) {
    dom.errorBanner.textContent = message;
    dom.errorBanner.hidden = false;
  }

  _clearError() {
    dom.errorBanner.textContent = '';
    dom.errorBanner.hidden = true;
  }

  // --------------------------------------------------------------------------
  // Teardown
  // --------------------------------------------------------------------------

  _teardown() {
    if (this._workletNode) {
      try { this._workletNode.disconnect(); } catch (_) {}
      this._workletNode = null;
    }
    if (this._micSource) {
      try { this._micSource.disconnect(); } catch (_) {}
      this._micSource = null;
    }
    if (this._micStream) {
      try { this._micStream.getTracks().forEach((t) => t.stop()); } catch (_) {}
      this._micStream = null;
    }
    if (this._audioCtx) {
      try { this._audioCtx.close().catch(() => {}); } catch (_) {}
      this._audioCtx = null;
    }
    if (this._ws) {
      try { this._ws.close(1000, 'Session ended'); } catch (_) {}
      this._ws = null;
    }

    // Clear global references as well to guarantee clean state on reconnects
    window.appWebSocket = null;
    window.audioContext = null;

    // Clear any in-flight streaming row and playback queue so reconnects start clean.
    if (this._playbackQueue) {
      this._playbackQueue.clear();
      this._playbackQueue = null;
    }
    this._streamingAgentRow = null;
    this._activeGenerationId = 0;
  }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function _escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function _friendlyError(err) {
  if (err instanceof DOMException) {
    if (err.name === 'NotAllowedError')  return 'Microphone permission was denied. Please allow access and try again.';
    if (err.name === 'NotFoundError')    return 'No microphone found. Please connect a microphone and try again.';
    if (err.name === 'NotReadableError') return 'Microphone is in use by another application.';
  }
  return err?.message ?? 'An unexpected error occurred. Please try again.';
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  cacheDom();

  // Pre-populate API key field from localStorage if available.
  if (dom.apiKeyInput) {
    try {
      const saved = localStorage.getItem('sarvam_api_key');
      if (saved) dom.apiKeyInput.value = saved;
    } catch (_) {}
  }

  const app = new JanSahayakApp();

  dom.btnStart.addEventListener('click', () => app.start());
  dom.btnStop.addEventListener('click',  () => app.stop());

  if (dom.btnRestart) {
    dom.btnRestart.addEventListener('click', () => {
      // 1. Aggressively tear down the active session
      app.stop();

      // 2. Wipe the conversation UI
      const box = dom.conversationBox || dom.transcript;
      if (box) {
        box.innerHTML = '<p class="placeholder">Conversation cleared. Ready to start a new session.</p>';
      }

      // 3. Reset the latency table and tool labels
      if (dom.latToolLabel) dom.latToolLabel.textContent = 'Tool';
      if (dom.latStt) dom.latStt.textContent = '—';
      if (dom.latLlm) dom.latLlm.textContent = '—';
      if (dom.latTool) dom.latTool.textContent = '—';
      if (dom.latTts) dom.latTts.textContent = '—';
      if (dom.latE2e) dom.latE2e.textContent = '—';

      // Clear any tool activity indicators
      app._clearToolActivity();
    });
  }
});
