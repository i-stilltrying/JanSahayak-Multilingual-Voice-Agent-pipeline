/**
 * frontend/audio-playback.js
 *
 * Phase 5/6 — Browser-side TTS audio playback.
 *
 * Architecture (spec §54, §142):
 *   Backend sends raw LINEAR16 PCM chunks over the WebSocket as binary frames.
 *   The PlaybackQueue decodes them and schedules them sequentially on a
 *   single AudioContext so playback is gap-free and ordering is preserved.
 *
 * Pipeline:
 *   Backend WebSocket binary frame (Int16 PCM, 16 kHz, mono)
 *     ↓
 *   PlaybackQueue.enqueue(arrayBuffer)
 *     ↓
 *   Convert Int16 → Float32 → AudioBuffer
 *     ↓
 *   AudioBufferSourceNode scheduled on AudioContext clock
 *     ↓
 *   Speaker
 *
 * Barge-in (spec §63):
 *   PlaybackQueue.clear() — cancels all scheduled sources immediately.
 *
 * Sample-rate mismatch:
 *   The backend sends 16 kHz audio. The AudioContext typically runs at
 *   44100 or 48000 Hz. The AudioBuffer is created at SOURCE_SAMPLE_RATE
 *   (16000) and the browser's AudioContext resamples it on playback.
 *   This is the simplest correct approach — no manual resampling needed.
 */

'use strict';

const SOURCE_SAMPLE_RATE = 16000;  // matches TTS output configured in backend

/**
 * PlaybackQueue
 *
 * Buffers incoming PCM chunks and plays them in order without gaps.
 * Uses the AudioContext presentation clock for scheduling so chunks
 * that arrive slightly late are still played in sequence.
 */
class PlaybackQueue {
  /**
   * @param {AudioContext} audioCtx  - The shared AudioContext.
   * @param {function} onPlaybackStarted - Callback when first chunk begins.
   * @param {function} onPlaybackEnded   - Callback when the queue drains.
   */
  constructor(audioCtx, onPlaybackStarted, onPlaybackEnded) {
    /** @type {AudioContext} */
    this._ctx = audioCtx;

    /** @type {function} */
    this._onPlaybackStarted = onPlaybackStarted || (() => {});

    /** @type {function} */
    this._onPlaybackEnded = onPlaybackEnded || (() => {});

    // Next available time on the AudioContext clock for scheduling.
    this._nextStartTime = 0;

    // Tracks all scheduled AudioBufferSourceNodes so we can stop them.
    /** @type {AudioBufferSourceNode[]} */
    this._activeSources = [];

    this._playing = false;
    this._totalBytesReceived = 0;
    this._chunksReceived = 0;
  }

  // --------------------------------------------------------------------------
  // Public API
  // --------------------------------------------------------------------------

  /**
   * Enqueue a PCM chunk for playback.
   * @param {ArrayBuffer} arrayBuffer  Raw Int16 LE PCM bytes at 16 kHz mono.
   */
  enqueue(arrayBuffer) {
    if (!arrayBuffer || arrayBuffer.byteLength === 0) return;

    this._totalBytesReceived += arrayBuffer.byteLength;
    this._chunksReceived++;

    const audioBuffer = this._decodeInt16PCM(arrayBuffer);
    if (!audioBuffer) return;

    const source = this._ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this._ctx.destination);

    // Ensure we're at least a small lookahead ahead of current time
    // to prevent underrun on the first chunk.
    const now = this._ctx.currentTime;
    if (this._nextStartTime < now) {
      this._nextStartTime = now + 0.05; // 50ms lookahead
    }

    source.start(this._nextStartTime);
    this._nextStartTime += audioBuffer.duration;

    // Track source for barge-in cancellation.
    this._activeSources.push(source);
    source.onended = () => {
      this._activeSources = this._activeSources.filter(s => s !== source);
      if (this._activeSources.length === 0 && this._playing) {
        this._playing = false;
        this._onPlaybackEnded();
      }
    };

    if (!this._playing) {
      this._playing = true;
      this._onPlaybackStarted();
    }
  }

  /**
   * Stop all audio immediately and clear the queue.
   * Called on barge-in (spec §63).
   */
  clear() {
    for (const source of this._activeSources) {
      try { source.stop(); } catch (_) {}
    }
    this._activeSources = [];
    this._nextStartTime = 0;
    this._playing = false;
  }

  /** Whether the queue is currently playing audio. */
  get isPlaying() {
    return this._playing;
  }

  // --------------------------------------------------------------------------
  // PCM decode
  // --------------------------------------------------------------------------

  /**
   * Convert a raw Int16 LE PCM ArrayBuffer → AudioBuffer at SOURCE_SAMPLE_RATE.
   * @param {ArrayBuffer} buffer
   * @returns {AudioBuffer|null}
   */
  _decodeInt16PCM(buffer) {
    const int16 = new Int16Array(buffer);
    const numSamples = int16.length;
    if (numSamples === 0) return null;

    let audioBuffer;
    try {
      audioBuffer = this._ctx.createBuffer(
        1,                  // mono
        numSamples,
        SOURCE_SAMPLE_RATE  // 16 kHz — browser AudioContext resamples to its own rate
      );
    } catch (e) {
      console.error('[PlaybackQueue] createBuffer failed:', e);
      return null;
    }

    const channelData = audioBuffer.getChannelData(0); // Float32Array
    const scale = 1.0 / 32768.0;
    for (let i = 0; i < numSamples; i++) {
      channelData[i] = int16[i] * scale;
    }
    return audioBuffer;
  }
}

// Export for use in app.js
window.PlaybackQueue = PlaybackQueue;
