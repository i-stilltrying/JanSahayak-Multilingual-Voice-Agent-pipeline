/**
 * frontend/audio-capture.js
 *
 * AudioWorklet processor — runs in the dedicated audio rendering thread.
 *
 * Responsibilities (spec §9, §7):
 *   1. Accept Float32 stereo/mono samples from the browser's media pipeline.
 *   2. Convert stereo → mono by averaging channels.
 *   3. Resample from the AudioContext sample rate (typically 44100 or 48000 Hz)
 *      to exactly 16 000 Hz using linear interpolation.
 *   4. Convert resampled Float32 samples → Int16 (LINEAR16).
 *   5. Post the resulting ArrayBuffer to the main thread via the port.
 *
 * The processor targets frames of ~20 ms (320 samples @ 16 kHz) which is the
 * standard frame size for conversational voice pipelines and keeps per-frame
 * overhead low.
 *
 * IMPORTANT: This file is loaded via AudioContext.audioWorklet.addModule().
 * It MUST be a valid AudioWorklet module — no ES module imports are allowed
 * inside a worklet context in current browser implementations.
 */

const TARGET_SAMPLE_RATE = 16000;
const TARGET_FRAME_SAMPLES = 320; // 20 ms @ 16 kHz

class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super(options);

    /**
     * Source sample rate is passed from the main thread via processorOptions
     * because `sampleRate` is a global inside the worklet context but we also
     * accept it explicitly for testability.
     */
    this._sourceSampleRate = (options.processorOptions || {}).sourceSampleRate || sampleRate;
    this._resampleRatio = this._sourceSampleRate / TARGET_SAMPLE_RATE;

    // Accumulator for resampled Int16 samples waiting to form a full frame.
    this._accumulator = new Int16Array(TARGET_FRAME_SAMPLES * 4); // over-allocate
    this._accLen = 0;

    // Resampled Float32 buffer — reused per process() call to avoid GC pressure.
    this._resampledBuf = new Float32Array(1024);
  }

  /**
   * Called by the audio rendering engine for every block of samples (~128 frames
   * per Web Audio spec).
   *
   * @param {Float32Array[][]} inputs  - inputs[0][channel] = channel data
   * @returns {boolean} true = keep processor alive
   */
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    // Step 1 — Mix down to mono.
    const mono = this._toMono(input);

    // Step 2 — Resample to 16 kHz.
    const resampled = this._resample(mono);

    // Step 3 — Convert Float32 → Int16 and accumulate.
    for (let i = 0; i < resampled.length; i++) {
      const s = Math.max(-1, Math.min(1, resampled[i]));
      this._accumulator[this._accLen++] = s < 0 ? s * 0x8000 : s * 0x7fff;

      // Emit a frame whenever we have TARGET_FRAME_SAMPLES samples.
      if (this._accLen === TARGET_FRAME_SAMPLES) {
        this._emitFrame();
      }
    }

    return true; // Keep processor alive.
  }

  // --------------------------------------------------------------------------
  // Private helpers
  // --------------------------------------------------------------------------

  /**
   * Mix a multi-channel input block to mono by averaging all channels.
   * If the input is already mono, returns it directly (no copy).
   */
  _toMono(input) {
    if (input.length === 1) return input[0];

    const len = input[0].length;
    const mono = new Float32Array(len);
    const scale = 1 / input.length;
    for (let ch = 0; ch < input.length; ch++) {
      for (let i = 0; i < len; i++) {
        mono[i] += input[ch][i] * scale;
      }
    }
    return mono;
  }

  /**
   * Linear-interpolation resampler from _sourceSampleRate → TARGET_SAMPLE_RATE.
   *
   * This is intentionally a simple nearest-neighbour / linear interpolation
   * implementation.  A polyphase filter would be higher quality but adds
   * complexity that is not justified for voice STT where the model is
   * already noise-tolerant.
   */
  _resample(input) {
    const outputLen = Math.round(input.length / this._resampleRatio);
    if (this._resampledBuf.length < outputLen) {
      this._resampledBuf = new Float32Array(outputLen * 2);
    }
    const out = this._resampledBuf;
    for (let i = 0; i < outputLen; i++) {
      const srcIdx = i * this._resampleRatio;
      const lo = Math.floor(srcIdx);
      const hi = Math.min(lo + 1, input.length - 1);
      const t = srcIdx - lo;
      out[i] = input[lo] * (1 - t) + input[hi] * t;
    }
    return out.subarray(0, outputLen);
  }

  /**
   * Copy the accumulated Int16 frame into a new ArrayBuffer and post it to
   * the main thread via the MessagePort.  Transferring the buffer avoids a
   * copy — the worklet side loses ownership, which is safe because we
   * immediately reinitialise _accumulator.
   */
  _emitFrame() {
    // Copy into a new Int16Array backed by a dedicated ArrayBuffer so we can
    // transfer it (avoids a structured-clone copy).
    const frame = new Int16Array(TARGET_FRAME_SAMPLES);
    frame.set(this._accumulator.subarray(0, TARGET_FRAME_SAMPLES));
    this._accLen = 0;

    this.port.postMessage({ type: 'audio', buffer: frame.buffer }, [frame.buffer]);
  }
}

registerProcessor('capture-processor', CaptureProcessor);
