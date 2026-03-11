/**
 * WhisperClient — ES module SDK for Whisper transcription.
 *
 * Three modes of operation:
 *
 * 1. **Batch recording** — record with VAD, upload chunks, transcribe at end:
 *    ```js
 *    const client = new WhisperClient({ server: 'http://localhost:8000' });
 *    await client.startRecording();
 *    const result = await client.stopRecording();
 *    console.log(result.text);
 *    ```
 *
 * 2. **Live streaming** — real-time utterance-by-utterance transcription:
 *    ```js
 *    const client = new WhisperClient({ server: 'http://localhost:8000' });
 *    client.onTranscript = (r) => console.log(`[${r.sequence}] ${r.text}`);
 *    await client.startStreaming();
 *    // ... each utterance fires onTranscript as the user speaks ...
 *    const { text, utterances } = await client.stopStreaming();
 *    ```
 *
 * 3. **Single-shot transcription** — transcribe an existing audio file:
 *    ```js
 *    const result = await client.transcribe(audioBlob);
 *    console.log(result.text);
 *    ```
 */

// -- VAD defaults --
const DEFAULT_CHUNK_MS = 1500;
const DEFAULT_VAD_THRESHOLD = 0.012;
const DEFAULT_VAD_HANGOVER_MS = 800;
const DEFAULT_VAD_WARMUP_MS = 3000;

// -- Streaming VAD defaults --
const DEFAULT_STREAM_UTTERANCE_GAP_MS = 1500;   // silence duration that ends an utterance
const DEFAULT_STREAM_MAX_UTTERANCE_MS = 25000;  // max utterance length before forced flush
const DEFAULT_STREAM_PRE_ROLL_CHUNKS = 5;       // chunks of pre-roll before speech onset
const STREAM_SAMPLE_RATE = 16000;               // request 16 kHz — Whisper's native rate

// -- Helpers --
function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const v = new DataView(buffer);
  const w = (off, str) => {
    for (let i = 0; i < str.length; i++) v.setUint8(off + i, str.charCodeAt(i));
  };
  w(0, 'RIFF');
  v.setUint32(4, 36 + dataSize, true);
  w(8, 'WAVE');
  w(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * bytesPerSample, true);
  v.setUint16(32, bytesPerSample, true);
  v.setUint16(34, 16, true);
  w(36, 'data');
  v.setUint32(40, dataSize, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([v], { type: 'audio/wav' });
}

function mergeBuffers(buffers, length) {
  const out = new Float32Array(length);
  let off = 0;
  for (const b of buffers) {
    out.set(b, off);
    off += b.length;
  }
  return out;
}

export class WhisperClient {
  /**
   * @param {object} [options]
   * @param {string} [options.server]    Base URL of the whisper server (default: same origin)
   * @param {string} [options.token]     Optional auth token (Hypha JWT)
   * @param {string} [options.model]     Whisper model size (default: 'small')
   * @param {string} [options.language]  Language code or 'auto' (default: 'auto')
   * @param {string} [options.prompt]    Optional initial prompt for Whisper
   */
  constructor(options = {}) {
    this.server = (options.server || '').replace(/\/+$/, '');
    this.token = options.token || '';
    this.model = options.model || 'small';
    this.language = options.language || 'auto';
    this.prompt = options.prompt || '';

    // Callbacks
    /** @type {((info: {chunks: number, totalBytes: number}) => void) | null} */
    this.onChunkUploaded = null;
    /** @type {((status: string) => void) | null} */
    this.onStatusChange = null;
    /**
     * Streaming mode only. Called for each transcribed utterance.
     * @type {((result: {text: string, sequence: number, processing_time_s: number}) => void) | null}
     */
    this.onTranscript = null;

    // Internal state
    this._recording = false;
    this._streaming = false;
    this._sessionId = null;
    this._mediaStream = null;
    this._audioCtx = null;
    this._sourceNode = null;
    this._processorNode = null;
    this._muteNode = null;
    this._chunkTimer = null;

    // PCM accumulator (batch mode)
    this._pcmBuffers = [];
    this._pcmLength = 0;

    // VAD state (shared)
    this._speaking = false;
    this._lastSpeech = 0;
    this._noiseFloor = 0.004;
    this._vadWarmupUntil = 0;

    // Chunk tracking (batch mode)
    this._chunksSent = 0;
    this._totalBytesSent = 0;

    // Streaming mode state
    this._streamPreRollBuffers = [];
    this._streamPreRollLength = 0;
    this._streamSpeechBuffers = [];
    this._streamSpeechLength = 0;
    this._streamInSpeech = false;
    this._streamInUtterance = false;
    this._streamLastSpeech = 0;
    this._streamSequence = 0;
    this._streamContext = '';
    this._streamAccumulated = '';
    this._streamTimer = null;
    this._streamSampleRate = 16000;
  }

  /** Whether the client is currently recording (batch mode). */
  get recording() {
    return this._recording;
  }

  /** Whether the client is currently streaming (live utterance mode). */
  get streaming() {
    return this._streaming;
  }

  // -- Auth header --
  _authHeaders() {
    const h = {};
    if (this.token) h['Authorization'] = `Bearer ${this.token}`;
    return h;
  }

  _setStatus(status) {
    if (this.onStatusChange) this.onStatusChange(status);
  }

  // -- Public API --

  /**
   * Check server health.
   * @returns {Promise<object>}
   */
  async getHealth() {
    const resp = await fetch(`${this.server}/api/health`);
    if (!resp.ok) throw new Error(`Health check failed: ${resp.status}`);
    return resp.json();
  }

  /**
   * Start recording from the microphone with VAD-filtered chunk streaming.
   * Resolves once the microphone is active.
   */
  async startRecording() {
    if (this._recording) throw new Error('Already recording');

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('getUserMedia not available in this browser');
    }

    this._mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this._sessionId = crypto.randomUUID();
    this._chunksSent = 0;
    this._totalBytesSent = 0;

    this._audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    this._sourceNode = this._audioCtx.createMediaStreamSource(this._mediaStream);
    this._processorNode = this._audioCtx.createScriptProcessor(4096, 1, 1);
    this._muteNode = this._audioCtx.createGain();
    this._muteNode.gain.value = 0;
    this._sourceNode.connect(this._processorNode);
    this._processorNode.connect(this._muteNode);
    this._muteNode.connect(this._audioCtx.destination);
    await this._audioCtx.resume();

    // Reset VAD state
    this._pcmBuffers = [];
    this._pcmLength = 0;
    this._speaking = false;
    this._lastSpeech = 0;
    this._noiseFloor = 0.004;
    this._vadWarmupUntil = performance.now() + DEFAULT_VAD_WARMUP_MS;

    this._processorNode.onaudioprocess = (e) => {
      if (!this._recording) return;
      const input = e.inputBuffer.getChannelData(0);

      // VAD: compute RMS and update adaptive noise floor
      let sum = 0;
      for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
      const rms = Math.sqrt(sum / input.length);
      this._noiseFloor = 0.97 * this._noiseFloor + 0.03 * Math.min(rms, this._noiseFloor * 3);
      const threshold = Math.max(DEFAULT_VAD_THRESHOLD * 0.35, this._noiseFloor * 2.2);
      this._speaking = rms > threshold;
      if (this._speaking) this._lastSpeech = performance.now();

      const copy = new Float32Array(input.length);
      copy.set(input);
      this._pcmBuffers.push(copy);
      this._pcmLength += copy.length;
    };

    // Flush chunks periodically
    this._chunkTimer = setInterval(() => this._flushChunk(false), DEFAULT_CHUNK_MS);

    this._recording = true;
    this._setStatus('recording');
  }

  /**
   * Stop recording, flush remaining audio, and transcribe the session.
   * @returns {Promise<object>} Transcription result
   */
  async stopRecording() {
    if (!this._recording) throw new Error('Not recording');
    this._recording = false;

    clearInterval(this._chunkTimer);
    this._chunkTimer = null;

    // Flush remaining audio
    if (this._pcmLength > 0 && this._audioCtx) {
      await this._flushChunk(true);
    }

    // Tear down audio nodes
    if (this._processorNode) this._processorNode.disconnect();
    if (this._sourceNode) this._sourceNode.disconnect();
    if (this._muteNode) this._muteNode.disconnect();
    if (this._mediaStream) this._mediaStream.getTracks().forEach((t) => t.stop());
    if (this._audioCtx) {
      this._audioCtx.close();
      this._audioCtx = null;
    }

    if (this._chunksSent === 0) {
      this._setStatus('no_speech');
      return { text: '', chunks: 0, file_size_bytes: 0, processing_time_s: 0, model: this.model, language: this.language, prompt: this.prompt };
    }

    // Transcribe
    this._setStatus('transcribing');
    const params = new URLSearchParams({
      model_size: this.model,
      language: this.language,
      prompt: this.prompt,
    });

    const resp = await fetch(`${this.server}/api/session/${this._sessionId}/transcribe?${params}`, {
      method: 'POST',
      headers: this._authHeaders(),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      this._setStatus('error');
      throw new Error(`Transcription failed (${resp.status}): ${errText}`);
    }

    const result = await resp.json();
    this._setStatus('done');
    return result;
  }

  /**
   * Single-shot transcription of an audio blob/file.
   * @param {Blob} audioBlob
   * @param {string} [filename]
   * @returns {Promise<object>} Transcription result
   */
  async transcribe(audioBlob, filename = 'audio.wav') {
    const form = new FormData();
    form.append('audio', audioBlob, filename);
    const params = new URLSearchParams({
      model_size: this.model,
      language: this.language,
      prompt: this.prompt,
    });

    this._setStatus('transcribing');
    const resp = await fetch(`${this.server}/api/transcribe?${params}`, {
      method: 'POST',
      headers: this._authHeaders(),
      body: form,
    });

    if (!resp.ok) {
      const errText = await resp.text();
      this._setStatus('error');
      throw new Error(`Transcription failed (${resp.status}): ${errText}`);
    }

    const result = await resp.json();
    this._setStatus('done');
    return result;
  }

  /**
   * Start live streaming mode: audio is transcribed utterance-by-utterance in
   * real time as the user speaks. Each completed utterance fires `onTranscript`.
   *
   * @param {object} [options]
   * @param {number} [options.utteranceGapMs]   Silence gap (ms) that ends an utterance (default 1500)
   * @param {number} [options.maxUtteranceMs]   Max utterance length before forced flush (default 25000)
   * @returns {Promise<void>}
   */
  async startStreaming(options = {}) {
    if (this._recording || this._streaming) throw new Error('Already active');

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('getUserMedia not available in this browser');
    }

    const utteranceGapMs = options.utteranceGapMs ?? DEFAULT_STREAM_UTTERANCE_GAP_MS;
    const maxUtteranceMs = options.maxUtteranceMs ?? DEFAULT_STREAM_MAX_UTTERANCE_MS;

    this._mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Request 16 kHz directly — this is Whisper's native sample rate.
    this._audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: STREAM_SAMPLE_RATE });
    this._streamSampleRate = this._audioCtx.sampleRate;
    this._sourceNode = this._audioCtx.createMediaStreamSource(this._mediaStream);
    this._processorNode = this._audioCtx.createScriptProcessor(4096, 1, 1);
    this._muteNode = this._audioCtx.createGain();
    this._muteNode.gain.value = 0;
    this._sourceNode.connect(this._processorNode);
    this._processorNode.connect(this._muteNode);
    this._muteNode.connect(this._audioCtx.destination);
    await this._audioCtx.resume();

    // Reset streaming state
    this._streamPreRollBuffers = [];
    this._streamPreRollLength = 0;
    this._streamSpeechBuffers = [];
    this._streamSpeechLength = 0;
    this._streamInSpeech = false;
    this._streamInUtterance = false;
    this._streamLastSpeech = 0;
    this._streamSequence = 0;
    this._streamContext = '';
    this._streamAccumulated = '';
    this._noiseFloor = 0.004;
    this._vadWarmupUntil = performance.now() + DEFAULT_VAD_WARMUP_MS;

    const maxUtteranceSamples = Math.ceil((maxUtteranceMs / 1000) * this._streamSampleRate);

    this._processorNode.onaudioprocess = (e) => {
      if (!this._streaming) return;
      const input = e.inputBuffer.getChannelData(0);

      // Adaptive VAD
      let sum = 0;
      for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
      const rms = Math.sqrt(sum / input.length);
      this._noiseFloor = 0.97 * this._noiseFloor + 0.03 * Math.min(rms, this._noiseFloor * 3);
      const threshold = Math.max(DEFAULT_VAD_THRESHOLD * 0.35, this._noiseFloor * 2.2);
      const now = performance.now();
      const inWarmup = now < this._vadWarmupUntil;
      const speaking = !inWarmup && rms > threshold;

      const copy = new Float32Array(input.length);
      copy.set(input);

      if (speaking) {
        this._streamLastSpeech = now;
        if (!this._streamInSpeech) {
          this._streamSpeechBuffers = [...this._streamPreRollBuffers, copy];
          this._streamSpeechLength = this._streamPreRollLength + copy.length;
          this._streamPreRollBuffers = [];
          this._streamPreRollLength = 0;
          this._streamInSpeech = true;
          this._streamInUtterance = true;
        } else {
          this._streamSpeechBuffers.push(copy);
          this._streamSpeechLength += copy.length;
        }
      } else {
        if (this._streamInUtterance) {
          this._streamSpeechBuffers.push(copy);
          this._streamSpeechLength += copy.length;
        } else {
          this._streamPreRollBuffers.push(copy);
          this._streamPreRollLength += copy.length;
          if (this._streamPreRollBuffers.length > DEFAULT_STREAM_PRE_ROLL_CHUNKS) {
            const evicted = this._streamPreRollBuffers.shift();
            this._streamPreRollLength -= evicted.length;
          }
        }
      }

      // Force-flush if utterance exceeds max duration
      if (this._streamInUtterance && this._streamSpeechLength >= maxUtteranceSamples) {
        this._streamInSpeech = false;
        this._streamInUtterance = false;
        this._streamLastSpeech = 0;
        this._flushUtterance();
      }
    };

    // Periodically check if utterance has ended (silence > gap)
    this._streamTimer = setInterval(() => {
      if (!this._streaming || !this._streamInUtterance) return;
      const now = performance.now();
      if (now - this._streamLastSpeech > utteranceGapMs) {
        this._streamInSpeech = false;
        this._streamInUtterance = false;
        this._flushUtterance();
      }
    }, 100);

    this._streaming = true;
    this._setStatus('streaming');
  }

  /**
   * Stop live streaming, flush any remaining utterance, and return the
   * accumulated transcript text.
   *
   * @returns {Promise<{text: string, utterances: number}>}
   */
  async stopStreaming() {
    if (!this._streaming) throw new Error('Not streaming');
    this._streaming = false;

    clearInterval(this._streamTimer);
    this._streamTimer = null;

    // Flush whatever utterance audio remains
    let finalFlushPromise = null;
    if (this._streamSpeechLength > 0) {
      this._streamInSpeech = false;
      this._streamInUtterance = false;
      finalFlushPromise = this._flushUtterance();
    }

    // Tear down audio nodes
    if (this._processorNode) this._processorNode.disconnect();
    if (this._sourceNode) this._sourceNode.disconnect();
    if (this._muteNode) this._muteNode.disconnect();
    if (this._mediaStream) this._mediaStream.getTracks().forEach((t) => t.stop());
    if (this._audioCtx) {
      this._audioCtx.close();
      this._audioCtx = null;
    }

    // Wait for final utterance to finish transcribing
    if (finalFlushPromise) await finalFlushPromise;

    this._setStatus('done');
    return {
      text: this._streamAccumulated.trim(),
      utterances: this._streamSequence,
    };
  }

  /**
   * Release resources. Safe to call multiple times.
   */
  destroy() {
    if (this._recording) {
      this._recording = false;
      clearInterval(this._chunkTimer);
    }
    if (this._streaming) {
      this._streaming = false;
      clearInterval(this._streamTimer);
      this._streamTimer = null;
      this._streamSpeechBuffers = [];
      this._streamSpeechLength = 0;
      this._streamPreRollBuffers = [];
      this._streamPreRollLength = 0;
    }
    if (this._processorNode) {
      this._processorNode.disconnect();
      this._processorNode = null;
    }
    if (this._sourceNode) {
      this._sourceNode.disconnect();
      this._sourceNode = null;
    }
    if (this._muteNode) {
      this._muteNode.disconnect();
      this._muteNode = null;
    }
    if (this._mediaStream) {
      this._mediaStream.getTracks().forEach((t) => t.stop());
      this._mediaStream = null;
    }
    if (this._audioCtx) {
      this._audioCtx.close();
      this._audioCtx = null;
    }
    this._chunkTimer = null;
    this._pcmBuffers = [];
    this._pcmLength = 0;
  }

  // -- Internal --

  async _flushUtterance() {
    const buffers = this._streamSpeechBuffers;
    const length = this._streamSpeechLength;
    this._streamSpeechBuffers = [];
    this._streamSpeechLength = 0;

    if (length === 0) return;

    const merged = mergeBuffers(buffers, length);
    const wavBlob = encodeWav(merged, this._streamSampleRate);
    if (wavBlob.size < 1200) return; // too short to be meaningful

    const seq = ++this._streamSequence;
    const form = new FormData();
    form.append('audio', wavBlob, 'utterance.wav');
    const params = new URLSearchParams({
      model_size: this.model,
      language: this.language,
      sequence: seq,
      context: this._streamContext,
    });

    try {
      const resp = await fetch(`${this.server}/api/stream/utterance?${params}`, {
        method: 'POST',
        headers: this._authHeaders(),
        body: form,
      });
      if (resp.ok) {
        const result = await resp.json();
        const trimmed = (result.text || '').trim();
        if (trimmed) {
          this._streamContext = (this._streamContext + ' ' + trimmed).trim().slice(-500);
          this._streamAccumulated = (this._streamAccumulated + ' ' + trimmed).trim();
        }
        if (this.onTranscript) this.onTranscript(result);
      } else {
        let detail = `HTTP ${resp.status}`;
        try {
          const j = await resp.json();
          const d = j.detail;
          if (typeof d === 'string') detail = d;
          else if (Array.isArray(d)) detail = d.map(e => e.msg || JSON.stringify(e)).join('; ');
          else if (d) detail = JSON.stringify(d);
        } catch (_) {}
        console.error(`[WhisperClient] stream utterance seq=${seq} failed: ${detail}`);
        if (this.onTranscript) this.onTranscript({ text: '', sequence: seq, error: detail });
      }
    } catch (err) {
      console.error(`[WhisperClient] stream utterance seq=${seq} network error:`, err);
      if (this.onTranscript) this.onTranscript({ text: '', sequence: seq, error: err.message });
    }
  }

  async _flushChunk(force = false) {
    if (this._pcmLength === 0 || !this._audioCtx) return;
    const now = performance.now();
    const inWarmup = now < this._vadWarmupUntil;
    const active = force || inWarmup || this._speaking || (now - this._lastSpeech < DEFAULT_VAD_HANGOVER_MS);

    const merged = mergeBuffers(this._pcmBuffers, this._pcmLength);
    this._pcmBuffers = [];
    this._pcmLength = 0;

    if (!active) return; // skip silent chunk

    const wavBlob = encodeWav(merged, this._audioCtx.sampleRate);
    if (wavBlob.size < 1200) return; // too tiny

    const form = new FormData();
    form.append('chunk', wavBlob, 'chunk.wav');
    try {
      const resp = await fetch(`${this.server}/api/session/${this._sessionId}/chunk`, {
        method: 'POST',
        headers: this._authHeaders(),
        body: form,
      });
      if (resp.ok) {
        this._chunksSent += 1;
        this._totalBytesSent += wavBlob.size;
        if (this.onChunkUploaded) {
          this.onChunkUploaded({ chunks: this._chunksSent, totalBytes: this._totalBytesSent });
        }
      }
    } catch (_) {
      // Network error — non-fatal, chunk is lost but recording continues
    }
  }
}
