export interface WhisperClientOptions {
  /** Base URL of the whisper server (default: same origin). */
  server?: string;
  /** Optional Hypha auth token. */
  token?: string;
  /** Whisper model size (default: 'small'). */
  model?: string;
  /** Language code or 'auto' (default: 'auto'). */
  language?: string;
  /** Optional initial prompt for Whisper. */
  prompt?: string;
  /** Enable speaker diarization (requires whisperX backend + HF token). Default: false. */
  diarize?: boolean;
  /** Enable word-level alignment (requires whisperX backend). Default: false. */
  alignWords?: boolean;
  /** Minimum number of speakers for diarization. Default: 1. */
  minSpeakers?: number;
  /** Maximum number of speakers for diarization. Default: 10. */
  maxSpeakers?: number;
}

export interface TranscriptSegment {
  text: string;
  start?: number;
  end?: number;
  speaker?: string;
  words?: Array<{ word: string; start?: number; end?: number; score?: number }>;
}

export interface TranscriptionResult {
  text: string;
  chunks: number;
  file_size_bytes: number;
  processing_time_s: number;
  model: string;
  language: string;
  prompt: string;
  /** Present when whisperX backend is used with align_words or diarize. */
  segments?: TranscriptSegment[];
}

/** Result from a single utterance in streaming mode. */
export interface StreamUtteranceResult {
  /** Transcribed text for this utterance. */
  text: string;
  /** 1-based sequence number (useful for ordering out-of-order responses). */
  sequence: number;
  file_size_bytes: number;
  processing_time_s: number;
  model: string;
  language: string;
  /** Present when whisperX backend returns segment data. */
  segments?: TranscriptSegment[];
}

/** Result returned by stopStreaming(). */
export interface StreamingResult {
  /** Full concatenated transcript of all utterances so far. */
  text: string;
  /** Total number of utterances transcribed. */
  utterances: number;
}

export interface StreamingOptions {
  /** Silence duration (ms) that ends an utterance. Default: 1000. */
  utteranceGapMs?: number;
  /** Max utterance duration (ms) before a forced flush. Default: 25000. */
  maxUtteranceMs?: number;
}

export interface ChunkUploadInfo {
  chunks: number;
  totalBytes: number;
}

export interface HealthResponse {
  status: string;
  backend: string;
  default_model: string;
  device: string;
  compute_type: string;
  auth_required: boolean;
}

export type StatusValue = 'recording' | 'transcribing' | 'streaming' | 'done' | 'error' | 'no_speech';

export class WhisperClient {
  /** Base URL of the whisper server. */
  server: string;
  /** Auth token. */
  token: string;
  /** Whisper model size. */
  model: string;
  /** Language code or 'auto'. */
  language: string;
  /** Optional initial prompt. */
  prompt: string;
  /** Enable speaker diarization. */
  diarize: boolean;
  /** Enable word-level alignment. */
  alignWords: boolean;
  /** Minimum speakers for diarization. */
  minSpeakers: number;
  /** Maximum speakers for diarization. */
  maxSpeakers: number;

  /** Called after each chunk is uploaded to the server (batch mode). */
  onChunkUploaded: ((info: ChunkUploadInfo) => void) | null;
  /** Called when the client status changes. */
  onStatusChange: ((status: StatusValue) => void) | null;
  /**
   * Streaming mode only. Called as each utterance is transcribed in real time.
   * May be called concurrently — use `sequence` to order results if needed.
   */
  onTranscript: ((result: StreamUtteranceResult) => void) | null;

  /** Whether the client is currently recording (batch mode). */
  readonly recording: boolean;
  /** Whether the client is currently streaming (live utterance mode). */
  readonly streaming: boolean;

  constructor(options?: WhisperClientOptions);

  /** Check server health. */
  getHealth(): Promise<HealthResponse>;

  /** Start recording from the microphone with VAD-filtered chunk streaming (batch mode). */
  startRecording(): Promise<void>;

  /** Stop recording, flush remaining audio, and transcribe the session (batch mode). */
  stopRecording(): Promise<TranscriptionResult>;

  /**
   * Start live streaming mode. Audio is transcribed utterance-by-utterance as
   * the user speaks. Each completed utterance fires `onTranscript`.
   */
  startStreaming(options?: StreamingOptions): Promise<void>;

  /**
   * Stop live streaming, flush any remaining utterance, and return the full
   * accumulated transcript.
   */
  stopStreaming(): Promise<StreamingResult>;

  /** Single-shot transcription of an audio blob/file. */
  transcribe(audioBlob: Blob, filename?: string): Promise<TranscriptionResult>;

  /** Release resources. Safe to call multiple times. */
  destroy(): void;
}
