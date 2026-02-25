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
}

export interface TranscriptionResult {
  text: string;
  chunks: number;
  file_size_bytes: number;
  processing_time_s: number;
  model: string;
  language: string;
  prompt: string;
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

export type StatusValue = 'recording' | 'transcribing' | 'done' | 'error' | 'no_speech';

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

  /** Called after each chunk is uploaded to the server. */
  onChunkUploaded: ((info: ChunkUploadInfo) => void) | null;
  /** Called when the client status changes. */
  onStatusChange: ((status: StatusValue) => void) | null;

  /** Whether the client is currently recording. */
  readonly recording: boolean;

  constructor(options?: WhisperClientOptions);

  /** Check server health. */
  getHealth(): Promise<HealthResponse>;

  /** Start recording from the microphone with VAD-filtered chunk streaming. */
  startRecording(): Promise<void>;

  /** Stop recording, flush remaining audio, and transcribe the session. */
  stopRecording(): Promise<TranscriptionResult>;

  /** Single-shot transcription of an audio blob/file. */
  transcribe(audioBlob: Blob, filename?: string): Promise<TranscriptionResult>;

  /** Release resources. Safe to call multiple times. */
  destroy(): void;
}
