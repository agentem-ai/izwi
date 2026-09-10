import { decodePcmI16Base64 } from "@/features/voice/realtime/support";

/** Schedules PCM once, releasing each buffer when heard. History owns the recording. */
export class SpeechPcmPlayer {
  private readonly sources = new Set<AudioBufferSourceNode>();
  private nextTime = 0;
  private sequence = 0;
  private sampleRate = 0;
  private requestId: string | null = null;
  private wakeCapacity: (() => void) | null = null;
  private finished = false;
  private stopped = false;

  constructor(
    private readonly context: AudioContext,
    private readonly onDrained: () => void,
    private readonly maxBufferedSeconds = 30,
  ) {}

  /** Invoke during the Generate gesture, before making the network request. */
  unlock(): Promise<void> {
    return this.context.resume();
  }

  start(requestId: string, sampleRate: number, format: string): void {
    if (this.requestId !== null || format !== "pcm_i16" ||
        !Number.isInteger(sampleRate) || sampleRate < 8000 || sampleRate > 192000) {
      throw new Error("Invalid speech stream audio format.");
    }
    this.requestId = requestId;
    this.sampleRate = sampleRate;
  }

  async push(event: { requestId: string; sequence: number; audioBase64: string; sampleCount: number; sampleRate?: number }): Promise<void> {
    if (this.stopped || this.finished || this.requestId !== event.requestId ||
        event.sequence !== this.sequence || !this.sampleRate) {
      throw new Error("Speech stream chunks arrived out of order.");
    }
    if (event.sampleRate !== undefined && event.sampleRate !== this.sampleRate) {
      throw new Error("Speech stream sample rate does not match its audio format.");
    }
    const duration = event.sampleCount / this.sampleRate;
    if (!Number.isSafeInteger(event.sampleCount) || event.sampleCount <= 0 ||
        duration > this.maxBufferedSeconds) {
      throw new Error("Speech playback buffer is full. Generation was stopped.");
    }
    // Check encoded size before allocating either the decoded PCM or Web Audio buffer.
    if (event.audioBase64.length !== 4 * Math.ceil(event.sampleCount * 2 / 3) ||
        atob(event.audioBase64).length !== event.sampleCount * 2) {
      throw new Error("Speech stream sample count does not match its PCM payload.");
    }
    // Backpressure the SSE reader while audio is queued; do not retain another PCM copy.
    while (Math.max(0, this.nextTime - this.context.currentTime) + duration > this.maxBufferedSeconds ||
        this.sources.size >= 1024) {
      await new Promise<void>((resolve) => { this.wakeCapacity = resolve; });
      if (this.stopped) throw new Error("Speech playback stopped.");
    }
    const samples = decodePcmI16Base64(event.audioBase64);
    if (samples.length !== event.sampleCount) {
      throw new Error("Speech stream sample count does not match its PCM payload.");
    }
    const buffer = this.context.createBuffer(1, samples.length, this.sampleRate);
    buffer.copyToChannel(new Float32Array(samples), 0);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    const scheduledAt = Math.max(this.context.currentTime + 0.02, this.nextTime);
    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      source.disconnect();
      this.wakeCapacity?.();
      this.wakeCapacity = null;
      if (this.finished && !this.sources.size && !this.stopped) this.onDrained();
    };
    source.start(scheduledAt);
    this.nextTime = scheduledAt + duration;
    this.sequence += 1;
  }

  finish(): void {
    if (this.finished || this.stopped) return;
    this.finished = true;
    if (!this.sources.size) this.onDrained();
  }

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    this.wakeCapacity?.();
    this.wakeCapacity = null;
    for (const source of this.sources) {
      source.onended = null;
      source.stop();
      source.disconnect();
    }
    this.sources.clear();
    void this.context.close().catch(() => {});
  }
}
