import { describe, expect, it, vi } from "vitest";
import { SpeechPcmPlayer } from "./pcmPlayer";

function audioFixture(sampleRate = 8000) {
  const sources: { buffer: AudioBuffer | null; start: ReturnType<typeof vi.fn>; stop: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn>; onended: (() => void) | null }[] = [];
  const samples: Float32Array[] = [];
  const context = {
    currentTime: 0,
    destination: {},
    resume: vi.fn().mockResolvedValue(undefined),
    close: vi.fn().mockResolvedValue(undefined),
    createBuffer: vi.fn((_channels: number, count: number, rate: number) => ({
      duration: count / rate,
      copyToChannel: (pcm: Float32Array) => samples.push(pcm),
    })),
    createBufferSource: vi.fn(() => {
      const source = { buffer: null, start: vi.fn(), stop: vi.fn(), connect: vi.fn(), disconnect: vi.fn(), onended: null };
      sources.push(source);
      return source;
    }),
  };
  const drained = vi.fn();
  const player = new SpeechPcmPlayer(context as unknown as AudioContext, drained, 1);
  player.start("request", sampleRate, "pcm_i16");
  const chunk = (sequence: number, count = 4000) => ({
    requestId: "request", sequence, sampleCount: count,
    audioBase64: btoa("\x00\x40".repeat(count)),
  });
  return { player, context, sources, samples, drained, chunk };
}

describe("progressive speech PCM playback", () => {
  it("schedules real PCM before finalization, maintains sample rate and drains the final tail once", async () => {
    const { player, context, sources, samples, drained, chunk } = audioFixture();
    await player.unlock();
    expect(context.resume).toHaveBeenCalledOnce();
    await player.push(chunk(0, 2000));
    expect(sources[0].start).toHaveBeenCalledWith(0.02);
    expect(samples[0][0]).toBe(0.5);
    expect(context.createBuffer).toHaveBeenCalledWith(1, 2000, 8000);
    await player.push(chunk(1, 2000));
    expect(sources[1].start).toHaveBeenCalledWith(0.27);
    player.finish();
    player.finish();
    sources[0].onended?.();
    expect(drained).not.toHaveBeenCalled();
    sources[1].onended?.();
    expect(drained).toHaveBeenCalledOnce();
  });

  it("plays Fish PCM at its source rate with matching sample values and duration", async () => {
    const { player, context, sources, samples, chunk } = audioFixture(44100);
    await player.push({ ...chunk(0, 11025), sampleRate: 44100 });
    await player.push({ ...chunk(1, 11025), sampleRate: 44100 });
    expect(context.createBuffer).toHaveBeenNthCalledWith(1, 1, 11025, 44100);
    expect(sources[0].buffer?.duration).toBe(0.25);
    expect(sources[1].start).toHaveBeenCalledWith(0.27);
    expect(samples[0]).toEqual(new Float32Array(11025).fill(0.5));
  });

  it("rejects a mismatched source rate before queuing or decoding PCM", async () => {
    const { player, context, chunk } = audioFixture(24000);
    await expect(player.push({ ...chunk(0), sampleRate: 44100 }))
      .rejects.toThrow("sample rate");
    expect(context.createBuffer).not.toHaveBeenCalled();
    await player.push({ ...chunk(0), sampleRate: 24000 });
    await expect(player.push({ ...chunk(1), sampleRate: 44100 }))
      .rejects.toThrow("sample rate");
    expect(context.createBuffer).toHaveBeenCalledOnce();
  });

  it("backpressures a fast producer until playback frees space, without allocating pending PCM", async () => {
    const { player, context, sources, chunk } = audioFixture();
    await player.push(chunk(0));
    let accepted = false;
    const pending = player.push(chunk(1)).then(() => { accepted = true; });
    await Promise.resolve();
    expect(accepted).toBe(false);
    expect(context.createBuffer).toHaveBeenCalledTimes(1);
    context.currentTime = 0.52;
    sources[0].onended?.();
    await pending;
    expect(accepted).toBe(true);
    expect(context.createBuffer).toHaveBeenCalledTimes(2);
  });

  it("stops scheduled audio and releases a blocked producer on cancellation", async () => {
    const { player, context, sources, chunk } = audioFixture();
    await player.push(chunk(0));
    const pending = player.push(chunk(1));
    player.stop();
    player.stop();
    await expect(pending).rejects.toThrow("stopped");
    expect(sources[0].stop).toHaveBeenCalledOnce();
    expect(context.close).toHaveBeenCalledOnce();
  });

  it("rejects wrong ordering, request identity, sample counts and oversized chunks before allocation", async () => {
    const { player, context, chunk } = audioFixture();
    await expect(player.push(chunk(1))).rejects.toThrow("order");
    await expect(player.push({ ...chunk(0), requestId: "foreign" })).rejects.toThrow("order");
    await expect(player.push({ ...chunk(0), sampleCount: 3 })).rejects.toThrow("sample count");
    await expect(player.push(chunk(0, 8001))).rejects.toThrow("buffer is full");
    expect(context.createBuffer).not.toHaveBeenCalled();
  });
});
