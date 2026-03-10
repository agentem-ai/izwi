import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Pause, Play, SkipBack, SkipForward } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import type { TranscriptionRecord } from "@/shared/api/audio";
import {
  formatAudioDuration,
  formatClockTime,
} from "@/features/transcription/playground/support";
import {
  formattedTranscriptFromRecord,
  transcriptEntriesFromRecord,
} from "@/features/transcription/utils/transcriptionTranscript";

interface TranscriptionReviewWorkspaceProps {
  record: Pick<
    TranscriptionRecord,
    | "aligner_model_id"
    | "audio_filename"
    | "duration_secs"
    | "language"
    | "model_id"
    | "raw_transcription"
    | "segments"
    | "transcription"
    | "words"
  > | null;
  audioUrl?: string | null;
  loading?: boolean;
  emptyMessage?: string;
}

const PLAYBACK_SPEEDS = [0.75, 1, 1.25, 1.5, 2];

function isEntryActive(
  currentTime: number,
  start: number,
  end: number,
  epsilon = 0.08,
): boolean {
  return (
    currentTime >= Math.max(0, start - epsilon) && currentTime < end + epsilon
  );
}

export function TranscriptionReviewWorkspace({
  record,
  audioUrl = null,
  loading = false,
  emptyMessage = "No transcript is available yet.",
}: TranscriptionReviewWorkspaceProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [audioError, setAudioError] = useState<string | null>(null);

  const transcriptEntries = useMemo(
    () => transcriptEntriesFromRecord(record),
    [record],
  );
  const transcriptText = useMemo(
    () => formattedTranscriptFromRecord(record),
    [record],
  );

  const viewerDuration = useMemo(() => {
    const transcriptDuration = transcriptEntries.reduce(
      (max, entry) => Math.max(max, entry.end),
      0,
    );
    const recordDuration =
      record?.duration_secs && record.duration_secs > 0
        ? record.duration_secs
        : 0;
    return Math.max(duration, recordDuration, transcriptDuration, 0);
  }, [duration, record, transcriptEntries]);

  const activeEntryIndex = transcriptEntries.findIndex((entry) =>
    isEntryActive(currentTime, entry.start, entry.end),
  );
  const activeEntry = activeEntryIndex >= 0 ? transcriptEntries[activeEntryIndex] : null;
  const edited = Boolean(
    record &&
      record.raw_transcription.trim() &&
      record.transcription.trim() !== record.raw_transcription.trim(),
  );

  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
      audio.playbackRate = 1;
    }
    setCurrentTime(0);
    setDuration(0);
    setIsPlaying(false);
    setPlaybackRate(1);
    setAudioError(null);
  }, [audioUrl, record?.audio_filename, record?.transcription]);

  async function togglePlayback(): Promise<void> {
    const audio = audioRef.current;
    if (!audio || !audioUrl) {
      return;
    }

    try {
      if (audio.paused) {
        await audio.play();
      } else {
        audio.pause();
      }
    } catch {
      setAudioError("Unable to start playback for this transcription audio.");
    }
  }

  function seek(nextTime: number): void {
    const audio = audioRef.current;
    const clamped = Math.max(0, Math.min(nextTime, viewerDuration || 0));
    if (audio) {
      audio.currentTime = clamped;
    }
    setCurrentTime(clamped);
  }

  function skip(deltaSeconds: number): void {
    seek(currentTime + deltaSeconds);
  }

  function updatePlaybackRate(nextRate: number): void {
    const audio = audioRef.current;
    if (audio) {
      audio.playbackRate = nextRate;
    }
    setPlaybackRate(nextRate);
  }

  if (loading) {
    return (
      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
        <CardContent className="flex min-h-[320px] items-center justify-center gap-2 py-12 text-sm text-[var(--text-muted)]">
          <Loader2 className="h-4 w-4 animate-spin" />
          Loading transcript...
        </CardContent>
      </Card>
    );
  }

  if (!record || transcriptEntries.length === 0) {
    return (
      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
        <CardContent className="py-12 text-center text-sm text-[var(--text-muted)]">
          {emptyMessage}
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="relative flex h-full flex-col">
      <audio
        ref={audioRef}
        src={audioUrl ?? undefined}
        preload="metadata"
        onLoadedMetadata={(event) => {
          const nextDuration = Number.isFinite(event.currentTarget.duration)
            ? event.currentTarget.duration
            : 0;
          setDuration(nextDuration);
          setAudioError(null);
        }}
        onTimeUpdate={(event) => {
          setCurrentTime(event.currentTarget.currentTime);
        }}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onRateChange={(event) =>
          setPlaybackRate(event.currentTarget.playbackRate)
        }
        onError={() =>
          setAudioError("Unable to load audio for this transcription record.")
        }
        className="hidden"
      />

      <div className="grid gap-6 pb-20 xl:grid-cols-[minmax(0,1fr),248px]">
        <div className="space-y-5">
          <div>
            <h3 className="mb-3 text-[13px] font-semibold tracking-wide text-[var(--text-primary)]">
              Transcript
            </h3>
            <div className="space-y-2.5">
              {transcriptEntries.map((entry, index) => {
                const active = index === activeEntryIndex;

                return (
                  <button
                    key={`${entry.start}-${entry.end}-${index}`}
                    type="button"
                    onClick={() => seek(entry.start)}
                    className="w-full rounded-lg border px-3.5 py-3 text-left transition-colors"
                    style={{
                      backgroundColor: active
                        ? "var(--accent-soft)"
                        : "transparent",
                      borderColor: active
                        ? "var(--text-primary)"
                        : "transparent",
                      boxShadow: active
                        ? "0 0 0 1px var(--text-primary) inset"
                        : undefined,
                    }}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0">
                        <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-subtle)]">
                          {formatClockTime(entry.start)} to {formatClockTime(entry.end)}
                        </div>
                      </div>
                      {active ? (
                        <span className="rounded-full bg-[var(--accent-soft)] px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.14em] text-[var(--text-primary)]">
                          Live
                        </span>
                      ) : null}
                    </div>
                    <p className="mt-1.5 text-[15px] leading-7 text-[var(--text-secondary)]">
                      {entry.text}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        <div className="space-y-5">
          <div>
            <h3 className="mb-3 text-[13px] font-semibold tracking-wide text-[var(--text-primary)]">
              Snapshot
            </h3>
            <div className="space-y-2.5">
              <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-3">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-subtle)]">
                  Transcript
                </div>
                <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                  {transcriptEntries.length} segments
                </div>
                <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                  {record.words.length} aligned words
                </div>
              </div>
              <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-3">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-subtle)]">
                  Quality
                </div>
                <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                  {record.aligner_model_id ? "Timed transcript" : "Plain transcript"}
                </div>
                <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                  {edited ? "Corrections saved" : "Raw transcript"}
                </div>
              </div>
              <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-3">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-subtle)]">
                  Session
                </div>
                <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                  {formatAudioDuration(record.duration_secs)}
                </div>
                <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                  {record.language || "Unknown language"}
                </div>
              </div>
              <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-3">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-subtle)]">
                  Models
                </div>
                <div className="mt-1 text-[12px] font-medium text-[var(--text-primary)]">
                  {record.model_id || "Unknown ASR model"}
                </div>
                <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                  {record.aligner_model_id || "No aligner metadata"}
                </div>
              </div>
              <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-3">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-subtle)]">
                  Current Cue
                </div>
                <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                  {activeEntry ? formatClockTime(activeEntry.start) : "Idle"}
                </div>
                <div className="mt-1 line-clamp-3 text-[11px] text-[var(--text-muted)]">
                  {activeEntry?.text || transcriptText}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="sticky bottom-0 -mx-4 -mb-4 mt-auto border-t border-[var(--border-muted)] bg-[var(--bg-surface-0)]/95 p-3 backdrop-blur sm:-mx-5 sm:-mb-5 sm:px-5 sm:py-3">
        <div className="flex flex-col gap-2.5">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="icon"
                className="h-8 w-8 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-1)] hover:bg-[var(--bg-surface-2)]"
                onClick={() => skip(-10)}
                disabled={!audioUrl}
                title="Rewind 10 seconds"
              >
                <SkipBack className="h-4 w-4" />
              </Button>
              <Button
                type="button"
                size="icon"
                className="h-9 w-9 rounded-full bg-[var(--text-primary)] text-[var(--bg-surface-0)] shadow-md hover:bg-[var(--text-secondary)]"
                onClick={() => void togglePlayback()}
                disabled={!audioUrl}
              >
                {isPlaying ? (
                  <Pause className="h-4 w-4" fill="currentColor" />
                ) : (
                  <Play className="ml-0.5 h-4 w-4" fill="currentColor" />
                )}
              </Button>
              <Button
                type="button"
                variant="outline"
                size="icon"
                className="h-8 w-8 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-1)] hover:bg-[var(--bg-surface-2)]"
                onClick={() => skip(10)}
                disabled={!audioUrl}
                title="Forward 10 seconds"
              >
                <SkipForward className="h-4 w-4" />
              </Button>
            </div>

            <div className="font-mono text-[13px] font-medium tabular-nums tracking-tight text-[var(--text-primary)]">
              {formatClockTime(currentTime)}{" "}
              <span className="font-normal text-[var(--text-muted)]">
                / {formatClockTime(viewerDuration)}
              </span>
            </div>

            <div className="group relative mx-1.5 flex h-9 flex-1 items-center">
              <div className="pointer-events-none absolute inset-x-0 top-1/2 h-7 -translate-y-1/2 overflow-hidden rounded bg-[var(--bg-surface-2)]/50">
                {viewerDuration > 0
                  ? transcriptEntries.map((entry, index) => {
                      const left = (entry.start / viewerDuration) * 100;
                      const width = ((entry.end - entry.start) / viewerDuration) * 100;
                      const active = index === activeEntryIndex;

                      return (
                        <div
                          key={`wave-${index}`}
                          className="absolute top-0 bottom-0 border-r border-[var(--bg-surface-0)] transition-all duration-200"
                          style={{
                            left: `${left}%`,
                            width: `${Math.max(width, 0.5)}%`,
                            backgroundColor: active
                              ? "var(--text-primary)"
                              : "var(--accent-soft)",
                            opacity: active ? 1 : 0.8,
                          }}
                        />
                      );
                    })
                  : null}
              </div>

              <input
                type="range"
                min={0}
                max={viewerDuration || 0}
                step={0.05}
                value={Math.min(currentTime, viewerDuration || 0)}
                onChange={(event) => seek(Number(event.target.value))}
                aria-label="Seek audio timeline"
                disabled={!audioUrl || viewerDuration <= 0}
                className="relative z-10 h-7 w-full appearance-none bg-transparent accent-[var(--text-primary)] focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-45 [&::-moz-range-progress]:bg-transparent [&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:w-3.5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-none [&::-moz-range-thumb]:bg-[var(--text-primary)] [&::-moz-range-thumb]:shadow-md [&::-moz-range-track]:h-1.5 [&::-moz-range-track]:rounded-full [&::-moz-range-track]:bg-transparent [&::-webkit-slider-runnable-track]:h-1.5 [&::-webkit-slider-runnable-track]:rounded-full [&::-webkit-slider-runnable-track]:bg-transparent [&::-webkit-slider-thumb]:-mt-1 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-none [&::-webkit-slider-thumb]:bg-[var(--text-primary)] [&::-webkit-slider-thumb]:shadow-md"
              />
            </div>

            <select
              className="h-7 rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 text-[11px] font-medium text-[var(--text-primary)] outline-none"
              value={playbackRate}
              onChange={(event) => updatePlaybackRate(Number(event.target.value))}
            >
              {PLAYBACK_SPEEDS.map((rate) => (
                <option key={rate} value={rate}>
                  {rate}x
                </option>
              ))}
            </select>
          </div>

          {audioError ? (
            <div className="rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
              {audioError}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
