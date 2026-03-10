import { useEffect, useMemo, useRef, useState } from "react";
import {
  Download,
  Loader2,
  Pause,
  Play,
  SkipBack,
  SkipForward,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import type { DiarizationRecord } from "../api";
import {
  speakerSummariesFromRecord,
  transcriptEntriesFromRecord,
} from "../utils/diarizationTranscript";

interface DiarizationReviewWorkspaceProps {
  record: Pick<
    DiarizationRecord,
    | "id"
    | "duration_secs"
    | "speaker_count"
    | "corrected_speaker_count"
    | "audio_filename"
    | "segments"
    | "utterances"
    | "words"
    | "speaker_name_overrides"
    | "transcript"
    | "raw_transcript"
  > | null;
  audioUrl?: string | null;
  loading?: boolean;
  emptyMessage?: string;
}

type SpeakerAccent = {
  solid: string;
  soft: string;
  border: string;
};

const PLAYBACK_SPEEDS = [0.75, 1, 1.25, 1.5, 2];

function formatClockTime(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) {
    return "0:00";
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function formatDurationLabel(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "0s";
  }
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return `${minutes}m ${remainingSeconds}s`;
}

function speakerAccent(index: number): SpeakerAccent {
  const hue = (index * 67 + 154) % 360;
  return {
    solid: `hsl(${hue} 72% 52%)`,
    soft: `hsla(${hue} 72% 52% / 0.16)`,
    border: `hsla(${hue} 72% 52% / 0.4)`,
  };
}

function isEntryActive(
  currentTime: number,
  start: number,
  end: number,
  epsilon = 0.08,
): boolean {
  return currentTime >= Math.max(0, start - epsilon) && currentTime < end + epsilon;
}

export function DiarizationReviewWorkspace({
  record,
  audioUrl = null,
  loading = false,
  emptyMessage = "No diarization transcript is available yet.",
}: DiarizationReviewWorkspaceProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [audioError, setAudioError] = useState<string | null>(null);

  const transcriptEntries = useMemo(
    () => (record ? transcriptEntriesFromRecord(record) : []),
    [record],
  );
  const speakerSummaries = useMemo(
    () => (record ? speakerSummariesFromRecord(record) : []),
    [record],
  );

  const viewerDuration = useMemo(() => {
    const transcriptDuration = transcriptEntries.reduce(
      (max, entry) => Math.max(max, entry.end),
      0,
    );
    const recordDuration =
      record?.duration_secs && record.duration_secs > 0 ? record.duration_secs : 0;
    return Math.max(duration, recordDuration, transcriptDuration, 0);
  }, [duration, record?.duration_secs, transcriptEntries]);

  const totalTalkTime = useMemo(
    () =>
      speakerSummaries.reduce(
        (sum, summary) => sum + Math.max(summary.totalDuration, 0),
        0,
      ),
    [speakerSummaries],
  );
  const activeEntryIndex = transcriptEntries.findIndex((entry) =>
    isEntryActive(currentTime, entry.start, entry.end),
  );
  const activeSpeaker =
    activeEntryIndex >= 0 ? transcriptEntries[activeEntryIndex]?.speaker ?? null : null;

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
  }, [audioUrl, record?.id]);

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
      setAudioError("Unable to start playback for this diarization audio.");
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
    <div className="space-y-4">
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
        onRateChange={(event) => setPlaybackRate(event.currentTarget.playbackRate)}
        onError={() =>
          setAudioError("Unable to load audio for this diarization record.")
        }
        className="hidden"
      />

      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
        <CardHeader className="pb-4">
          <CardTitle className="text-sm text-[var(--text-primary)]">
            Review Playback
          </CardTitle>
          <CardDescription className="text-[var(--text-muted)]">
            Click the timeline or any transcript turn to jump to that speaker moment.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3">
              <div className="text-[11px] uppercase tracking-wider text-[var(--text-subtle)]">
                Current speaker
              </div>
              <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                {activeSpeaker ?? "Paused"}
              </div>
            </div>
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3">
              <div className="text-[11px] uppercase tracking-wider text-[var(--text-subtle)]">
                Position
              </div>
              <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                {formatClockTime(currentTime)} / {formatClockTime(viewerDuration)}
              </div>
            </div>
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3">
              <div className="text-[11px] uppercase tracking-wider text-[var(--text-subtle)]">
                Corrected speakers
              </div>
              <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                {record.corrected_speaker_count ?? record.speaker_count}
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              type="button"
              size="sm"
              className="h-9 gap-1.5"
              onClick={() => void togglePlayback()}
              disabled={!audioUrl}
            >
              {isPlaying ? (
                <Pause className="h-3.5 w-3.5" />
              ) : (
                <Play className="h-3.5 w-3.5" />
              )}
              {isPlaying ? "Pause" : "Play"}
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 border-[var(--border-muted)] bg-[var(--bg-surface-0)]"
              onClick={() => skip(-10)}
              disabled={!audioUrl}
            >
              <SkipBack className="mr-1.5 h-3.5 w-3.5" />
              10s
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 border-[var(--border-muted)] bg-[var(--bg-surface-0)]"
              onClick={() => skip(10)}
              disabled={!audioUrl}
            >
              <SkipForward className="mr-1.5 h-3.5 w-3.5" />
              10s
            </Button>
            <div className="ml-auto flex flex-wrap items-center gap-1.5">
              {PLAYBACK_SPEEDS.map((rate) => (
                <Button
                  key={rate}
                  type="button"
                  variant={playbackRate === rate ? "secondary" : "outline"}
                  size="sm"
                  className="h-8 border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-2 text-[11px]"
                  onClick={() => updatePlaybackRate(rate)}
                >
                  {rate.toFixed(rate % 1 === 0 ? 1 : 2)}x
                </Button>
              ))}
              {audioUrl ? (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-2"
                  asChild
                >
                  <a
                    href={audioUrl}
                    download={record.audio_filename ?? `${record.id}.wav`}
                  >
                    <Download className="mr-1.5 h-3.5 w-3.5" />
                    Audio
                  </a>
                </Button>
              ) : null}
            </div>
          </div>

          <Slider
            min={0}
            max={viewerDuration || 0}
            step={0.05}
            value={[Math.min(currentTime, viewerDuration || 0)]}
            onValueChange={(values) => seek(values[0] ?? 0)}
            disabled={!audioUrl || viewerDuration <= 0}
          />

          {audioError ? (
            <div className="rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-sm text-[var(--danger-text)]">
              {audioError}
            </div>
          ) : null}
        </CardContent>
      </Card>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr),280px]">
        <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
          <CardHeader className="pb-4">
            <CardTitle className="text-sm text-[var(--text-primary)]">
              Speaker Timeline
            </CardTitle>
            <CardDescription className="text-[var(--text-muted)]">
              Each block represents one diarized turn in the corrected transcript.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <ScrollArea className="w-full whitespace-nowrap" type="always">
              <div className="flex min-w-max gap-2 pb-2">
                {transcriptEntries.map((entry, index) => {
                  const accent = speakerAccent(index);
                  const segmentWidth = Math.max(
                    ((entry.end - entry.start) / Math.max(viewerDuration || 1, 1)) * 100,
                    8,
                  );
                  const active = index === activeEntryIndex;

                  return (
                    <button
                      key={`${entry.speaker}-${entry.start}-${entry.end}-${index}`}
                      type="button"
                      onClick={() => seek(entry.start)}
                      className="rounded-xl border px-3 py-2 text-left transition-transform hover:-translate-y-0.5"
                      style={{
                        width: `${segmentWidth}%`,
                        minWidth: `${Math.max(segmentWidth, 10)}%`,
                        backgroundColor: accent.soft,
                        borderColor: active ? accent.solid : accent.border,
                        boxShadow: active
                          ? `0 0 0 1px ${accent.solid} inset`
                          : undefined,
                      }}
                      title={`${entry.speaker} • ${formatClockTime(entry.start)} - ${formatClockTime(entry.end)}`}
                    >
                      <div className="truncate text-[11px] font-semibold text-[var(--text-primary)]">
                        {entry.speaker}
                      </div>
                      <div className="mt-1 text-[10px] text-[var(--text-muted)]">
                        {formatDurationLabel(entry.end - entry.start)}
                      </div>
                    </button>
                  );
                })}
              </div>
            </ScrollArea>

            <Separator className="bg-[var(--border-muted)]" />

            <ScrollArea className="h-[320px] rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)]">
              <div className="space-y-2 p-3">
                {transcriptEntries.map((entry, index) => {
                  const active = index === activeEntryIndex;
                  const accent = speakerAccent(index);

                  return (
                    <button
                      key={`${entry.speaker}-${entry.start}-${entry.end}-${index}`}
                      type="button"
                      onClick={() => seek(entry.start)}
                      className="w-full rounded-xl border p-3 text-left transition-colors"
                      data-active={active ? "true" : "false"}
                      style={{
                        backgroundColor: active ? accent.soft : "var(--bg-surface-1)",
                        borderColor: active ? accent.solid : "var(--border-muted)",
                        boxShadow: active ? `0 0 0 1px ${accent.solid} inset` : undefined,
                      }}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="min-w-0">
                          <div className="truncate text-xs font-semibold text-[var(--text-primary)]">
                            {entry.speaker}
                          </div>
                          <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                            {formatClockTime(entry.start)} - {formatClockTime(entry.end)}
                          </div>
                        </div>
                        {active ? (
                          <span
                            className="rounded-md px-2 py-1 text-[10px] font-semibold"
                            style={{
                              color: accent.solid,
                              backgroundColor: accent.soft,
                            }}
                          >
                            Live
                          </span>
                        ) : null}
                      </div>
                      <p className="mt-2 text-sm leading-relaxed text-[var(--text-secondary)]">
                        {entry.text}
                      </p>
                    </button>
                  );
                })}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>

        <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
          <CardHeader className="pb-4">
            <CardTitle className="text-sm text-[var(--text-primary)]">
              Talk Time
            </CardTitle>
            <CardDescription className="text-[var(--text-muted)]">
              Summary of corrected speaker activity across the recording.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {speakerSummaries.map((summary, index) => {
              const accent = speakerAccent(index);
              const share =
                totalTalkTime > 0 ? (summary.totalDuration / totalTalkTime) * 100 : 0;
              const isCurrentSpeaker = summary.displaySpeaker === activeSpeaker;

              return (
                <div
                  key={summary.displaySpeaker}
                  className="rounded-xl border p-3"
                  style={{
                    backgroundColor: isCurrentSpeaker
                      ? accent.soft
                      : "var(--bg-surface-0)",
                    borderColor: isCurrentSpeaker
                      ? accent.solid
                      : "var(--border-muted)",
                  }}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-semibold text-[var(--text-primary)]">
                        {summary.displaySpeaker}
                      </div>
                      <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                        {summary.utteranceCount} turns • {summary.wordCount} words
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-semibold text-[var(--text-primary)]">
                        {formatDurationLabel(summary.totalDuration)}
                      </div>
                      <div className="text-[11px] text-[var(--text-muted)]">
                        {share.toFixed(0)}%
                      </div>
                    </div>
                  </div>
                  <div className="mt-3 h-2 rounded-full bg-[var(--bg-surface-2)]">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.max(share, 6)}%`,
                        backgroundColor: accent.solid,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
