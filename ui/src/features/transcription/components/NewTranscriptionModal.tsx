import { useCallback, useRef, useState } from "react";
import { AlertTriangle, Loader2, Mic, Square, Upload } from "lucide-react";

import { api, type TranscriptionRecord } from "@/api";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { StatusBadge } from "@/components/ui/status-badge";
import {
  LANGUAGE_OPTIONS,
  transcodeToWav,
} from "@/features/transcription/playground/support";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface NewTranscriptionModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedModel: string | null;
  selectedModelReady: boolean;
  timestampAlignerModelId: string | null;
  timestampAlignerReady: boolean;
  onModelRequired: () => void;
  onTimestampAlignerRequired: () => void;
  onCreated: (record: TranscriptionRecord) => void;
}

interface SubmitAudioOptions {
  filename?: string;
  transcode?: boolean;
}

export function NewTranscriptionModal({
  isOpen,
  onClose,
  selectedModel,
  selectedModelReady,
  timestampAlignerModelId,
  timestampAlignerReady,
  onModelRequired,
  onTimestampAlignerRequired,
  onCreated,
}: NewTranscriptionModalProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const [selectedLanguage, setSelectedLanguage] = useState("English");
  const [includeTimestamps, setIncludeTimestamps] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const requireReadyModel = useCallback(() => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      setError("Select and load an ASR model before creating a transcription.");
      return false;
    }
    return true;
  }, [onModelRequired, selectedModel, selectedModelReady]);

  const requireTimestampAligner = useCallback(() => {
    if (!includeTimestamps) {
      return true;
    }
    if (timestampAlignerModelId && timestampAlignerReady) {
      return true;
    }
    onTimestampAlignerRequired();
    setError("Load the timestamp aligner model to include timestamps.");
    return false;
  }, [
    includeTimestamps,
    onTimestampAlignerRequired,
    timestampAlignerModelId,
    timestampAlignerReady,
  ]);

  const submitAudio = useCallback(
    async (audioBlob: Blob, options: SubmitAudioOptions = {}) => {
      if (!requireReadyModel() || !requireTimestampAligner()) {
        return;
      }

      setIsSubmitting(true);
      setError(null);

      try {
        const uploadBlob =
          options.transcode === false
            ? audioBlob
            : await transcodeToWav(audioBlob, 16000);
        const record = await api.createTranscriptionRecord({
          audio_file: uploadBlob,
          audio_filename: options.filename || "audio.wav",
          model_id: selectedModel || undefined,
          aligner_model_id: includeTimestamps
            ? timestampAlignerModelId || undefined
            : undefined,
          language: selectedLanguage,
          include_timestamps: includeTimestamps,
        });
        onCreated(record);
        onClose();
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to create transcription.",
        );
      } finally {
        setIsSubmitting(false);
      }
    },
    [
      includeTimestamps,
      onClose,
      onCreated,
      requireReadyModel,
      requireTimestampAligner,
      selectedLanguage,
      selectedModel,
      timestampAlignerModelId,
    ],
  );

  const handleFileUpload = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) {
        return;
      }

      await submitAudio(file, {
        filename: file.name,
        transcode: false,
      });
      event.target.value = "";
    },
    [submitAudio],
  );

  const startRecording = useCallback(async () => {
    if (!requireReadyModel() || !requireTimestampAligner()) {
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeCandidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
      ];
      let mediaRecorder: MediaRecorder | null = null;
      for (const mimeType of mimeCandidates) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          mediaRecorder = new MediaRecorder(stream, { mimeType });
          break;
        }
      }
      if (!mediaRecorder) {
        mediaRecorder = new MediaRecorder(stream);
      }

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        setIsRecording(false);
        const audioBlob = new Blob(audioChunksRef.current, {
          type: mediaRecorder?.mimeType || "audio/webm",
        });
        await submitAudio(audioBlob, {
          filename: "recording.webm",
          transcode: true,
        });
      };

      mediaRecorder.start(1000);
      setIsRecording(true);
      setError(null);
    } catch {
      setError("Could not access the microphone. Please grant permission.");
    }
  }, [requireReadyModel, requireTimestampAligner, submitAudio]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
    }
  }, [isRecording]);

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-3xl border-[var(--border-strong)] bg-[var(--bg-surface-0)] p-0">
        <div className="border-b border-[var(--border-muted)] px-6 py-5">
          <DialogTitle className="text-xl font-semibold text-[var(--text-primary)]">
            New transcript
          </DialogTitle>
          <DialogDescription className="mt-1 text-sm text-[var(--text-muted)]">
            Choose language and timing options, then upload audio or capture a recording.
          </DialogDescription>
        </div>

        <div className="space-y-5 px-6 py-5">
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr),auto]">
            <div className="space-y-3">
              <div>
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                  Language
                </div>
                <Select
                  value={selectedLanguage}
                  onValueChange={setSelectedLanguage}
                  disabled={isSubmitting}
                >
                  <SelectTrigger className="h-10 w-full text-sm">
                    <SelectValue placeholder="Language" />
                  </SelectTrigger>
                  <SelectContent>
                    {LANGUAGE_OPTIONS.map((language) => (
                      <SelectItem key={language} value={language}>
                        {language}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <label className="flex items-center justify-between rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3 text-sm">
                <div>
                  <div className="font-medium text-[var(--text-primary)]">
                    Include timestamps
                  </div>
                  <div className="mt-1 text-xs text-[var(--text-muted)]">
                    Add word and segment timing when the aligner model is ready.
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={includeTimestamps}
                  onChange={(event) => setIncludeTimestamps(event.target.checked)}
                  className="app-checkbox h-4 w-4"
                  disabled={isSubmitting}
                />
              </label>
            </div>

            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3">
              <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                Active model
              </div>
              <div className="mt-2 text-sm font-medium text-[var(--text-primary)]">
                {selectedModel || "No model selected"}
              </div>
              <div className="mt-2">
                <StatusBadge tone={selectedModelReady ? "success" : "warning"}>
                  {selectedModelReady ? "Ready" : "Load model"}
                </StatusBadge>
              </div>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <button
              type="button"
              onClick={() => {
                if (isRecording) {
                  stopRecording();
                } else {
                  void startRecording();
                }
              }}
              disabled={isSubmitting}
              className="flex min-h-[12rem] flex-col items-center justify-center rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-5 text-center transition-colors hover:bg-[var(--bg-surface-2)] disabled:cursor-not-allowed disabled:opacity-60"
            >
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-0)]">
                {isRecording ? (
                  <Square className="h-7 w-7 fill-current text-[var(--danger-text)]" />
                ) : (
                  <Mic className="h-7 w-7 text-[var(--text-primary)]" />
                )}
              </div>
              <div className="text-base font-semibold text-[var(--text-primary)]">
                {isRecording ? "Stop recording" : "Record audio"}
              </div>
              <div className="mt-2 text-sm text-[var(--text-muted)]">
                {isRecording
                  ? "Finish recording and create a transcription job."
                  : "Capture audio from your microphone."}
              </div>
            </button>

            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isSubmitting}
              className="flex min-h-[12rem] flex-col items-center justify-center rounded-2xl border border-dashed border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-5 text-center transition-colors hover:bg-[var(--bg-surface-2)] disabled:cursor-not-allowed disabled:opacity-60"
            >
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-0)]">
                <Upload className="h-7 w-7 text-[var(--text-primary)]" />
              </div>
              <div className="text-base font-semibold text-[var(--text-primary)]">
                Upload audio
              </div>
              <div className="mt-2 text-sm text-[var(--text-muted)]">
                WAV, MP3, M4A, AAC
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={(event) => void handleFileUpload(event)}
              />
            </button>
          </div>

          {error ? (
            <div className="rounded-xl border border-[var(--danger-border)] bg-[var(--danger-bg)] px-4 py-3 text-sm text-[var(--danger-text)]">
              <div className="flex items-start gap-2">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                <p>{error}</p>
              </div>
            </div>
          ) : null}
        </div>

        <div className="flex items-center justify-between border-t border-[var(--border-muted)] px-6 py-4">
          <p className="text-xs text-[var(--text-muted)]">
            The record will appear in the history table as soon as the upload is accepted.
          </p>
          <div className="flex items-center gap-2">
            <Button type="button" variant="ghost" onClick={onClose} disabled={isSubmitting}>
              Cancel
            </Button>
            {isSubmitting ? (
              <div className="inline-flex items-center gap-2 text-sm text-[var(--text-muted)]">
                <Loader2 className="h-4 w-4 animate-spin" />
                Creating job...
              </div>
            ) : null}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
