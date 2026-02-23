import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  Mic,
  Square,
  Play,
  Check,
  X,
  BookmarkPlus,
  Loader2,
  RefreshCw,
  Library,
} from "lucide-react";
import clsx from "clsx";
import { api, type SavedVoiceSummary } from "../api";
import { blobToBase64Payload } from "../utils/audioBase64";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

interface VoiceCloneProps {
  onVoiceCloneReady: (audioBase64: string, transcript: string) => void;
  onClear: () => void;
}

function downmixToMono(audioBuffer: AudioBuffer): Float32Array {
  const frameCount = audioBuffer.length;
  const channelCount = audioBuffer.numberOfChannels;
  const mono = new Float32Array(frameCount);

  if (channelCount === 1) {
    mono.set(audioBuffer.getChannelData(0));
    return mono;
  }

  for (let channel = 0; channel < channelCount; channel += 1) {
    const data = audioBuffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i += 1) {
      mono[i] += data[i] / channelCount;
    }
  }

  return mono;
}

function encodeWavPcm16(
  samples: Float32Array,
  sampleRate: number,
): ArrayBuffer {
  const bytesPerSample = 2;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    const intSample =
      sample < 0 ? Math.round(sample * 0x8000) : Math.round(sample * 0x7fff);
    view.setInt16(offset, intSample, true);
    offset += bytesPerSample;
  }

  return buffer;
}

async function normalizeToWavBlob(inputBlob: Blob): Promise<Blob> {
  const arrayBuffer = await inputBlob.arrayBuffer();
  const audioContext = new AudioContext();

  try {
    const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const monoSamples = downmixToMono(decoded);
    const wavBuffer = encodeWavPcm16(monoSamples, decoded.sampleRate);
    return new Blob([wavBuffer], { type: "audio/wav" });
  } finally {
    void audioContext.close();
  }
}

export function VoiceClone({ onVoiceCloneReady, onClear }: VoiceCloneProps) {
  const [mode, setMode] = useState<"upload" | "record" | "saved" | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isConfirmed, setIsConfirmed] = useState(false);
  const [savedVoices, setSavedVoices] = useState<SavedVoiceSummary[]>([]);
  const [savedVoicesLoading, setSavedVoicesLoading] = useState(false);
  const [savedVoicesError, setSavedVoicesError] = useState<string | null>(null);
  const [selectedSavedVoiceId, setSelectedSavedVoiceId] = useState("");
  const [isApplyingSavedVoice, setIsApplyingSavedVoice] = useState(false);
  const [saveVoiceName, setSaveVoiceName] = useState("");
  const [isSavingVoice, setIsSavingVoice] = useState(false);
  const [saveVoiceStatus, setSaveVoiceStatus] = useState<{
    tone: "success" | "error";
    message: string;
  } | null>(null);
  const [isCreateVoiceModalOpen, setIsCreateVoiceModalOpen] = useState(false);
  const [createVoiceName, setCreateVoiceName] = useState("");
  const [createVoiceTranscript, setCreateVoiceTranscript] = useState("");
  const [createVoiceAudioBlob, setCreateVoiceAudioBlob] = useState<Blob | null>(null);
  const [createVoiceAudioUrl, setCreateVoiceAudioUrl] = useState<string | null>(null);
  const [createVoiceInputMode, setCreateVoiceInputMode] = useState<
    "upload" | "record" | null
  >(null);
  const [createVoiceIsRecording, setCreateVoiceIsRecording] = useState(false);
  const [createVoiceSaving, setCreateVoiceSaving] = useState(false);
  const [createVoiceError, setCreateVoiceError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const createVoiceFileInputRef = useRef<HTMLInputElement>(null);
  const createVoiceRecorderRef = useRef<MediaRecorder | null>(null);
  const createVoiceChunksRef = useRef<Blob[]>([]);
  const isConfirmingRef = useRef(false);

  const loadSavedVoices = useCallback(async () => {
    setSavedVoicesLoading(true);
    setSavedVoicesError(null);
    try {
      const records = await api.listSavedVoices();
      setSavedVoices(records);
    } catch (err) {
      setSavedVoicesError(
        err instanceof Error ? err.message : "Failed to load saved voices.",
      );
    } finally {
      setSavedVoicesLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSavedVoices();
  }, [loadSavedVoices]);

  useEffect(() => {
    return () => {
      if (createVoiceAudioUrl) {
        URL.revokeObjectURL(createVoiceAudioUrl);
      }
    };
  }, [createVoiceAudioUrl]);

  const prepareAudioBlob = useCallback(
    async (inputBlob: Blob, inputMode: "upload" | "record" | "saved") => {
      try {
        const wavBlob = await normalizeToWavBlob(inputBlob);
        if (audioUrl) {
          URL.revokeObjectURL(audioUrl);
        }
        setAudioBlob(wavBlob);
        setAudioUrl(URL.createObjectURL(wavBlob));
        setMode(inputMode);
        setError(null);
        setIsConfirmed(false);
      } catch (err) {
        console.error("[VoiceClone] Failed to normalize audio to WAV:", err);
        setError(
          "Could not process this audio format. Please upload/record a standard audio file.",
        );
      }
    },
    [audioUrl],
  );

  const resetCreateVoiceDraft = useCallback(() => {
    if (createVoiceAudioUrl) {
      URL.revokeObjectURL(createVoiceAudioUrl);
    }
    setCreateVoiceName("");
    setCreateVoiceTranscript("");
    setCreateVoiceAudioBlob(null);
    setCreateVoiceAudioUrl(null);
    setCreateVoiceInputMode(null);
    setCreateVoiceError(null);
    setCreateVoiceSaving(false);
    setCreateVoiceIsRecording(false);
  }, [createVoiceAudioUrl]);

  const prepareCreateVoiceAudioBlob = useCallback(
    async (inputBlob: Blob) => {
      try {
        const wavBlob = await normalizeToWavBlob(inputBlob);
        if (createVoiceAudioUrl) {
          URL.revokeObjectURL(createVoiceAudioUrl);
        }
        setCreateVoiceAudioBlob(wavBlob);
        setCreateVoiceAudioUrl(URL.createObjectURL(wavBlob));
        setCreateVoiceError(null);
      } catch (err) {
        console.error("[VoiceClone] Failed to normalize modal voice audio:", err);
        setCreateVoiceError(
          "Could not process this recording. Please try recording again.",
        );
      }
    },
    [createVoiceAudioUrl],
  );

  const openCreateVoiceModal = useCallback(() => {
    resetCreateVoiceDraft();
    setCreateVoiceTranscript(transcript.trim());
    setIsCreateVoiceModalOpen(true);
  }, [resetCreateVoiceDraft, transcript]);

  const closeCreateVoiceModal = useCallback(() => {
    if (createVoiceSaving) {
      return;
    }
    if (createVoiceRecorderRef.current && createVoiceIsRecording) {
      createVoiceRecorderRef.current.stop();
    }
    setIsCreateVoiceModalOpen(false);
    resetCreateVoiceDraft();
  }, [createVoiceIsRecording, createVoiceSaving, resetCreateVoiceDraft]);

  useEffect(() => {
    if (!isCreateVoiceModalOpen) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeCreateVoiceModal();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [closeCreateVoiceModal, isCreateVoiceModalOpen]);

  // Auto-confirm voice cloning when both audio and transcript are available
  const autoConfirm = useCallback(() => {
    if (
      !audioBlob ||
      !transcript.trim() ||
      isConfirmed ||
      isConfirmingRef.current
    ) {
      return;
    }

    isConfirmingRef.current = true;

    const reader = new FileReader();
    reader.onloadend = () => {
      const base64 = reader.result as string;
      const base64Audio = base64.split(",")[1];
      if (base64Audio) {
        console.log(
          "[VoiceClone] Auto-confirming voice clone - audio length:",
          base64Audio.length,
          "transcript:",
          transcript.trim(),
        );
        onVoiceCloneReady(base64Audio, transcript.trim());
        setIsConfirmed(true);
      }
      isConfirmingRef.current = false;
    };
    reader.onerror = () => {
      console.error("[VoiceClone] Auto-confirm FileReader error");
      isConfirmingRef.current = false;
    };
    reader.readAsDataURL(audioBlob);
  }, [audioBlob, transcript, isConfirmed, onVoiceCloneReady]);

  // Trigger auto-confirm when audio becomes available (transcript already exists)
  // or when transcript is entered (audio already exists)
  useEffect(() => {
    if (audioBlob && transcript.trim()) {
      if (!isConfirmed) {
        // Initial auto-confirm with delay to debounce rapid transcript changes
        const timer = setTimeout(autoConfirm, 300);
        return () => clearTimeout(timer);
      } else {
        // Already confirmed - update parent with new transcript (debounced)
        const timer = setTimeout(() => {
          if (!isConfirmingRef.current) {
            isConfirmingRef.current = true;
            const reader = new FileReader();
            reader.onloadend = () => {
              const base64 = reader.result as string;
              const base64Audio = base64.split(",")[1];
              if (base64Audio) {
                console.log(
                  "[VoiceClone] Updating transcript - audio length:",
                  base64Audio.length,
                  "transcript:",
                  transcript.trim(),
                );
                onVoiceCloneReady(base64Audio, transcript.trim());
              }
              isConfirmingRef.current = false;
            };
            reader.onerror = () => {
              isConfirmingRef.current = false;
            };
            reader.readAsDataURL(audioBlob);
          }
        }, 500);
        return () => clearTimeout(timer);
      }
    }
  }, [audioBlob, transcript, isConfirmed, autoConfirm, onVoiceCloneReady]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith("audio/")) {
      setError("Please upload an audio file");
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError("File size must be less than 10MB");
      return;
    }

    setSelectedSavedVoiceId("");
    setSaveVoiceStatus(null);
    void prepareAudioBlob(file, "upload");
  };

  const handleSavedVoiceSelect = async (voiceId: string) => {
    setSelectedSavedVoiceId(voiceId);
    if (!voiceId) {
      return;
    }

    setIsApplyingSavedVoice(true);
    setError(null);
    setSaveVoiceStatus(null);

    try {
      const [voice, audioResponse] = await Promise.all([
        api.getSavedVoice(voiceId),
        fetch(api.savedVoiceAudioUrl(voiceId)),
      ]);

      if (!audioResponse.ok) {
        throw new Error(`Failed to load saved voice audio (${audioResponse.status})`);
      }

      setTranscript(voice.reference_text);
      const voiceAudio = await audioResponse.blob();
      await prepareAudioBlob(voiceAudio, "saved");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load selected saved voice.",
      );
    } finally {
      setIsApplyingSavedVoice(false);
    }
  };

  const handleCreateVoiceFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (!file.type.startsWith("audio/")) {
      setCreateVoiceError("Please upload an audio file.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setCreateVoiceError("File size must be less than 10MB.");
      return;
    }

    if (createVoiceRecorderRef.current && createVoiceIsRecording) {
      createVoiceRecorderRef.current.stop();
      setCreateVoiceIsRecording(false);
    }

    setCreateVoiceInputMode("upload");
    setCreateVoiceError(null);
    void prepareCreateVoiceAudioBlob(file);
  };

  const startCreateVoiceRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeTypes = [
        "audio/wav",
        "audio/ogg",
        "audio/ogg;codecs=opus",
        "audio/webm;codecs=opus",
        "audio/webm",
      ];

      let selectedMimeType = "";
      for (const mimeType of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          selectedMimeType = mimeType;
          break;
        }
      }

      const options = selectedMimeType
        ? { mimeType: selectedMimeType }
        : undefined;
      const mediaRecorder = new MediaRecorder(stream, options);
      createVoiceRecorderRef.current = mediaRecorder;
      createVoiceChunksRef.current = [];
      const actualMimeType = mediaRecorder.mimeType || "audio/webm";

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          createVoiceChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(createVoiceChunksRef.current, {
          type: actualMimeType,
        });
        void prepareCreateVoiceAudioBlob(blob);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setCreateVoiceInputMode("record");
      setCreateVoiceIsRecording(true);
      setCreateVoiceError(null);
    } catch (err) {
      setCreateVoiceError(
        "Microphone access denied. Please allow microphone access.",
      );
      console.error("[VoiceClone] Modal recording error:", err);
    }
  };

  const stopCreateVoiceRecording = () => {
    if (createVoiceRecorderRef.current && createVoiceIsRecording) {
      createVoiceRecorderRef.current.stop();
      setCreateVoiceIsRecording(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Try to use a format that's more compatible with backend processing
      // Prefer formats in order: wav, ogg, webm
      const mimeTypes = [
        "audio/wav",
        "audio/ogg",
        "audio/ogg;codecs=opus",
        "audio/webm;codecs=opus",
        "audio/webm",
      ];

      let selectedMimeType = "";
      for (const mimeType of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          selectedMimeType = mimeType;
          break;
        }
      }

      const options = selectedMimeType
        ? { mimeType: selectedMimeType }
        : undefined;
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      const actualMimeType = mediaRecorder.mimeType || "audio/webm";

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: actualMimeType });
        void prepareAudioBlob(blob, "record");
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setMode("record");
      setError(null);
      setSelectedSavedVoiceId("");
      setSaveVoiceStatus(null);
    } catch (err) {
      setError("Microphone access denied. Please allow microphone access.");
      console.error("Recording error:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handlePlay = () => {
    audioRef.current?.play();
  };

  const handleClear = () => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioBlob(null);
    setAudioUrl(null);
    setTranscript("");
    setMode(null);
    setError(null);
    setIsConfirmed(false);
    setSelectedSavedVoiceId("");
    setSaveVoiceName("");
    setSaveVoiceStatus(null);
    onClear();
  };

  const handleSaveVoice = async () => {
    if (!audioBlob || !transcript.trim() || isSavingVoice) {
      return;
    }

    const trimmedName = saveVoiceName.trim();
    if (!trimmedName) {
      setSaveVoiceStatus({
        tone: "error",
        message: "Enter a voice name before saving.",
      });
      return;
    }

    setIsSavingVoice(true);
    setSaveVoiceStatus(null);

    try {
      const audioBase64 = await blobToBase64Payload(audioBlob);
      await api.createSavedVoice({
        name: trimmedName,
        reference_text: transcript.trim(),
        audio_base64: audioBase64,
        audio_mime_type: audioBlob.type || "audio/wav",
        audio_filename: `voice-clone-saved-${Date.now()}.wav`,
        source_route_kind: "voice_cloning",
      });

      setSaveVoiceName("");
      setSaveVoiceStatus({
        tone: "success",
        message: `Saved voice profile "${trimmedName}".`,
      });
      await loadSavedVoices();
    } catch (err) {
      setSaveVoiceStatus({
        tone: "error",
        message: err instanceof Error ? err.message : "Failed to save voice profile.",
      });
    } finally {
      setIsSavingVoice(false);
    }
  };

  const handleCreateVoiceFromModal = async () => {
    if (!createVoiceAudioBlob || !createVoiceTranscript.trim() || createVoiceSaving) {
      return;
    }

    const trimmedName = createVoiceName.trim();
    if (!trimmedName) {
      setCreateVoiceError("Enter a voice name before saving.");
      return;
    }

    setCreateVoiceSaving(true);
    setCreateVoiceError(null);

    try {
      const audioBase64 = await blobToBase64Payload(createVoiceAudioBlob);
      const createdVoice = await api.createSavedVoice({
        name: trimmedName,
        reference_text: createVoiceTranscript.trim(),
        audio_base64: audioBase64,
        audio_mime_type: createVoiceAudioBlob.type || "audio/wav",
        audio_filename: `voice-profile-${Date.now()}.wav`,
        source_route_kind: "voice_cloning",
      });

      await loadSavedVoices();
      setMode("saved");
      setIsCreateVoiceModalOpen(false);
      resetCreateVoiceDraft();
      await handleSavedVoiceSelect(createdVoice.id);
    } catch (err) {
      setCreateVoiceError(
        err instanceof Error ? err.message : "Failed to create voice profile.",
      );
    } finally {
      setCreateVoiceSaving(false);
    }
  };

  const handleConfirm = async () => {
    if (!audioBlob || !transcript.trim()) {
      setError("Please provide both audio and transcript");
      return;
    }

    try {
      // Convert blob to base64
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result as string;
        // Remove data URL prefix
        const base64Audio = base64.split(",")[1];
        console.log(
          "[VoiceClone] Calling onVoiceCloneReady with audio length:",
          base64Audio?.length,
          "transcript:",
          transcript.trim(),
        );
        onVoiceCloneReady(base64Audio, transcript.trim());
        setIsConfirmed(true);
      };
      reader.onerror = () => {
        setError("Failed to read audio file");
        console.error("[VoiceClone] FileReader error");
      };
      reader.readAsDataURL(audioBlob);
    } catch (err) {
      setError("Failed to process audio");
      console.error(err);
    }
  };

  return (
    <div className="space-y-3">
      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="p-2 rounded bg-red-950/50 border border-red-900/50 text-red-400 text-xs"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Transcript input - always visible */}
      <div>
        <label className="block text-xs text-gray-500 mb-1.5">
          Transcript
          <span className="text-red-400 ml-1">*</span>
        </label>
        <textarea
          value={transcript}
          onChange={(e) => setTranscript(e.target.value)}
          placeholder="Enter what you will say in the recording..."
          rows={3}
          className="textarea text-sm"
        />
        <p className="text-xs text-gray-600 mt-1">
          Type transcript text, then upload, record, or choose a saved voice
        </p>
      </div>

      {/* Audio controls */}
      {!audioBlob ? (
        <div className="space-y-2">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            {/* Upload button */}
            <button
              onClick={() => {
                setMode("upload");
                fileInputRef.current?.click();
              }}
              className={clsx(
                "flex flex-col items-center gap-2 p-4 rounded-lg border transition-colors min-h-[80px]",
                mode === "upload"
                  ? "border-white/40 bg-[#1a1a1a]"
                  : "border-[#2a2a2a] bg-[#161616] hover:bg-[#1a1a1a]",
              )}
            >
              <Upload className="w-5 h-5 text-gray-400" />
              <span className="text-xs text-gray-400">Upload Audio</span>
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              onChange={handleFileUpload}
              className="hidden"
            />

            {/* Record button */}
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={clsx(
                "flex flex-col items-center gap-2 p-4 rounded-lg border transition-colors min-h-[80px]",
                isRecording || mode === "record"
                  ? "border-white/50 bg-[#1a1a1a]"
                  : "border-[#2a2a2a] bg-[#161616] hover:bg-[#1a1a1a]",
              )}
            >
              {isRecording ? (
                <>
                  <Square className="w-5 h-5 text-white" />
                  <span className="text-xs text-white">Stop Recording</span>
                </>
              ) : (
                <>
                  <Mic className="w-5 h-5 text-gray-400" />
                  <span className="text-xs text-gray-400">Record Voice</span>
                </>
              )}
            </button>

            {/* Saved voice dropdown */}
            <button
              onClick={() => {
                setMode("saved");
                setError(null);
                if (!savedVoices.length && !savedVoicesLoading) {
                  void loadSavedVoices();
                }
              }}
              className={clsx(
                "flex flex-col items-center gap-2 p-4 rounded-lg border transition-colors min-h-[80px]",
                mode === "saved"
                  ? "border-white/40 bg-[#1a1a1a]"
                  : "border-[#2a2a2a] bg-[#161616] hover:bg-[#1a1a1a]",
              )}
            >
              <Library className="w-5 h-5 text-gray-400" />
              <span className="text-xs text-gray-400">Saved Voice</span>
            </button>
          </div>

          <div className="flex justify-end">
            <button
              onClick={openCreateVoiceModal}
              className="btn btn-secondary text-xs min-h-[34px]"
            >
              <BookmarkPlus className="w-3.5 h-3.5" />
              Create Voice
            </button>
          </div>

          <AnimatePresence>
            {mode === "saved" && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="p-3 rounded-lg border border-[#2a2a2a] bg-[#161616] space-y-2 overflow-hidden"
              >
                <div className="flex items-center justify-between gap-2">
                  <label className="text-xs text-gray-400">
                    Select Saved Voice
                  </label>
                  <button
                    onClick={() => void loadSavedVoices()}
                    disabled={savedVoicesLoading}
                    className="btn btn-ghost text-xs min-h-[32px]"
                  >
                    <RefreshCw
                      className={clsx(
                        "w-3.5 h-3.5",
                        savedVoicesLoading && "animate-spin",
                      )}
                    />
                    Refresh
                  </button>
                </div>
                <Select
                  value={selectedSavedVoiceId}
                  onValueChange={(value) => {
                    void handleSavedVoiceSelect(value);
                  }}
                  disabled={
                    savedVoicesLoading ||
                    isApplyingSavedVoice ||
                    savedVoices.length === 0
                  }
                >
                  <SelectTrigger className="text-sm">
                    <SelectValue placeholder="Choose a saved voice" />
                  </SelectTrigger>
                  <SelectContent>
                    {savedVoices.map((voice) => (
                      <SelectItem key={voice.id} value={voice.id}>
                        {voice.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {savedVoicesError && (
                  <p className="text-xs text-red-400">{savedVoicesError}</p>
                )}
                {!savedVoicesLoading && !savedVoicesError && savedVoices.length === 0 && (
                  <p className="text-xs text-gray-500">
                    No saved voices yet. Save one from Voice Design or after uploading/recording.
                  </p>
                )}
                {isApplyingSavedVoice && (
                  <div className="text-xs text-gray-400 inline-flex items-center gap-2">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Loading selected voice...
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      ) : (
        <div className="space-y-3">
          {/* Audio player */}
          <div className="p-3 rounded-lg bg-[#161616] border border-[#2a2a2a]">
            <div className="flex items-center gap-2 mb-2">
              <button
                onClick={handlePlay}
                className="p-1.5 rounded bg-[#1f1f1f] hover:bg-[#2a2a2a]"
              >
                <Play className="w-3.5 h-3.5 text-white" />
              </button>
              <div className="flex-1 text-xs text-gray-500">
                {mode === "upload"
                  ? "Uploaded audio"
                  : mode === "record"
                    ? "Recorded audio"
                    : "Saved voice audio"}
              </div>
              <div className="text-xs text-gray-600">
                {(audioBlob.size / 1024).toFixed(0)} KB
              </div>
            </div>
            <audio
              ref={audioRef}
              src={audioUrl || ""}
              className="w-full h-8"
              controls
            />
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-2">
            {isConfirmed ? (
              <div className="flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded bg-[#1a1a1a] border border-[#2a2a2a] text-gray-300 text-sm min-h-[44px]">
                <Check className="w-4 h-4" />
                Voice Ready
              </div>
            ) : (
              <button
                onClick={handleConfirm}
                disabled={!transcript.trim()}
                className="btn btn-primary flex-1 text-sm min-h-[44px]"
              >
                <Check className="w-4 h-4" />
                Use This Voice
              </button>
            )}
            <button
              onClick={handleClear}
              className="btn btn-ghost text-sm min-h-[44px] min-w-[44px]"
            >
              <X className="w-3.5 h-3.5" />
              Clear
            </button>
          </div>

          {mode !== "saved" && (
            <div className="p-3 rounded-lg border border-[#2a2a2a] bg-[#161616] space-y-2">
              <label className="text-xs text-gray-400 block">
                Save This Voice
              </label>
              <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr),auto]">
                <input
                  value={saveVoiceName}
                  onChange={(event) => setSaveVoiceName(event.target.value)}
                  placeholder="Voice name"
                  className="input text-sm"
                  disabled={isSavingVoice}
                />
                <button
                  onClick={handleSaveVoice}
                  disabled={isSavingVoice || !transcript.trim()}
                  className="btn btn-secondary min-h-[40px] sm:min-w-[130px]"
                >
                  {isSavingVoice ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <BookmarkPlus className="w-4 h-4" />
                      Save Voice
                    </>
                  )}
                </button>
              </div>
              <p className="text-[10px] text-gray-500">
                Stores this audio sample and transcript for one-click reuse in cloning.
              </p>
              <AnimatePresence>
                {saveVoiceStatus && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className={clsx(
                      "p-2 rounded border text-xs",
                      saveVoiceStatus.tone === "success"
                        ? "bg-[var(--status-positive-bg)] border-[var(--status-positive-border)] text-[var(--status-positive-text)]"
                        : "bg-[var(--danger-bg)] border-[var(--danger-border)] text-[var(--danger-text)]",
                    )}
                  >
                    {saveVoiceStatus.message}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}
        </div>
      )}

      <AnimatePresence>
        {isCreateVoiceModalOpen && (
          <motion.div
            className="fixed inset-0 z-[70] bg-black/75 p-4 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={closeCreateVoiceModal}
          >
            <motion.div
              initial={{ y: 14, opacity: 0, scale: 0.985 }}
              animate={{ y: 0, opacity: 1, scale: 1 }}
              exit={{ y: 14, opacity: 0, scale: 0.985 }}
              transition={{ duration: 0.16 }}
              onClick={(event) => event.stopPropagation()}
              className="mx-auto mt-[8vh] w-full max-w-xl rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4 sm:p-5 shadow-2xl"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="text-sm font-medium text-[var(--text-primary)]">
                    Create Voice Profile
                  </h3>
                  <p className="mt-1 text-xs text-[var(--text-subtle)]">
                    Upload or record a sample, set transcript text, then save with a name.
                  </p>
                </div>
                <button
                  onClick={closeCreateVoiceModal}
                  className="btn btn-ghost min-h-[32px] min-w-[32px]"
                  disabled={createVoiceSaving}
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>

              <div className="mt-4 space-y-3">
                <div>
                  <label className="block text-xs text-[var(--text-muted)] mb-1.5">
                    Voice Name
                    <span className="text-red-400 ml-1">*</span>
                  </label>
                  <input
                    value={createVoiceName}
                    onChange={(event) => setCreateVoiceName(event.target.value)}
                    className="input text-sm"
                    placeholder="e.g. Support Agent Voice"
                    disabled={createVoiceSaving}
                  />
                </div>

                <div>
                  <label className="block text-xs text-[var(--text-muted)] mb-1.5">
                    Transcript
                    <span className="text-red-400 ml-1">*</span>
                  </label>
                  <textarea
                    value={createVoiceTranscript}
                    onChange={(event) =>
                      setCreateVoiceTranscript(event.target.value)
                    }
                    rows={3}
                    className="textarea text-sm"
                    placeholder="Enter exactly what is spoken in the recording..."
                    disabled={createVoiceSaving}
                  />
                </div>

                <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3">
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <span className="text-xs text-[var(--text-muted)]">
                      Voice Sample
                    </span>
                    <span className="text-[10px] text-[var(--text-subtle)]">
                      5–20 seconds recommended
                    </span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    <button
                      onClick={() => {
                        setCreateVoiceInputMode("upload");
                        createVoiceFileInputRef.current?.click();
                      }}
                      className={clsx(
                        "flex items-center justify-center gap-2 rounded-lg border px-3 py-2 text-xs transition-colors",
                        createVoiceInputMode === "upload"
                          ? "border-[var(--border-strong)] bg-[var(--bg-surface-3)] text-[var(--text-primary)]"
                          : "border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] hover:border-[var(--border-strong)]",
                      )}
                      disabled={createVoiceSaving || createVoiceIsRecording}
                    >
                      <Upload className="w-3.5 h-3.5" />
                      Upload Recording
                    </button>
                    <input
                      ref={createVoiceFileInputRef}
                      type="file"
                      accept="audio/*"
                      onChange={handleCreateVoiceFileUpload}
                      className="hidden"
                    />

                    <button
                      onClick={
                        createVoiceIsRecording
                          ? stopCreateVoiceRecording
                          : startCreateVoiceRecording
                      }
                      className={clsx(
                        "flex items-center justify-center gap-2 rounded-lg border px-3 py-2 text-xs transition-colors",
                        createVoiceIsRecording || createVoiceInputMode === "record"
                          ? "border-[var(--border-strong)] bg-[var(--bg-surface-3)] text-[var(--text-primary)]"
                          : "border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] hover:border-[var(--border-strong)]",
                      )}
                      disabled={createVoiceSaving}
                    >
                      {createVoiceIsRecording ? (
                        <>
                          <Square className="w-3.5 h-3.5" />
                          Stop Recording
                        </>
                      ) : (
                        <>
                          <Mic className="w-3.5 h-3.5" />
                          Record Voice
                        </>
                      )}
                    </button>
                  </div>

                  {createVoiceAudioUrl ? (
                    <div className="mt-3 space-y-2">
                      <audio
                        src={createVoiceAudioUrl}
                        controls
                        className="w-full"
                      />
                      <div className="flex justify-end">
                        <button
                          onClick={() => {
                            if (createVoiceAudioUrl) {
                              URL.revokeObjectURL(createVoiceAudioUrl);
                            }
                            setCreateVoiceAudioBlob(null);
                            setCreateVoiceAudioUrl(null);
                          }}
                          className="btn btn-ghost text-xs min-h-[30px]"
                          disabled={createVoiceSaving || createVoiceIsRecording}
                        >
                          Clear Sample
                        </button>
                      </div>
                    </div>
                  ) : (
                    <p className="mt-2 text-xs text-[var(--text-subtle)]">
                      Choose upload or record to attach a sample.
                    </p>
                  )}
                </div>

                <AnimatePresence>
                  {createVoiceError && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="p-2 rounded border text-xs bg-[var(--danger-bg)] border-[var(--danger-border)] text-[var(--danger-text)]"
                    >
                      {createVoiceError}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              <div className="mt-4 flex items-center justify-end gap-2">
                <button
                  onClick={closeCreateVoiceModal}
                  className="btn btn-ghost text-sm min-h-[38px]"
                  disabled={createVoiceSaving}
                >
                  Cancel
                </button>
                <button
                  onClick={() => void handleCreateVoiceFromModal()}
                  disabled={
                    createVoiceSaving ||
                    !createVoiceName.trim() ||
                    !createVoiceTranscript.trim() ||
                    !createVoiceAudioBlob
                  }
                  className="btn btn-secondary text-sm min-h-[38px]"
                >
                  {createVoiceSaving ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <BookmarkPlus className="w-4 h-4" />
                      Save Voice
                    </>
                  )}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
