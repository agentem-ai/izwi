import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Mic, Square, Play, Check, X } from "lucide-react";
import clsx from "clsx";

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

function encodeWavPcm16(samples: Float32Array, sampleRate: number): ArrayBuffer {
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
  const [mode, setMode] = useState<"upload" | "record" | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isConfirmed, setIsConfirmed] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const isConfirmingRef = useRef(false);

  const prepareAudioBlob = useCallback(
    async (inputBlob: Blob, inputMode: "upload" | "record") => {
      try {
        const wavBlob = await normalizeToWavBlob(inputBlob);
        if (audioUrl) {
          URL.revokeObjectURL(audioUrl);
        }
        setAudioBlob(wavBlob);
        setAudioUrl(URL.createObjectURL(wavBlob));
        setMode(inputMode);
        setError(null);
      } catch (err) {
        console.error("[VoiceClone] Failed to normalize audio to WAV:", err);
        setError(
          "Could not process this audio format. Please upload/record a standard audio file.",
        );
      }
    },
    [audioUrl],
  );

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

    void prepareAudioBlob(file, "upload");
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
    onClear();
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
          Type what you'll say, then record or upload audio
        </p>
      </div>

      {/* Audio controls */}
      {!audioBlob ? (
        <div className="grid grid-cols-2 gap-2">
          {/* Upload button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex flex-col items-center gap-2 p-4 rounded-lg border border-[#2a2a2a] bg-[#161616] hover:bg-[#1a1a1a] transition-colors"
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
              "flex flex-col items-center gap-2 p-4 rounded-lg border transition-colors",
              isRecording
                ? "border-red-500/50 bg-red-950/20"
                : "border-[#2a2a2a] bg-[#161616] hover:bg-[#1a1a1a]",
            )}
          >
            {isRecording ? (
              <>
                <Square className="w-5 h-5 text-red-400" />
                <span className="text-xs text-red-400">Stop Recording</span>
              </>
            ) : (
              <>
                <Mic className="w-5 h-5 text-gray-400" />
                <span className="text-xs text-gray-400">Record Voice</span>
              </>
            )}
          </button>
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
                {mode === "upload" ? "Uploaded audio" : "Recorded audio"}
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
              <div className="flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded bg-green-950/50 border border-green-800/50 text-green-400 text-sm">
                <Check className="w-4 h-4" />
                Voice Ready
              </div>
            ) : (
              <button
                onClick={handleConfirm}
                disabled={!transcript.trim()}
                className="btn btn-primary flex-1 text-sm"
              >
                <Check className="w-4 h-4" />
                Use This Voice
              </button>
            )}
            <button onClick={handleClear} className="btn btn-ghost text-sm">
              <X className="w-3.5 h-3.5" />
              Clear
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
