import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, MicOff, Volume2, Loader2, PhoneOff, AudioLines } from "lucide-react";
import clsx from "clsx";

import { api, ChatMessage, ModelInfo } from "../api";
import { SPEAKERS } from "../types";

type RuntimeStatus =
  | "idle"
  | "listening"
  | "user_speaking"
  | "processing"
  | "assistant_speaking";

interface TranscriptEntry {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: number;
}

interface VoicePageProps {
  models: ModelInfo[];
  loading: boolean;
  onError?: (message: string) => void;
}

const SYSTEM_PROMPT: ChatMessage = {
  role: "system",
  content:
    "You are a helpful voice assistant. Reply with concise spoken-friendly language. Avoid markdown. Keep responses brief unless asked for details.",
};

function parseFinalAnswer(content: string): string {
  const openTag = "<think>";
  const closeTag = "</think>";
  let out = content;

  while (true) {
    const start = out.indexOf(openTag);
    if (start === -1) break;
    const end = out.indexOf(closeTag, start + openTag.length);
    if (end === -1) {
      out = out.slice(0, start);
      break;
    }
    out = `${out.slice(0, start)}${out.slice(end + closeTag.length)}`;
  }

  return out.trim();
}

async function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      const base64 = result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

function encodeWavPcm16(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const byteRate = sampleRate * blockAlign;
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
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    const int16 = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    view.setInt16(offset, int16, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

async function transcodeToWav(
  inputBlob: Blob,
  targetSampleRate = 16000,
): Promise<Blob> {
  if (inputBlob.type === "audio/wav" || inputBlob.type === "audio/x-wav") {
    return inputBlob;
  }

  const decodeContext = new AudioContext();
  try {
    const sourceBytes = await inputBlob.arrayBuffer();
    const decoded = await decodeContext.decodeAudioData(sourceBytes.slice(0));

    const monoBuffer = decodeContext.createBuffer(
      1,
      decoded.length,
      decoded.sampleRate,
    );
    const mono = monoBuffer.getChannelData(0);

    for (let i = 0; i < decoded.length; i += 1) {
      let sum = 0;
      for (let ch = 0; ch < decoded.numberOfChannels; ch += 1) {
        sum += decoded.getChannelData(ch)[i] ?? 0;
      }
      mono[i] = sum / decoded.numberOfChannels;
    }

    const rendered = await (() => {
      if (decoded.sampleRate === targetSampleRate) {
        return Promise.resolve(monoBuffer);
      }

      const targetLength = Math.ceil(
        (monoBuffer.length * targetSampleRate) / monoBuffer.sampleRate,
      );
      const offline = new OfflineAudioContext(1, targetLength, targetSampleRate);
      const source = offline.createBufferSource();
      source.buffer = monoBuffer;
      source.connect(offline.destination);
      source.start(0);
      return offline.startRendering();
    })();

    return encodeWavPcm16(rendered.getChannelData(0), targetSampleRate);
  } finally {
    decodeContext.close().catch(() => {});
  }
}

export function VoicePage({ models, loading, onError }: VoicePageProps) {
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [conversation, setConversation] = useState<ChatMessage[]>([]);
  const [audioLevel, setAudioLevel] = useState(0);

  const [selectedAsrModel, setSelectedAsrModel] = useState<string | null>(null);
  const [selectedTextModel, setSelectedTextModel] = useState<string | null>(null);
  const [selectedTtsModel, setSelectedTtsModel] = useState<string | null>(null);
  const [selectedSpeaker, setSelectedSpeaker] = useState("Serena");

  const [vadThreshold, setVadThreshold] = useState(0.02);
  const [silenceDurationMs, setSilenceDurationMs] = useState(900);
  const [minSpeechMs, setMinSpeechMs] = useState(300);

  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const vadTimerRef = useRef<number | null>(null);
  const speechStartRef = useRef<number | null>(null);
  const silenceMsRef = useRef(0);
  const processingRef = useRef(false);
  const runtimeStatusRef = useRef<RuntimeStatus>("idle");
  const conversationRef = useRef<ChatMessage[]>([]);
  const isSessionActiveRef = useRef(false);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  const availableModels = useMemo(
    () =>
      models.filter((m) => m.status === "downloaded" || m.status === "ready"),
    [models],
  );

  const asrModels = useMemo(
    () =>
      availableModels.filter(
        (m) =>
          m.variant.includes("Qwen3-ASR") ||
          m.variant.includes("ForcedAligner") ||
          m.variant.includes("Voxtral"),
      ),
    [availableModels],
  );

  const textModels = useMemo(
    () => availableModels.filter((m) => m.variant === "Qwen3-0.6B-4bit"),
    [availableModels],
  );

  const ttsModels = useMemo(
    () =>
      availableModels.filter(
        (m) => m.variant.includes("Qwen3-TTS") && !m.variant.includes("Tokenizer"),
      ),
    [availableModels],
  );

  useEffect(() => {
    runtimeStatusRef.current = runtimeStatus;
  }, [runtimeStatus]);

  useEffect(() => {
    conversationRef.current = conversation;
  }, [conversation]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript, runtimeStatus]);

  useEffect(() => {
    if (!selectedAsrModel || !asrModels.some((m) => m.variant === selectedAsrModel)) {
      setSelectedAsrModel(asrModels[0]?.variant ?? null);
    }
  }, [asrModels, selectedAsrModel]);

  useEffect(() => {
    if (
      !selectedTextModel ||
      !textModels.some((m) => m.variant === selectedTextModel)
    ) {
      setSelectedTextModel(textModels[0]?.variant ?? null);
    }
  }, [textModels, selectedTextModel]);

  useEffect(() => {
    if (!selectedTtsModel || !ttsModels.some((m) => m.variant === selectedTtsModel)) {
      setSelectedTtsModel(ttsModels[0]?.variant ?? null);
    }
  }, [ttsModels, selectedTtsModel]);

  const clearAudioPlayback = useCallback(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
      audio.src = "";
    }

    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
  }, []);

  const stopSession = useCallback(() => {
    isSessionActiveRef.current = false;
    processingRef.current = false;
    silenceMsRef.current = 0;
    speechStartRef.current = null;
    setRuntimeStatus("idle");
    setAudioLevel(0);

    if (vadTimerRef.current != null) {
      window.clearInterval(vadTimerRef.current);
      vadTimerRef.current = null;
    }

    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state === "recording") {
      recorder.stop();
    }
    mediaRecorderRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }

    analyserRef.current = null;
    clearAudioPlayback();
  }, [clearAudioPlayback]);

  useEffect(() => {
    return () => stopSession();
  }, [stopSession]);

  const addTranscript = useCallback((role: "user" | "assistant", text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    setTranscript((prev) => [
      ...prev,
      {
        id: `${role}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        role,
        text: trimmed,
        timestamp: Date.now(),
      },
    ]);
  }, []);

  const processUtterance = useCallback(
    async (audioBlob: Blob) => {
      if (!selectedAsrModel || !selectedTextModel || !selectedTtsModel) {
        setError("Select ASR, text, and TTS models before starting voice mode.");
        setRuntimeStatus("listening");
        processingRef.current = false;
        return;
      }

      try {
        setRuntimeStatus("processing");
        const wavBlob = await transcodeToWav(audioBlob, 16000);
        const audioBase64 = await blobToBase64(wavBlob);
        const asr = await api.asrTranscribe({
          audio_base64: audioBase64,
          model_id: selectedAsrModel,
          language: "Auto",
        });

        const userText = asr.transcription.trim();
        if (!userText) {
          processingRef.current = false;
          if (isSessionActiveRef.current) {
            setRuntimeStatus("listening");
          }
          return;
        }

        addTranscript("user", userText);

        const requestMessages: ChatMessage[] = [
          SYSTEM_PROMPT,
          ...conversationRef.current,
          { role: "user", content: userText },
        ];

        const chat = await api.chatCompletions({
          model_id: selectedTextModel,
          messages: requestMessages,
          max_tokens: 1536,
        });

        const rawAssistant = chat.message.content || "";
        const assistantText = parseFinalAnswer(rawAssistant) || rawAssistant.trim();
        addTranscript("assistant", assistantText);

        setConversation((prev) => [
          ...prev,
          { role: "user", content: userText },
          { role: "assistant", content: assistantText },
        ]);

        const tts = await api.generateTTSWithStats({
          text: assistantText,
          model_id: selectedTtsModel,
          speaker: selectedSpeaker,
          format: "wav",
        });

        clearAudioPlayback();
        const nextUrl = URL.createObjectURL(tts.audioBlob);
        audioUrlRef.current = nextUrl;

        const audio = audioRef.current;
        if (!audio) throw new Error("Audio output is not available");
        audio.src = nextUrl;

        setRuntimeStatus("assistant_speaking");
        await audio.play();
      } catch (err) {
        const message = err instanceof Error ? err.message : "Voice turn failed";
        setError(message);
        onError?.(message);
        if (isSessionActiveRef.current) {
          setRuntimeStatus("listening");
        } else {
          setRuntimeStatus("idle");
        }
      } finally {
        processingRef.current = false;
      }
    },
    [
      addTranscript,
      clearAudioPlayback,
      onError,
      selectedAsrModel,
      selectedSpeaker,
      selectedTextModel,
      selectedTtsModel,
    ],
  );

  const startSession = useCallback(async () => {
    if (!selectedAsrModel || !selectedTextModel || !selectedTtsModel) {
      const message = "Select ASR, text, and TTS models before starting voice mode.";
      setError(message);
      onError?.(message);
      return;
    }

    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.2;
      source.connect(analyser);
      analyserRef.current = analyser;

      let recorder: MediaRecorder | null = null;
      const mimeCandidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
      ];
      for (const mimeType of mimeCandidates) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          recorder = new MediaRecorder(stream, { mimeType });
          break;
        }
      }
      if (!recorder) {
        recorder = new MediaRecorder(stream);
      }

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder?.mimeType || "audio/webm" });
        chunksRef.current = [];

        if (blob.size < 1200) {
          processingRef.current = false;
          if (isSessionActiveRef.current && runtimeStatusRef.current !== "assistant_speaking") {
            setRuntimeStatus("listening");
          }
          return;
        }

        void processUtterance(blob);
      };

      mediaRecorderRef.current = recorder;
      isSessionActiveRef.current = true;
      processingRef.current = false;
      silenceMsRef.current = 0;
      speechStartRef.current = null;
      setRuntimeStatus("listening");

      const VAD_INTERVAL = 80;
      vadTimerRef.current = window.setInterval(() => {
        const analyserNode = analyserRef.current;
        const recorderNode = mediaRecorderRef.current;
        if (!analyserNode || !recorderNode || !isSessionActiveRef.current) return;

        const data = new Uint8Array(analyserNode.fftSize);
        analyserNode.getByteTimeDomainData(data);

        let sumSquares = 0;
        for (let i = 0; i < data.length; i += 1) {
          const centered = (data[i] - 128) / 128;
          sumSquares += centered * centered;
        }
        const rms = Math.sqrt(sumSquares / data.length);
        setAudioLevel(rms);

        const isSpeech = rms >= vadThreshold;
        const isRecording = recorderNode.state === "recording";
        const now = Date.now();

        if (isSpeech) {
          silenceMsRef.current = 0;

          if (runtimeStatusRef.current === "assistant_speaking") {
            clearAudioPlayback();
            setRuntimeStatus("listening");
          }

          if (!isRecording && !processingRef.current) {
            chunksRef.current = [];
            recorderNode.start();
            speechStartRef.current = now;
            setRuntimeStatus("user_speaking");
          }
          return;
        }

        if (isRecording) {
          silenceMsRef.current += VAD_INTERVAL;
          const speechDuration = speechStartRef.current ? now - speechStartRef.current : 0;
          if (speechDuration >= minSpeechMs && silenceMsRef.current >= silenceDurationMs) {
            processingRef.current = true;
            setRuntimeStatus("processing");
            recorderNode.stop();
            silenceMsRef.current = 0;
            speechStartRef.current = null;
          }
        }
      }, VAD_INTERVAL);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Failed to start microphone session";
      setError(message);
      onError?.(message);
      stopSession();
    }
  }, [
    clearAudioPlayback,
    minSpeechMs,
    onError,
    processUtterance,
    selectedAsrModel,
    selectedTextModel,
    selectedTtsModel,
    silenceDurationMs,
    stopSession,
    vadThreshold,
  ]);

  const toggleSession = () => {
    if (runtimeStatus === "idle") {
      void startSession();
    } else {
      stopSession();
    }
  };

  const statusLabel = {
    idle: "Idle",
    listening: "Listening",
    user_speaking: "User speaking",
    processing: "Thinking",
    assistant_speaking: "Assistant speaking",
  }[runtimeStatus];

  const vadPercent = Math.min(100, Math.round((audioLevel / Math.max(vadThreshold, 0.001)) * 40));

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col items-center justify-center py-24 gap-3">
          <motion.div
            className="w-8 h-8 border-2 border-white border-t-transparent rounded-full"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
          <p className="text-sm text-gray-400">Loading models...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-white">Voice</h1>
        <p className="text-sm text-gray-500 mt-1">
          Real-time assistant pipeline: ASR ↔ LLM ↔ TTS with VAD and barge-in
        </p>
      </div>

      <div className="grid lg:grid-cols-[360px,1fr] gap-4 lg:gap-6">
        <div className="card p-4 space-y-4">
          <div className="space-y-1">
            <label className="text-xs text-gray-500">ASR Model</label>
            <select
              value={selectedAsrModel ?? ""}
              onChange={(e) => setSelectedAsrModel(e.target.value)}
              className="input"
            >
              {asrModels.map((m) => (
                <option key={m.variant} value={m.variant}>
                  {m.variant}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs text-gray-500">Text Model</label>
            <select
              value={selectedTextModel ?? ""}
              onChange={(e) => setSelectedTextModel(e.target.value)}
              className="input"
            >
              {textModels.map((m) => (
                <option key={m.variant} value={m.variant}>
                  {m.variant}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs text-gray-500">TTS Model</label>
            <select
              value={selectedTtsModel ?? ""}
              onChange={(e) => setSelectedTtsModel(e.target.value)}
              className="input"
            >
              {ttsModels.map((m) => (
                <option key={m.variant} value={m.variant}>
                  {m.variant}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs text-gray-500">Assistant Voice</label>
            <select
              value={selectedSpeaker}
              onChange={(e) => setSelectedSpeaker(e.target.value)}
              className="input"
            >
              {SPEAKERS.map((speaker) => (
                <option key={speaker.id} value={speaker.id}>
                  {speaker.name} ({speaker.language})
                </option>
              ))}
            </select>
          </div>

          <div className="pt-2 border-t border-[#2a2a2a] space-y-3">
            <div>
              <label className="text-xs text-gray-500">
                VAD Sensitivity ({vadThreshold.toFixed(3)})
              </label>
              <input
                type="range"
                min={0.005}
                max={0.08}
                step={0.001}
                value={vadThreshold}
                onChange={(e) => setVadThreshold(parseFloat(e.target.value))}
                className="w-full mt-1"
              />
            </div>

            <div>
              <label className="text-xs text-gray-500">
                End Silence (ms): {silenceDurationMs}
              </label>
              <input
                type="range"
                min={400}
                max={1800}
                step={50}
                value={silenceDurationMs}
                onChange={(e) => setSilenceDurationMs(parseInt(e.target.value, 10))}
                className="w-full mt-1"
              />
            </div>

            <div>
              <label className="text-xs text-gray-500">
                Minimum Speech (ms): {minSpeechMs}
              </label>
              <input
                type="range"
                min={150}
                max={1200}
                step={50}
                value={minSpeechMs}
                onChange={(e) => setMinSpeechMs(parseInt(e.target.value, 10))}
                className="w-full mt-1"
              />
            </div>
          </div>

          <button
            onClick={toggleSession}
            className={clsx(
              "btn w-full text-sm",
              runtimeStatus === "idle" ? "btn-primary" : "btn-danger",
            )}
            disabled={
              asrModels.length === 0 || textModels.length === 0 || ttsModels.length === 0
            }
          >
            {runtimeStatus === "idle" ? (
              <>
                <Mic className="w-4 h-4" />
                Start Voice Session
              </>
            ) : (
              <>
                <PhoneOff className="w-4 h-4" />
                Stop Session
              </>
            )}
          </button>

          {(asrModels.length === 0 || textModels.length === 0 || ttsModels.length === 0) && (
            <p className="text-xs text-amber-400">
              Download required ASR/Text/TTS models first in My Models.
            </p>
          )}
        </div>

        <div className="card p-4 flex flex-col min-h-[620px]">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className="text-sm text-white font-medium">Session Status</span>
              <span
                className={clsx(
                  "text-xs px-2 py-1 rounded",
                  runtimeStatus === "listening" && "bg-emerald-500/20 text-emerald-300",
                  runtimeStatus === "user_speaking" && "bg-blue-500/20 text-blue-300",
                  runtimeStatus === "processing" && "bg-amber-500/20 text-amber-300",
                  runtimeStatus === "assistant_speaking" && "bg-cyan-500/20 text-cyan-300",
                  runtimeStatus === "idle" && "bg-gray-600/30 text-gray-300",
                )}
              >
                {statusLabel}
              </span>
            </div>

            <div className="flex items-center gap-2">
              {runtimeStatus === "assistant_speaking" ? (
                <Volume2 className="w-4 h-4 text-cyan-300" />
              ) : runtimeStatus === "user_speaking" ? (
                <AudioLines className="w-4 h-4 text-blue-300" />
              ) : runtimeStatus === "processing" ? (
                <Loader2 className="w-4 h-4 text-amber-300 animate-spin" />
              ) : runtimeStatus === "listening" ? (
                <Mic className="w-4 h-4 text-emerald-300" />
              ) : (
                <MicOff className="w-4 h-4 text-gray-500" />
              )}
            </div>
          </div>

          <div className="mb-4">
            <div className="h-2 rounded bg-[#1b1b1b] border border-[#2a2a2a] overflow-hidden">
              <div
                className={clsx(
                  "h-full transition-all duration-75",
                  runtimeStatus === "assistant_speaking" ? "bg-cyan-400" : "bg-emerald-400",
                )}
                style={{ width: `${vadPercent}%` }}
              />
            </div>
            <p className="text-[11px] text-gray-600 mt-1">
              Live input level (barge-in enabled while assistant is speaking)
            </p>
          </div>

          <div className="flex-1 overflow-y-auto pr-1 space-y-3">
            {transcript.length === 0 ? (
              <div className="h-full flex items-center justify-center text-center">
                <div>
                  <p className="text-sm text-gray-400">No transcript yet.</p>
                  <p className="text-xs text-gray-600 mt-1">
                    Start a voice session and speak to begin.
                  </p>
                </div>
              </div>
            ) : (
              transcript.map((entry) => {
                const isUser = entry.role === "user";
                return (
                  <div
                    key={entry.id}
                    className={clsx("flex", isUser ? "justify-end" : "justify-start")}
                  >
                    <div
                      className={clsx(
                        "max-w-[85%] rounded-lg px-3 py-2.5 border text-sm whitespace-pre-wrap",
                        isUser
                          ? "bg-white text-black border-white"
                          : "bg-[#171717] text-gray-200 border-[#2a2a2a]",
                      )}
                    >
                      <div
                        className={clsx(
                          "text-[10px] mb-1 uppercase tracking-wide",
                          isUser ? "text-black/60" : "text-gray-500",
                        )}
                      >
                        {isUser ? "User" : "Assistant"}
                      </div>
                      {entry.text}
                    </div>
                  </div>
                );
              })
            )}
            <div ref={transcriptEndRef} />
          </div>

          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-3 p-2 rounded bg-red-950/50 border border-red-900/50 text-red-300 text-xs"
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <audio
        ref={audioRef}
        className="hidden"
        onEnded={() => {
          clearAudioPlayback();
          if (isSessionActiveRef.current && !processingRef.current) {
            setRuntimeStatus("listening");
          } else if (!isSessionActiveRef.current) {
            setRuntimeStatus("idle");
          }
        }}
      />
    </div>
  );
}
