import { useState, useMemo } from "react";
import {
  Download,
  Play,
  Square,
  Trash2,
  HardDrive,
  Search,
  Loader2,
  X,
  RefreshCw,
} from "lucide-react";
import { ModelInfo } from "../api";
import { withQwen3Prefix } from "../utils/modelDisplay";
import clsx from "clsx";

interface MyModelsPageProps {
  models: ModelInfo[];
  loading: boolean;
  downloadProgress: Record<
    string,
    { percent: number; currentFile: string; status: string }
  >;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onRefresh: () => void;
}

type FilterType = "all" | "downloaded" | "loaded" | "not_downloaded";
type CategoryType = "all" | "tts" | "asr" | "chat";

const MODEL_DETAILS: Record<
  string,
  {
    shortName: string;
    fullName: string;
    description: string;
    category: "tts" | "asr" | "chat";
    capabilities: string[];
    size: string;
  }
> = {
  // TTS 0.6B Base models
  "Qwen3-TTS-12Hz-0.6B-Base": {
    shortName: "0.6B Base",
    fullName: "Qwen3-TTS 12Hz 0.6B Base",
    description: "Voice cloning with reference audio samples",
    category: "tts",
    capabilities: ["Voice Cloning"],
    size: "2.3 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-Base-4bit": {
    shortName: "0.6B Base 4-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B Base (MLX 4-bit)",
    description:
      "Quantized base model for lower VRAM without losing cloning support",
    category: "tts",
    capabilities: ["Voice Cloning", "4-bit"],
    size: "1.6 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-Base-8bit": {
    shortName: "0.6B Base 8-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B Base (MLX 8-bit)",
    description:
      "8-bit MLX weights for better quality while staying memory friendly",
    category: "tts",
    capabilities: ["Voice Cloning", "8-bit"],
    size: "1.9 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-Base-bf16": {
    shortName: "0.6B Base BF16",
    fullName: "Qwen3-TTS 12Hz 0.6B Base (MLX bf16)",
    description: "BF16 MLX weights for highest fidelity base voices",
    category: "tts",
    capabilities: ["Voice Cloning", "BF16"],
    size: "2.3 GB",
  },
  // TTS 0.6B CustomVoice models
  "Qwen3-TTS-12Hz-0.6B-CustomVoice": {
    shortName: "0.6B CustomVoice",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice",
    description: "Pre-trained with 9 built-in voice profiles",
    category: "tts",
    capabilities: ["Text to Speech"],
    size: "2.3 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit": {
    shortName: "0.6B Custom 4-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice (MLX 4-bit)",
    description: "Quantized CustomVoice for laptops with tight memory",
    category: "tts",
    capabilities: ["Text to Speech", "4-bit"],
    size: "1.6 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit": {
    shortName: "0.6B Custom 8-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice (MLX 8-bit)",
    description:
      "Balanced 8-bit CustomVoice for better quality with reduced VRAM",
    category: "tts",
    capabilities: ["Text to Speech", "8-bit"],
    size: "1.8 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16": {
    shortName: "0.6B Custom BF16",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice (MLX bf16)",
    description: "BF16 precision for premium CustomVoice rendering",
    category: "tts",
    capabilities: ["Text to Speech", "BF16"],
    size: "2.3 GB",
  },
  // TTS 1.7B Base models
  "Qwen3-TTS-12Hz-1.7B-Base": {
    shortName: "1.7B Base",
    fullName: "Qwen3-TTS 12Hz 1.7B Base",
    description: "Higher quality voice cloning capabilities",
    category: "tts",
    capabilities: ["Voice Cloning"],
    size: "4.2 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-Base-4bit": {
    shortName: "1.7B Base 4-bit",
    fullName: "Qwen3-TTS 12Hz 1.7B Base (MLX 4-bit)",
    description: "Quantized base model for lower-memory voice cloning",
    category: "tts",
    capabilities: ["Voice Cloning", "4-bit"],
    size: "2.2 GB",
  },
  // TTS 1.7B CustomVoice
  "Qwen3-TTS-12Hz-1.7B-CustomVoice": {
    shortName: "1.7B CustomVoice",
    fullName: "Qwen3-TTS 12Hz 1.7B CustomVoice",
    description: "Premium quality with 9 built-in voices",
    category: "tts",
    capabilities: ["Text to Speech"],
    size: "4.2 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit": {
    shortName: "1.7B Custom 4-bit",
    fullName: "Qwen3-TTS 12Hz 1.7B CustomVoice (MLX 4-bit)",
    description: "Quantized CustomVoice model with lower VRAM requirements",
    category: "tts",
    capabilities: ["Text to Speech", "4-bit"],
    size: "2.2 GB",
  },
  // TTS 1.7B VoiceDesign models
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
    shortName: "1.7B VoiceDesign",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign",
    description: "Generate voices from text descriptions",
    category: "tts",
    capabilities: ["Voice Design"],
    size: "4.2 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit": {
    shortName: "1.7B Design 4-bit",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign (MLX 4-bit)",
    description: "Quantized VoiceDesign for creative voices on 16GB devices",
    category: "tts",
    capabilities: ["Voice Design", "4-bit"],
    size: "2.2 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit": {
    shortName: "1.7B Design 8-bit",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign (MLX 8-bit)",
    description: "8-bit VoiceDesign balancing VRAM usage and quality",
    category: "tts",
    capabilities: ["Voice Design", "8-bit"],
    size: "2.9 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16": {
    shortName: "1.7B Design BF16",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign (MLX bf16)",
    description: "BF16 VoiceDesign with best timbre quality",
    category: "tts",
    capabilities: ["Voice Design", "BF16"],
    size: "4.2 GB",
  },
  // LFM2 Audio
  "LFM2-Audio-1.5B": {
    shortName: "LFM2 Audio 1.5B",
    fullName: "LFM2-Audio 1.5B by Liquid AI",
    description: "End-to-end audio foundation model",
    category: "tts",
    capabilities: ["TTS", "ASR", "Audio Chat"],
    size: "3.0 GB",
  },
  "LFM2.5-Audio-1.5B": {
    shortName: "LFM2.5 Audio 1.5B",
    fullName: "LFM2.5-Audio 1.5B by Liquid AI",
    description: "Improved end-to-end audio foundation model",
    category: "tts",
    capabilities: ["TTS", "ASR", "Audio Chat"],
    size: "3.2 GB",
  },
  "LFM2.5-Audio-1.5B-4bit": {
    shortName: "LFM2.5 Audio 1.5B 4-bit",
    fullName: "LFM2.5-Audio 1.5B (MLX 4-bit)",
    description: "Quantized LFM2.5 audio model for lower memory usage",
    category: "tts",
    capabilities: ["TTS", "ASR", "Audio Chat", "4-bit"],
    size: "0.8 GB",
  },
  // Text Chat
  "Qwen3-0.6B-4bit": {
    shortName: "Qwen3 Chat 0.6B",
    fullName: "Qwen3 0.6B (MLX 4-bit)",
    description: "Compact text-to-text model for local chat",
    category: "chat",
    capabilities: ["Text Chat", "4-bit"],
    size: "0.9 GB",
  },
  "Qwen3-1.7B-4bit": {
    shortName: "Qwen3 Chat 1.7B",
    fullName: "Qwen3 1.7B (MLX 4-bit)",
    description: "Higher-quality 1.7B chat model in 4-bit precision",
    category: "chat",
    capabilities: ["Text Chat", "4-bit"],
    size: "1.1 GB",
  },
  "Gemma-3-1b-it": {
    shortName: "Gemma 3 1B",
    fullName: "Gemma 3 1B Instruct",
    description: "Lightweight Gemma 3 instruction model for local chat",
    category: "chat",
    capabilities: ["Text Chat", "Instruction Tuned"],
    size: "2.1 GB",
  },
  "Gemma-3-4b-it": {
    shortName: "Gemma 3 4B",
    fullName: "Gemma 3 4B Instruct",
    description: "Higher-quality Gemma 3 instruction model for local chat",
    category: "chat",
    capabilities: ["Text Chat", "Instruction Tuned"],
    size: "8.0 GB",
  },
  // ASR 0.6B models
  "Qwen3-ASR-0.6B": {
    shortName: "ASR 0.6B",
    fullName: "Qwen3-ASR 0.6B",
    description: "Fast speech-to-text, 52 languages",
    category: "asr",
    capabilities: ["Transcription"],
    size: "1.8 GB",
  },
  "Qwen3-ASR-0.6B-4bit": {
    shortName: "ASR 0.6B 4-bit",
    fullName: "Qwen3-ASR 0.6B (MLX 4-bit)",
    description: "Lightweight ASR for real-time transcription on smaller GPUs",
    category: "asr",
    capabilities: ["Transcription", "4-bit"],
    size: "0.7 GB",
  },
  "Qwen3-ASR-0.6B-8bit": {
    shortName: "ASR 0.6B 8-bit",
    fullName: "Qwen3-ASR 0.6B (MLX 8-bit)",
    description: "8-bit ASR with higher accuracy and modest footprint",
    category: "asr",
    capabilities: ["Transcription", "8-bit"],
    size: "0.9 GB",
  },
  "Qwen3-ASR-0.6B-bf16": {
    shortName: "ASR 0.6B BF16",
    fullName: "Qwen3-ASR 0.6B (MLX bf16)",
    description: "BF16 precision ASR for top accuracy",
    category: "asr",
    capabilities: ["Transcription", "BF16"],
    size: "1.5 GB",
  },
  // ASR 1.7B models
  "Qwen3-ASR-1.7B": {
    shortName: "ASR 1.7B",
    fullName: "Qwen3-ASR 1.7B",
    description: "High-quality speech-to-text, 52 languages",
    category: "asr",
    capabilities: ["Transcription"],
    size: "4.4 GB",
  },
  "Qwen3-ASR-1.7B-4bit": {
    shortName: "ASR 1.7B 4-bit",
    fullName: "Qwen3-ASR 1.7B (MLX 4-bit)",
    description: "Quantized 1.7B ASR for RTX 4090 / M3 workloads",
    category: "asr",
    capabilities: ["Transcription", "4-bit"],
    size: "1.5 GB",
  },
  "Qwen3-ASR-1.7B-8bit": {
    shortName: "ASR 1.7B 8-bit",
    fullName: "Qwen3-ASR 1.7B (MLX 8-bit)",
    description: "8-bit ASR for high fidelity transcripts on Apple Silicon",
    category: "asr",
    capabilities: ["Transcription", "8-bit"],
    size: "2.3 GB",
  },
  "Qwen3-ASR-1.7B-bf16": {
    shortName: "ASR 1.7B BF16",
    fullName: "Qwen3-ASR 1.7B (MLX bf16)",
    description: "BF16 ASR providing maximum quality and accuracy",
    category: "asr",
    capabilities: ["Transcription", "BF16"],
    size: "3.8 GB",
  },
  "Qwen3-ForcedAligner-0.6B": {
    shortName: "ForcedAligner 0.6B",
    fullName: "Qwen3-ForcedAligner 0.6B",
    description: "Aligns transcript text to precise speech timestamps",
    category: "asr",
    capabilities: ["Forced Alignment", "Word timestamps"],
    size: "1.7 GB",
  },
  "Qwen3-ForcedAligner-0.6B-4bit": {
    shortName: "ForcedAligner 0.6B 4-bit",
    fullName: "Qwen3-ForcedAligner 0.6B (MLX 4-bit)",
    description: "Quantized forced aligner for low-memory alignment workflows",
    category: "asr",
    capabilities: ["Forced Alignment", "Word timestamps", "4-bit"],
    size: "0.7 GB",
  },
  "Parakeet-TDT-0.6B-v2": {
    shortName: "Parakeet v2",
    fullName: "Parakeet-TDT 0.6B v2",
    description: "English FastConformer-TDT ASR model in .nemo format",
    category: "asr",
    capabilities: ["Transcription", "Word timestamps"],
    size: "4.6 GB",
  },
  "Parakeet-TDT-0.6B-v2-4bit": {
    shortName: "Parakeet v2 4-bit",
    fullName: "Parakeet-TDT 0.6B v2 (MLX 4-bit)",
    description: "Quantized English Parakeet model for lower-memory ASR",
    category: "asr",
    capabilities: ["Transcription", "Word timestamps", "4-bit"],
    size: "2.5 GB",
  },
  "Parakeet-TDT-0.6B-v3": {
    shortName: "Parakeet v3",
    fullName: "Parakeet-TDT 0.6B v3",
    description: "Multilingual FastConformer-TDT ASR model in .nemo format",
    category: "asr",
    capabilities: ["Transcription", "25 EU languages"],
    size: "9.3 GB",
  },
  "Parakeet-TDT-0.6B-v3-4bit": {
    shortName: "Parakeet v3 4-bit",
    fullName: "Parakeet-TDT 0.6B v3 (MLX 4-bit)",
    description: "Quantized multilingual Parakeet model for lower-memory ASR",
    category: "asr",
    capabilities: ["Transcription", "Word timestamps", "4-bit"],
    size: "2.9 GB",
  },
  "diar_streaming_sortformer_4spk-v2.1": {
    shortName: "Sortformer 4spk",
    fullName: "Streaming Sortformer 4spk v2.1",
    description: "Streaming speaker diarization model from NVIDIA in .nemo format",
    category: "asr",
    capabilities: ["Diarization", "Up to 4 speakers", "Streaming"],
    size: "0.5 GB",
  },
  // Voxtral
  "Voxtral-Mini-4B-Realtime-2602": {
    shortName: "Voxtral 4B",
    fullName: "Voxtral Mini 4B Realtime",
    description: "Realtime streaming ASR from Mistral AI",
    category: "asr",
    capabilities: ["Transcription", "Realtime"],
    size: "8.0 GB",
  },
};

function parseSize(sizeStr: string): number {
  const match = sizeStr.match(/^([\d.]+)\s*(GB|MB|KB|B)?$/i);
  if (!match) return 0;
  const value = parseFloat(match[1]);
  const unit = (match[2] || "B").toUpperCase();
  const multipliers: Record<string, number> = {
    B: 1,
    KB: 1024,
    MB: 1024 * 1024,
    GB: 1024 * 1024 * 1024,
  };
  return value * (multipliers[unit] || 1);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function getStatusLabel(status: ModelInfo["status"]): string {
  switch (status) {
    case "ready":
      return "Loaded";
    case "downloaded":
      return "Downloaded";
    case "downloading":
      return "Downloading";
    case "loading":
      return "Loading";
    case "not_downloaded":
      return "Not Downloaded";
    case "error":
      return "Error";
    default:
      return status;
  }
}

function getStatusDotClass(status: ModelInfo["status"]): string {
  switch (status) {
    case "ready":
      return "bg-[var(--status-positive-solid)]";
    case "downloaded":
      return "bg-[var(--text-secondary)]";
    case "downloading":
    case "loading":
      return "bg-[var(--status-warning-text)]";
    case "error":
      return "bg-[var(--danger-text)]";
    default:
      return "bg-[var(--text-subtle)]";
  }
}

function getStatusBadgeClass(status: ModelInfo["status"]): string {
  switch (status) {
    case "ready":
      return "bg-[var(--status-positive-bg)] border-[var(--status-positive-border)] text-[var(--status-positive-text)]";
    case "downloaded":
      return "bg-[var(--bg-surface-2)] border-[var(--border-strong)] text-[var(--text-secondary)]";
    case "downloading":
    case "loading":
      return "bg-[var(--status-warning-bg)] border-[var(--status-warning-border)] text-[var(--status-warning-text)]";
    case "error":
      return "bg-[var(--danger-bg)] border-[var(--danger-border)] text-[var(--danger-text)]";
    default:
      return "bg-[var(--bg-surface-2)] border-[var(--border-muted)] text-[var(--text-muted)]";
  }
}

function getCategoryLabel(category: CategoryType): string {
  switch (category) {
    case "tts":
      return "Text to Speech";
    case "asr":
      return "Transcription";
    case "chat":
      return "Chat";
    default:
      return "Unknown";
  }
}

function getPrecisionLabel(capabilities: string[]): string | null {
  if (capabilities.some((cap) => /4-bit/i.test(cap))) return "4-bit";
  if (capabilities.some((cap) => /8-bit/i.test(cap))) return "8-bit";
  if (capabilities.some((cap) => /bf16/i.test(cap))) return "BF16";
  return null;
}

function requiresManualDownload(variant: string): boolean {
  return variant === "Gemma-3-1b-it";
}

export function MyModelsPage({
  models,
  loading,
  downloadProgress,
  onDownload,
  onCancelDownload,
  onLoad,
  onUnload,
  onDelete,
  onRefresh,
}: MyModelsPageProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<FilterType>("all");
  const [categoryFilter, setCategoryFilter] = useState<CategoryType>("all");
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const filteredModels = useMemo(() => {
    return models
      .filter((m) => !m.variant.includes("Tokenizer"))
      .filter((m) => {
        const details = MODEL_DETAILS[m.variant];
        if (!details) return false;

        // Search filter
        if (searchQuery) {
          const query = searchQuery.toLowerCase();
          const matchesSearch =
            details.shortName.toLowerCase().includes(query) ||
            details.fullName.toLowerCase().includes(query) ||
            details.description.toLowerCase().includes(query) ||
            details.capabilities.some((c) => c.toLowerCase().includes(query));
          if (!matchesSearch) return false;
        }

        // Status filter
        if (statusFilter !== "all") {
          if (statusFilter === "downloaded" && m.status !== "downloaded")
            return false;
          if (statusFilter === "loaded" && m.status !== "ready") return false;
          if (
            statusFilter === "not_downloaded" &&
            m.status !== "not_downloaded"
          )
            return false;
        }

        // Category filter
        if (categoryFilter !== "all" && details.category !== categoryFilter) {
          return false;
        }

        return true;
      })
      .sort((a, b) => {
        // Stable sort independent of status so cards do not jump while downloading/loading.
        const sizeA = parseSize(MODEL_DETAILS[a.variant]?.size || "0");
        const sizeB = parseSize(MODEL_DETAILS[b.variant]?.size || "0");
        if (sizeA !== sizeB) return sizeA - sizeB;
        return a.variant.localeCompare(b.variant);
      });
  }, [models, searchQuery, statusFilter, categoryFilter]);

  const stats = useMemo(() => {
    const visibleModels = models.filter(
      (m) => !m.variant.includes("Tokenizer") && MODEL_DETAILS[m.variant],
    );
    return {
      total: visibleModels.length,
      loaded: visibleModels.filter((m) => m.status === "ready").length,
      downloaded: visibleModels.filter(
        (m) => m.status === "downloaded" || m.status === "ready",
      ).length,
      totalSize: visibleModels
        .filter((m) => m.status === "downloaded" || m.status === "ready")
        .reduce(
          (acc, m) => acc + parseSize(MODEL_DETAILS[m.variant]?.size || "0"),
          0,
        ),
    };
  }, [models]);

  const handleDelete = (variant: string) => {
    setConfirmDelete(null);
    onDelete(variant);
  };

  const destructiveDeleteButtonClass =
    "flex items-center gap-1.5 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-2.5 py-1.5 text-xs font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]";

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-center gap-2 py-24 text-[var(--text-muted)]">
          <Loader2 className="w-4 h-4 animate-spin" />
          <p className="text-sm">Loading models...</p>
        </div>
      </div>
    );
  }

  const hasActiveFilters =
    searchQuery.trim().length > 0 ||
    statusFilter !== "all" ||
    categoryFilter !== "all";

  return (
    <div className="max-w-6xl mx-auto space-y-4">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-xl font-semibold text-[var(--text-primary)]">
            Models
          </h1>
          <p className="mt-1 text-sm text-[var(--text-muted)]">
            Download, load, and remove local models
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {onRefresh && (
            <button
              onClick={async () => {
                setIsRefreshing(true);
                await onRefresh();
                setIsRefreshing(false);
              }}
              disabled={isRefreshing}
              className="flex items-center gap-1.5 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2 text-xs font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-surface-2)] disabled:opacity-60"
              title="Refresh models"
            >
              <RefreshCw
                className={clsx(
                  "w-3.5 h-3.5",
                  isRefreshing && "animate-spin",
                )}
              />
              Refresh
            </button>
          )}

          <div className="flex items-center gap-2 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2">
            <HardDrive className="w-4 h-4 text-[var(--text-subtle)]" />
            <div className="text-sm">
              <span className="font-medium text-[var(--text-primary)]">
                {formatBytes(stats.totalSize)}
              </span>
              <span className="ml-1 text-[var(--text-muted)]">used</span>
            </div>
          </div>

          <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2 text-sm">
            <span className="font-medium text-[var(--text-primary)]">
              {stats.loaded}
            </span>
            <span className="ml-1 text-[var(--text-muted)]">loaded</span>
            <span className="mx-1 text-[var(--text-subtle)]">/</span>
            <span className="text-[var(--text-muted)]">{stats.downloaded}</span>
            <span className="ml-1 text-[var(--text-subtle)]">downloaded</span>
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3 sm:p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--text-subtle)]" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search models..."
              className="w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] py-2.5 pl-10 pr-3 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-subtle)] focus:border-[var(--border-strong)] focus:outline-none"
            />
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {[
              { id: "all" as FilterType, label: "All" },
              { id: "loaded" as FilterType, label: "Loaded" },
              { id: "downloaded" as FilterType, label: "Downloaded" },
              { id: "not_downloaded" as FilterType, label: "Not downloaded" },
            ].map((option) => (
              <button
                key={option.id}
                onClick={() => setStatusFilter(option.id)}
                className={clsx(
                  "rounded-md border px-3 py-1.5 text-xs font-medium transition-colors",
                  statusFilter === option.id
                    ? "border-[var(--border-strong)] bg-[var(--bg-surface-3)] text-[var(--text-primary)]"
                    : "border-[var(--border-muted)] bg-[var(--bg-surface-1)] text-[var(--text-muted)] hover:text-[var(--text-primary)]",
                )}
              >
                {option.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <select
              value={categoryFilter}
              onChange={(event) =>
                setCategoryFilter(event.target.value as CategoryType)
              }
              className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] focus:border-[var(--border-strong)] focus:outline-none"
              aria-label="Filter models by category"
            >
              <option value="all">All categories</option>
              <option value="tts">Text to Speech</option>
              <option value="asr">Transcription</option>
              <option value="chat">Chat</option>
            </select>

            {hasActiveFilters && (
              <button
                onClick={() => {
                  setSearchQuery("");
                  setStatusFilter("all");
                  setCategoryFilter("all");
                }}
                className="flex items-center gap-1 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1.5 text-xs text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
              >
                <X className="h-3.5 w-3.5" />
                Clear
              </button>
            )}
          </div>
        </div>
      </div>

      {filteredModels.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] py-16 text-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-2)]">
            <HardDrive className="h-5 w-5 text-[var(--text-subtle)]" />
          </div>
          <h3 className="text-base font-medium text-[var(--text-primary)]">
            No models found
          </h3>
          <p className="mt-1 max-w-sm text-sm text-[var(--text-muted)]">
            {hasActiveFilters
              ? "Try adjusting your filters."
              : "Download a model to get started."}
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {filteredModels.map((model) => {
            const details = MODEL_DETAILS[model.variant];
            if (!details) return null;

            const displayName = withQwen3Prefix(
              details.shortName,
              model.variant,
            );
            const precisionLabel = getPrecisionLabel(details.capabilities);
            const isDownloading = model.status === "downloading";
            const isLoading = model.status === "loading";
            const isReady = model.status === "ready";
            const isDownloaded = model.status === "downloaded";
            const progressValue = downloadProgress[model.variant];
            const progress =
              progressValue?.percent ?? model.download_progress ?? 0;

            return (
              <div
                key={model.variant}
                className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3 sm:p-4"
              >
                <div className="flex flex-col gap-3 md:flex-row md:items-center">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      {isDownloading || isLoading ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin text-[var(--status-warning-text)]" />
                      ) : (
                        <span
                          className={clsx(
                            "h-2 w-2 rounded-full",
                            getStatusDotClass(model.status),
                          )}
                        />
                      )}
                      <h3 className="truncate text-sm font-medium text-[var(--text-primary)]">
                        {displayName}
                      </h3>
                      <span
                        className={clsx(
                          "rounded border px-2 py-0.5 text-[11px] font-medium",
                          getStatusBadgeClass(model.status),
                        )}
                      >
                        {getStatusLabel(model.status)}
                      </span>
                    </div>

                    <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-[var(--text-subtle)]">
                      <span>{getCategoryLabel(details.category)}</span>
                      {precisionLabel && (
                        <>
                          <span aria-hidden>•</span>
                          <span>{precisionLabel}</span>
                        </>
                      )}
                      <span aria-hidden>•</span>
                      <span>{details.size}</span>
                    </div>

                    {isDownloading && (
                      <div className="mt-2 flex items-center gap-2">
                        <div className="h-1.5 w-full max-w-[220px] overflow-hidden rounded-full bg-[var(--bg-surface-3)]">
                          <div
                            className="h-full rounded-full bg-[var(--accent-solid)] transition-all duration-300"
                            style={{ width: `${progress}%` }}
                          />
                        </div>
                        <span className="text-xs text-[var(--text-muted)]">
                          {Math.round(progress)}%
                        </span>
                      </div>
                    )}
                  </div>

                  <div className="flex flex-wrap items-center gap-1.5">
                    {model.status === "not_downloaded" &&
                      (requiresManualDownload(model.variant) ? (
                        <button
                          className="flex items-center gap-1.5 rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-muted)] disabled:cursor-not-allowed disabled:opacity-60"
                          disabled
                          title="Manual download required. See docs/user/manual-gemma-3-1b-download.md."
                        >
                          <Download className="h-3.5 w-3.5" />
                          Manual download
                        </button>
                      ) : (
                        <button
                          onClick={() => onDownload(model.variant)}
                          className="flex items-center gap-1.5 rounded-md bg-[var(--accent-solid)] px-3 py-1.5 text-xs font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90"
                        >
                          <Download className="h-3.5 w-3.5" />
                          Download
                        </button>
                      ))}

                    {isDownloading && onCancelDownload && (
                      <button
                        onClick={() => onCancelDownload(model.variant)}
                        className="flex items-center gap-1 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-2.5 py-1.5 text-xs font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                      >
                        <X className="h-3.5 w-3.5" />
                        Cancel
                      </button>
                    )}

                    {isDownloaded && (
                      <>
                        <button
                          onClick={() => onLoad(model.variant)}
                          className="flex items-center gap-1.5 rounded-md bg-[var(--accent-solid)] px-3 py-1.5 text-xs font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90"
                        >
                          <Play className="h-3.5 w-3.5" />
                          Load
                        </button>
                        {confirmDelete === model.variant ? (
                          <>
                            <button
                              onClick={() => setConfirmDelete(null)}
                              className="flex items-center gap-1.5 rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-2.5 py-1.5 text-xs font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-surface-3)]"
                            >
                              <X className="h-3.5 w-3.5" />
                              Cancel
                            </button>
                            <button
                              onClick={() => handleDelete(model.variant)}
                              className={destructiveDeleteButtonClass}
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                              Confirm
                            </button>
                          </>
                        ) : (
                          <button
                            onClick={() => setConfirmDelete(model.variant)}
                            className={destructiveDeleteButtonClass}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                            Delete
                          </button>
                        )}
                      </>
                    )}

                    {isReady && (
                      <>
                        <button
                          onClick={() => onUnload(model.variant)}
                          className="flex items-center gap-1.5 rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-surface-3)]"
                        >
                          <Square className="h-3.5 w-3.5" />
                          Unload
                        </button>
                        {confirmDelete === model.variant ? (
                          <>
                            <button
                              onClick={() => setConfirmDelete(null)}
                              className="flex items-center gap-1.5 rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-2.5 py-1.5 text-xs font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-surface-3)]"
                            >
                              <X className="h-3.5 w-3.5" />
                              Cancel
                            </button>
                            <button
                              onClick={() => handleDelete(model.variant)}
                              className={destructiveDeleteButtonClass}
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                              Confirm
                            </button>
                          </>
                        ) : (
                          <button
                            onClick={() => setConfirmDelete(model.variant)}
                            className={destructiveDeleteButtonClass}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                            Delete
                          </button>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
