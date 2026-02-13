import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Download,
  Play,
  Square,
  Trash2,
  HardDrive,
  Search,
  Filter,
  ChevronDown,
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
  // TTS 1.7B CustomVoice
  "Qwen3-TTS-12Hz-1.7B-CustomVoice": {
    shortName: "1.7B CustomVoice",
    fullName: "Qwen3-TTS 12Hz 1.7B CustomVoice",
    description: "Premium quality with 9 built-in voices",
    category: "tts",
    capabilities: ["Text to Speech"],
    size: "4.2 GB",
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
  // Text Chat
  "Qwen3-0.6B-4bit": {
    shortName: "Qwen3 Chat 0.6B",
    fullName: "Qwen3 0.6B (MLX 4-bit)",
    description: "Compact text-to-text model for local chat",
    category: "chat",
    capabilities: ["Text Chat", "4-bit"],
    size: "0.9 GB",
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

function getStatusColor(status: ModelInfo["status"]): string {
  switch (status) {
    case "ready":
      return "bg-white";
    case "downloaded":
      return "bg-gray-300";
    case "downloading":
    case "loading":
      return "bg-gray-400";
    case "error":
      return "bg-gray-600";
    default:
      return "bg-gray-500";
  }
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
  const [showFilters, setShowFilters] = useState(false);
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
        // Sort: loaded first, then downloaded, then not downloaded
        const statusOrder = {
          ready: 0,
          loading: 1,
          downloaded: 2,
          downloading: 3,
          not_downloaded: 4,
          error: 5,
        };
        const statusDiff = statusOrder[a.status] - statusOrder[b.status];
        if (statusDiff !== 0) return statusDiff;

        // Then sort by size
        const sizeA = parseSize(MODEL_DETAILS[a.variant]?.size || "0");
        const sizeB = parseSize(MODEL_DETAILS[b.variant]?.size || "0");
        return sizeA - sizeB;
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
    "flex items-center gap-1.5 px-2.5 py-1.5 rounded border border-red-500/45 bg-red-500/15 text-red-300 text-xs font-medium hover:bg-red-500/25 hover:text-red-200 transition-colors";

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
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
        <div>
          <h1 className="text-xl font-semibold text-white">Models</h1>
          <p className="text-sm text-gray-500 mt-1">
            Manage your downloaded models
          </p>
        </div>

        {/* Stats and Refresh */}
        <div className="flex items-center gap-4">
          {onRefresh && (
            <button
              onClick={async () => {
                setIsRefreshing(true);
                await onRefresh();
                setIsRefreshing(false);
              }}
              disabled={isRefreshing}
              className="p-2 rounded-lg bg-[#161616] border border-[#2a2a2a] hover:bg-[#1f1f1f] transition-colors disabled:opacity-50"
              title="Refresh models"
            >
              <RefreshCw
                className={clsx(
                  "w-4 h-4 text-gray-400",
                  isRefreshing && "animate-spin",
                )}
              />
            </button>
          )}
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#141414] border border-[#262626]">
            <HardDrive className="w-4 h-4 text-gray-500" />
            <div className="text-sm">
              <span className="text-white font-medium">
                {formatBytes(stats.totalSize)}
              </span>
              <span className="text-gray-500 ml-1">used</span>
            </div>
          </div>
          <div className="text-sm text-gray-500">
            <span className="text-white font-medium">{stats.loaded}</span>
            <span className="mx-1">/</span>
            <span>{stats.downloaded} loaded</span>
          </div>
        </div>
      </div>

      {/* Search and filters */}
      <div className="flex flex-col sm:flex-row gap-3 mb-6">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search models..."
            className="w-full pl-10 pr-4 py-2.5 bg-[#161616] border border-[#2a2a2a] rounded-lg text-sm text-white placeholder-gray-600 focus:outline-none focus:border-[#404040]"
          />
        </div>

        {/* Filter button */}
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={clsx(
            "flex items-center gap-2 px-4 py-2.5 rounded-lg border text-sm transition-colors",
            showFilters
              ? "bg-[#1a1a1a] border-[#404040] text-white"
              : "bg-[#141414] border-[#262626] text-gray-400 hover:text-white",
          )}
        >
          <Filter className="w-4 h-4" />
          Filters
          {(statusFilter !== "all" || categoryFilter !== "all") && (
            <span className="w-1.5 h-1.5 rounded-full bg-white" />
          )}
          <ChevronDown
            className={clsx(
              "w-3.5 h-3.5 transition-transform",
              showFilters && "rotate-180",
            )}
          />
        </button>
      </div>

      {/* Filter options */}
      <AnimatePresence>
        {showFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden mb-6"
          >
            <div className="p-4 rounded-lg bg-[#141414] border border-[#262626]">
              <div className="flex flex-wrap gap-6">
                {/* Status filter */}
                <div>
                  <label className="block text-xs text-gray-500 mb-2">
                    Status
                  </label>
                  <div className="flex gap-2">
                    {[
                      { id: "all" as FilterType, label: "All" },
                      { id: "loaded" as FilterType, label: "Loaded" },
                      { id: "downloaded" as FilterType, label: "Downloaded" },
                      {
                        id: "not_downloaded" as FilterType,
                        label: "Not Downloaded",
                      },
                    ].map((option) => (
                      <button
                        key={option.id}
                        onClick={() => setStatusFilter(option.id)}
                        className={clsx(
                          "px-3 py-1.5 rounded text-xs transition-colors",
                          statusFilter === option.id
                            ? "bg-white text-black"
                            : "bg-[#1f1f1f] text-gray-400 hover:text-white",
                        )}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Category filter */}
                <div>
                  <label className="block text-xs text-gray-500 mb-2">
                    Category
                  </label>
                  <div className="flex gap-2">
                    {[
                      { id: "all" as CategoryType, label: "All" },
                      { id: "tts" as CategoryType, label: "Text to Speech" },
                      { id: "asr" as CategoryType, label: "Transcription" },
                      { id: "chat" as CategoryType, label: "Chat" },
                    ].map((option) => (
                      <button
                        key={option.id}
                        onClick={() => setCategoryFilter(option.id)}
                        className={clsx(
                          "px-3 py-1.5 rounded text-xs transition-colors",
                          categoryFilter === option.id
                            ? "bg-white text-black"
                            : "bg-[#1f1f1f] text-gray-400 hover:text-white",
                        )}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Models grid */}
      {filteredModels.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="w-16 h-16 rounded-full bg-[#161616] flex items-center justify-center mb-4">
            <HardDrive className="w-7 h-7 text-gray-600" />
          </div>
          <h3 className="text-base font-medium text-gray-300 mb-1">
            No models found
          </h3>
          <p className="text-sm text-gray-600 max-w-xs">
            {searchQuery || statusFilter !== "all" || categoryFilter !== "all"
              ? "Try adjusting your search or filters"
              : "Download models to get started"}
          </p>
        </div>
      ) : (
        <div className="grid gap-2">
          {filteredModels.map((model) => {
            const details = MODEL_DETAILS[model.variant];
            if (!details) return null;
            const displayName = withQwen3Prefix(
              details.shortName,
              model.variant,
            );

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
                className="p-3 rounded-lg bg-[#141414] border border-[#262626] hover:border-[#333333] transition-colors"
              >
                <div className="flex items-center gap-3">
                  {/* Status indicator */}
                  <div className="flex-shrink-0">
                    {isDownloading || isLoading ? (
                      <Loader2 className="w-4 h-4 text-white animate-spin" />
                    ) : (
                      <div
                        className={clsx(
                          "w-2 h-2 rounded-full",
                          getStatusColor(model.status),
                        )}
                      />
                    )}
                  </div>

                  {/* Model info - more horizontal layout */}
                  <div className="flex-1 min-w-0 flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
                    <div className="flex items-center gap-2 min-w-0">
                      <h3 className="text-sm font-medium text-white truncate">
                        {displayName}
                      </h3>
                      <span
                        className={clsx(
                          "text-[10px] px-1.5 py-0.5 rounded whitespace-nowrap bg-[#1a1a1a] border border-[#2a2a2a]",
                          isReady
                            ? "text-white"
                            : isDownloaded
                              ? "text-gray-300"
                              : "text-gray-500",
                        )}
                      >
                        {getStatusLabel(model.status)}
                      </span>
                    </div>
                    <p className="text-xs text-gray-500 truncate hidden md:block">
                      {details.description}
                    </p>
                    <div className="flex items-center gap-3 sm:ml-auto">
                      <div className="flex items-center gap-1.5">
                        {details.capabilities.map((cap) => (
                          <span
                            key={cap}
                            className="text-[10px] px-1.5 py-0.5 rounded bg-[#1f1f1f] text-gray-400 whitespace-nowrap"
                          >
                            {cap}
                          </span>
                        ))}
                      </div>
                      <span className="text-xs text-gray-600 whitespace-nowrap">
                        {details.size}
                      </span>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {model.status === "not_downloaded" && (
                      <button
                        onClick={() => onDownload(model.variant)}
                        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded bg-white text-black text-xs font-medium hover:bg-gray-200 transition-colors"
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span className="hidden sm:inline">Download</span>
                      </button>
                    )}

                    {isDownloading && (
                      <div className="flex items-center gap-2 px-2 py-1.5">
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-gray-500">
                            {Math.round(progress)}%
                          </span>
                          <div className="w-16 h-1 bg-[#1f1f1f] rounded-full overflow-hidden">
                            <div
                              className="h-full bg-white rounded-full transition-all duration-300"
                              style={{ width: `${progress}%` }}
                            />
                          </div>
                        </div>
                        {onCancelDownload && (
                          <button
                            onClick={() => onCancelDownload(model.variant)}
                            className="flex items-center gap-1 px-2 py-1 rounded bg-[var(--danger-bg)] border border-[var(--danger-border)] text-[11px] text-[var(--danger-text)] hover:bg-[var(--danger-bg-hover)] transition-colors"
                          >
                            <X className="w-3 h-3" />
                            Cancel
                          </button>
                        )}
                      </div>
                    )}

                    {isDownloaded && (
                      <>
                        <button
                          onClick={() => onLoad(model.variant)}
                          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded bg-white text-black text-xs font-medium hover:bg-gray-200 transition-colors"
                        >
                          <Play className="w-3.5 h-3.5" />
                          <span className="hidden sm:inline">Load</span>
                        </button>
                        {confirmDelete === model.variant ? (
                          <div className="flex items-center gap-1.5">
                            <button
                              onClick={() => setConfirmDelete(null)}
                              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded bg-[#262626] border border-[#303030] text-gray-300 text-xs font-medium hover:text-white hover:bg-[#2e2e2e] transition-colors"
                              title="Cancel"
                            >
                              <X className="w-3.5 h-3.5" />
                              Cancel
                            </button>
                            <button
                              onClick={() => handleDelete(model.variant)}
                              className={destructiveDeleteButtonClass}
                              title="Confirm delete"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                              Confirm Delete
                            </button>
                          </div>
                        ) : (
                          <button
                            onClick={() => setConfirmDelete(model.variant)}
                            className={destructiveDeleteButtonClass}
                            title="Delete model"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                            <span className="hidden sm:inline">Delete</span>
                          </button>
                        )}
                      </>
                    )}

                    {isReady && (
                      <>
                        <button
                          onClick={() => onUnload(model.variant)}
                          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded bg-[#1f1f1f] border border-[#262626] text-white text-xs font-medium hover:bg-[#262626] transition-colors"
                        >
                          <Square className="w-3.5 h-3.5" />
                          <span className="hidden sm:inline">Unload</span>
                        </button>
                        {confirmDelete === model.variant ? (
                          <div className="flex items-center gap-1.5">
                            <button
                              onClick={() => setConfirmDelete(null)}
                              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded bg-[#262626] border border-[#303030] text-gray-300 text-xs font-medium hover:text-white hover:bg-[#2e2e2e] transition-colors"
                              title="Cancel"
                            >
                              <X className="w-3.5 h-3.5" />
                              Cancel
                            </button>
                            <button
                              onClick={() => handleDelete(model.variant)}
                              className={destructiveDeleteButtonClass}
                              title="Confirm delete"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                              Confirm Delete
                            </button>
                          </div>
                        ) : (
                          <button
                            onClick={() => setConfirmDelete(model.variant)}
                            className={destructiveDeleteButtonClass}
                            title="Delete model"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                            <span className="hidden sm:inline">Delete</span>
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
