import { useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  CheckCircle2,
  Download,
  Loader2,
  Play,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { ModelInfo } from "../api";
import { TranscriptionPlayground } from "../components/TranscriptionPlayground";
import { VIEW_CONFIGS } from "../types";

interface TranscriptionPageProps {
  models: ModelInfo[];
  selectedModel: string | null;
  loading: boolean;
  downloadProgress: Record<
    string,
    {
      percent: number;
      currentFile: string;
      status: string;
      downloadedBytes: number;
      totalBytes: number;
    }
  >;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onSelect: (variant: string) => void;
  onError: (message: string) => void;
}

export function TranscriptionPage({
  models,
  selectedModel,
  loading,
  downloadProgress,
  onDownload,
  onCancelDownload,
  onLoad,
  onUnload,
  onDelete,
  onSelect,
  onError,
}: TranscriptionPageProps) {
  const viewConfig = VIEW_CONFIGS.transcription;
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [modalIntentModel, setModalIntentModel] = useState<string | null>(null);
  const [autoCloseOnIntentReady, setAutoCloseOnIntentReady] = useState(false);
  const [pendingDeleteVariant, setPendingDeleteVariant] = useState<string | null>(
    null,
  );

  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024)
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  const transcriptionModels = useMemo(
    () =>
      models
        .filter((model) => !model.variant.includes("Tokenizer"))
        .filter((model) => viewConfig.modelFilter(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models, viewConfig],
  );

  const preferredModelOrder = [
    "Qwen3-ASR-0.6B-4bit",
    "Qwen3-ASR-0.6B",
    "Parakeet-TDT-0.6B-v3",
    "Parakeet-TDT-0.6B-v2",
    "Qwen3-ASR-1.7B-4bit",
    "Qwen3-ASR-1.7B",
  ];

  const resolvedSelectedModel = (() => {
    if (
      selectedModel &&
      transcriptionModels.some((model) => model.variant === selectedModel)
    ) {
      return selectedModel;
    }

    for (const variant of preferredModelOrder) {
      const readyPreferred = transcriptionModels.find(
        (model) => model.variant === variant && model.status === "ready",
      );
      if (readyPreferred) {
        return readyPreferred.variant;
      }
    }

    const readyModel = transcriptionModels.find((model) => model.status === "ready");
    if (readyModel) {
      return readyModel.variant;
    }

    for (const variant of preferredModelOrder) {
      const preferred = transcriptionModels.find((model) => model.variant === variant);
      if (preferred) {
        return preferred.variant;
      }
    }

    return transcriptionModels[0]?.variant ?? null;
  })();

  const selectedModelInfo =
    transcriptionModels.find((model) => model.variant === resolvedSelectedModel) ??
    null;
  const selectedModelReady = selectedModelInfo?.status === "ready";

  const closeModelModal = () => {
    setIsModelModalOpen(false);
    setAutoCloseOnIntentReady(false);
    setPendingDeleteVariant(null);
  };

  useEffect(() => {
    if (!isModelModalOpen || !modalIntentModel || !autoCloseOnIntentReady) {
      return;
    }
    const targetModel = transcriptionModels.find(
      (model) => model.variant === modalIntentModel,
    );
    if (targetModel?.status === "ready") {
      closeModelModal();
    }
  }, [
    autoCloseOnIntentReady,
    closeModelModal,
    isModelModalOpen,
    modalIntentModel,
    transcriptionModels,
  ]);

  const openModelManager = () => {
    setModalIntentModel(resolvedSelectedModel);
    setAutoCloseOnIntentReady(false);
    setIsModelModalOpen(true);
  };

  const getStatusLabel = (status: ModelInfo["status"]): string => {
    switch (status) {
      case "ready":
        return "Loaded";
      case "loading":
        return "Loading";
      case "downloading":
        return "Downloading";
      case "downloaded":
        return "Downloaded";
      case "not_downloaded":
        return "Not downloaded";
      case "error":
        return "Error";
      default:
        return status;
    }
  };

  const getStatusClass = (status: ModelInfo["status"]): string => {
    switch (status) {
      case "ready":
        return "bg-emerald-500/15 border-emerald-500/40 text-emerald-300";
      case "loading":
      case "downloading":
        return "bg-blue-500/15 border-blue-500/40 text-blue-300";
      case "downloaded":
        return "bg-white/10 border-white/20 text-gray-300";
      case "error":
        return "bg-red-500/15 border-red-500/40 text-red-300";
      default:
        return "bg-[#1c1c1c] border-[#2a2a2a] text-gray-500";
    }
  };

  const renderPrimaryAction = (
    model: ModelInfo,
    isActiveModel: boolean,
  ): JSX.Element | null => {
    if (model.status === "downloading" && onCancelDownload) {
      return (
        <button
          onClick={(event) => {
            event.stopPropagation();
            onCancelDownload(model.variant);
          }}
          className="btn btn-danger text-xs"
        >
          <X className="w-3.5 h-3.5" />
          Cancel
        </button>
      );
    }

    if (model.status === "not_downloaded" || model.status === "error") {
      return (
        <button
          onClick={(event) => {
            event.stopPropagation();
            onDownload(model.variant);
          }}
          className="btn btn-primary text-xs"
        >
          <Download className="w-3.5 h-3.5" />
          Download
        </button>
      );
    }

    if (model.status === "downloaded") {
      return (
        <button
          onClick={(event) => {
            event.stopPropagation();
            onLoad(model.variant);
          }}
          className="btn btn-primary text-xs"
        >
          <Play className="w-3.5 h-3.5" />
          Load
        </button>
      );
    }

    if (model.status === "loading") {
      return (
        <button className="btn btn-secondary text-xs" disabled>
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
          Loading
        </button>
      );
    }

    if (model.status === "ready") {
      if (isActiveModel) {
        return (
          <button className="btn btn-secondary text-xs" disabled>
            <CheckCircle2 className="w-3.5 h-3.5" />
            Active
          </button>
        );
      }
      return (
        <button
          onClick={(event) => {
            event.stopPropagation();
            onSelect(model.variant);
            closeModelModal();
          }}
          className="btn btn-primary text-xs"
        >
          <CheckCircle2 className="w-3.5 h-3.5" />
          Use Model
        </button>
      );
    }

    return null;
  };

  const activeReadyModelVariant =
    transcriptionModels.find((model) => model.status === "ready")?.variant ?? null;

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-white">Transcription</h1>
      </div>

      <TranscriptionPlayground
        selectedModel={resolvedSelectedModel}
        selectedModelReady={selectedModelReady}
        modelLabel={selectedModelInfo?.variant ?? null}
        onOpenModelManager={openModelManager}
        onModelRequired={() => {
          setModalIntentModel(resolvedSelectedModel);
          setAutoCloseOnIntentReady(true);
          setIsModelModalOpen(true);
          onError("Select and load an ASR model to start transcribing.");
        }}
      />

      <AnimatePresence>
        {isModelModalOpen && (
          <motion.div
            className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm p-4 sm:p-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={closeModelModal}
          >
            <motion.div
              initial={{ y: 16, opacity: 0, scale: 0.98 }}
              animate={{ y: 0, opacity: 1, scale: 1 }}
              exit={{ y: 16, opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.2 }}
              className="mx-auto max-w-3xl max-h-[90vh] overflow-hidden card"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="px-4 sm:px-5 py-4 border-b border-[#262626] flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-semibold text-white">
                    Transcription Models
                  </h2>
                  <p className="text-xs text-gray-500 mt-1">
                    Manage ASR models for this route.
                  </p>
                </div>
                <button
                  className="btn btn-ghost text-xs"
                  onClick={closeModelModal}
                >
                  <X className="w-3.5 h-3.5" />
                  Close
                </button>
              </div>

              <div className="p-4 sm:p-5 overflow-y-auto max-h-[calc(90vh-88px)] space-y-3">
                {loading ? (
                  <div className="flex items-center gap-2 text-sm text-gray-400 py-4">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading models...
                  </div>
                ) : transcriptionModels.length === 0 ? (
                  <div className="text-sm text-gray-400 py-4">
                    No transcription models available for this route.
                  </div>
                ) : (
                  transcriptionModels.map((model) => {
                    const isSelected = resolvedSelectedModel === model.variant;
                    const isIntent = modalIntentModel === model.variant;
                    const isActiveModel = activeReadyModelVariant === model.variant;
                    const progressValue = downloadProgress[model.variant];
                    const progress =
                      progressValue?.percent ?? model.download_progress ?? 0;

                    return (
                      <div
                        key={model.variant}
                        className={clsx(
                          "rounded-lg border p-3 sm:p-4 transition-colors",
                          isIntent
                            ? "border-white/35 bg-[#1a1a1a]"
                            : isSelected
                              ? "border-white/25 bg-[#181818]"
                              : "border-[#2a2a2a] bg-[#141414]",
                        )}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0">
                            <div className="text-sm font-medium text-white truncate">
                              {model.variant}
                            </div>
                            <div className="mt-1 flex items-center gap-2 flex-wrap">
                              <span
                                className={clsx(
                                  "inline-flex items-center rounded-md border px-2 py-0.5 text-[11px]",
                                  getStatusClass(model.status),
                                )}
                              >
                                {getStatusLabel(model.status)}
                              </span>
                              {isActiveModel && (
                                <span className="inline-flex items-center gap-1 text-[11px] text-emerald-300">
                                  <CheckCircle2 className="w-3 h-3" />
                                  Active
                                </span>
                              )}
                            </div>
                          </div>

                          <div className="flex items-center gap-2">
                            {renderPrimaryAction(model, isActiveModel)}
                            {model.status === "ready" && (
                              <button
                                onClick={(event) => {
                                  event.stopPropagation();
                                  onUnload(model.variant);
                                }}
                                className="btn btn-secondary text-xs"
                              >
                                <Square className="w-3.5 h-3.5" />
                                Unload
                              </button>
                            )}
                            {(model.status === "downloaded" ||
                              model.status === "ready") && (
                              pendingDeleteVariant === model.variant ? (
                                <div className="flex items-center gap-1">
                                  <button
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      setPendingDeleteVariant(null);
                                      onDelete(model.variant);
                                    }}
                                    className="btn btn-danger text-xs"
                                  >
                                    <Check className="w-3.5 h-3.5" />
                                    Confirm
                                  </button>
                                  <button
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      setPendingDeleteVariant(null);
                                    }}
                                    className="btn btn-secondary text-xs"
                                  >
                                    <X className="w-3.5 h-3.5" />
                                    Cancel
                                  </button>
                                </div>
                              ) : (
                                <button
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    setPendingDeleteVariant(model.variant);
                                  }}
                                  className="btn btn-danger text-xs"
                                >
                                  <Trash2 className="w-3.5 h-3.5" />
                                  Delete
                                </button>
                              )
                            )}
                          </div>
                        </div>

                        {model.status === "downloading" && (
                          <div className="mt-3">
                            <div className="h-1.5 rounded bg-[#1f1f1f] overflow-hidden">
                              <div
                                className="h-full rounded bg-white transition-all duration-300"
                                style={{ width: `${progress}%` }}
                              />
                            </div>
                            <div className="mt-1 text-[11px] text-gray-500">
                              Downloading {Math.round(progress)}%
                              {progressValue && progressValue.totalBytes > 0 && (
                                <>
                                  {" "}
                                  ({formatBytes(progressValue.downloadedBytes)} /{" "}
                                  {formatBytes(progressValue.totalBytes)})
                                </>
                              )}
                            </div>
                            {progressValue?.currentFile && (
                              <div className="mt-0.5 text-[11px] text-gray-600 truncate">
                                {progressValue.currentFile}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
