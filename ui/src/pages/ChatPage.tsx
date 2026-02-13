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
import { ChatPlayground } from "../components/ChatPlayground";
import { VIEW_CONFIGS } from "../types";
import { withQwen3Prefix } from "../utils/modelDisplay";

interface ChatPageProps {
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

function getChatModelName(variant: string): string {
  if (variant === "Qwen3-0.6B-4bit") {
    return withQwen3Prefix("Chat 0.6B 4-bit", variant);
  }
  if (variant === "Gemma-3-1b-it") {
    return "Gemma 3 1B Instruct";
  }
  if (variant === "Gemma-3-4b-it") {
    return "Gemma 3 4B Instruct";
  }
  return variant;
}

function isThinkingChatModel(variant: string): boolean {
  return variant === "Qwen3-0.6B-4bit";
}

function getStatusLabel(status: ModelInfo["status"]): string {
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
}

function getStatusClass(status: ModelInfo["status"]): string {
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
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function requiresManualDownload(variant: string): boolean {
  return variant === "Gemma-3-1b-it";
}

export function ChatPage({
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
}: ChatPageProps) {
  const viewConfig = VIEW_CONFIGS.chat;
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [modalIntentModel, setModalIntentModel] = useState<string | null>(null);
  const [autoCloseOnIntentReady, setAutoCloseOnIntentReady] = useState(false);
  const [pendingDeleteVariant, setPendingDeleteVariant] = useState<string | null>(
    null,
  );

  const chatModels = useMemo(
    () =>
      models
        .filter((model) => viewConfig.modelFilter(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models, viewConfig],
  );

  const resolvedSelectedModel = (() => {
    if (selectedModel && viewConfig.modelFilter(selectedModel)) {
      return selectedModel;
    }
    const readyModel = chatModels.find((model) => model.status === "ready");
    if (readyModel) {
      return readyModel.variant;
    }
    return chatModels[0]?.variant ?? null;
  })();

  const selectedModelInfo =
    chatModels.find((model) => model.variant === resolvedSelectedModel) ?? null;
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
    const targetModel = chatModels.find((m) => m.variant === modalIntentModel);
    if (targetModel?.status === "ready") {
      closeModelModal();
    }
  }, [
    autoCloseOnIntentReady,
    chatModels,
    isModelModalOpen,
    modalIntentModel,
  ]);

  const handleModelSelect = (variant: string) => {
    const model = chatModels.find((m) => m.variant === variant);
    if (!model) {
      return;
    }

    onSelect(variant);

    if (model.status !== "ready") {
      setModalIntentModel(variant);
      setAutoCloseOnIntentReady(true);
      setIsModelModalOpen(true);
    }
  };

  const openModelManager = () => {
    setModalIntentModel(resolvedSelectedModel);
    setAutoCloseOnIntentReady(false);
    setIsModelModalOpen(true);
  };

  const modelOptions = chatModels.map((model) => ({
    value: model.variant,
    label: getChatModelName(model.variant),
    statusLabel: getStatusLabel(model.status),
    isReady: model.status === "ready",
  }));

  const activeReadyModelVariant =
    chatModels.find((model) => model.status === "ready")?.variant ?? null;

  const renderModelPrimaryAction = (
    model: ModelInfo,
    isActiveModel: boolean,
  ) => {
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
      if (requiresManualDownload(model.variant)) {
        return (
          <button
            className="btn btn-secondary text-xs"
            disabled
            title="Manual download required. See docs/user/manual-gemma-3-1b-download.md."
          >
            <Download className="w-3.5 h-3.5" />
            Manual download
          </button>
        );
      }

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

  return (
    <div className="max-w-6xl mx-auto">
      <ChatPlayground
        selectedModel={resolvedSelectedModel}
        selectedModelReady={selectedModelReady}
        modelLabel={selectedModelInfo ? getChatModelName(selectedModelInfo.variant) : null}
        supportsThinking={
          resolvedSelectedModel ? isThinkingChatModel(resolvedSelectedModel) : false
        }
        modelOptions={modelOptions}
        onSelectModel={handleModelSelect}
        onOpenModelManager={openModelManager}
        onModelRequired={() => {
          setModalIntentModel(resolvedSelectedModel);
          setAutoCloseOnIntentReady(true);
          setIsModelModalOpen(true);
          onError("Select a model and load it to start chatting.");
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
                    Chat Models
                  </h2>
                  <p className="text-xs text-gray-500 mt-1">
                    Select, download, load, and unload models for this route.
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
                ) : chatModels.length === 0 ? (
                  <div className="text-sm text-gray-400 py-4">
                    No chat models available for this route.
                  </div>
                ) : (
                  chatModels.map((model) => {
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
                              {getChatModelName(model.variant)}
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
                            {renderModelPrimaryAction(model, isActiveModel)}
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
