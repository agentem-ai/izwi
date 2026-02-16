import { useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, Download, Loader2, Play, Trash2, X } from "lucide-react";
import { ModelInfo } from "../api";
import { DiarizationPlayground } from "../components/DiarizationPlayground";

interface DiarizationPageProps {
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

interface DiarizationModelGroup {
  key: string;
  title: string;
  description: string;
  selectable: boolean;
  models: ModelInfo[];
}

function isDiarizationVariant(variant: string): boolean {
  const normalized = variant.toLowerCase();
  return normalized.includes("sortformer") || normalized.includes("diar");
}

function isPipelineAsrVariant(variant: string): boolean {
  return variant === "Qwen3-ASR-0.6B" || variant.startsWith("Qwen3-ASR-0.6B-");
}

function isPipelineAlignerVariant(variant: string): boolean {
  return variant === "Qwen3-ForcedAligner-0.6B";
}

function isPipelineLlmVariant(variant: string): boolean {
  return variant === "Qwen3-1.7B";
}

export function DiarizationPage({
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
}: DiarizationPageProps) {
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [modalIntentModel, setModalIntentModel] = useState<string | null>(null);
  const [autoCloseOnIntentReady, setAutoCloseOnIntentReady] = useState(false);
  const [pendingDeleteVariant, setPendingDeleteVariant] = useState<
    string | null
  >(null);

  const diarizationModels = useMemo(
    () =>
      models
        .filter((model) => isDiarizationVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models],
  );
  const pipelineModelGroups = useMemo<DiarizationModelGroup[]>(
    () => [
      {
        key: "diarization",
        title: "Diarization",
        description: "Speaker segmentation model used by this route.",
        selectable: true,
        models: models
          .filter((model) => isDiarizationVariant(model.variant))
          .sort((a, b) => a.variant.localeCompare(b.variant)),
      },
      {
        key: "asr",
        title: "ASR",
        description: "Transcript generation model in the diarization pipeline.",
        selectable: false,
        models: models
          .filter((model) => isPipelineAsrVariant(model.variant))
          .sort((a, b) => a.variant.localeCompare(b.variant)),
      },
      {
        key: "aligner",
        title: "Forced Aligner",
        description: "Word timing alignment model for speaker attribution.",
        selectable: false,
        models: models
          .filter((model) => isPipelineAlignerVariant(model.variant))
          .sort((a, b) => a.variant.localeCompare(b.variant)),
      },
      {
        key: "llm",
        title: "Transcript Refiner",
        description: "LLM used to polish final diarized transcript output.",
        selectable: false,
        models: models
          .filter((model) => isPipelineLlmVariant(model.variant))
          .sort((a, b) => a.variant.localeCompare(b.variant)),
      },
    ],
    [models],
  );
  const pipelineModels = useMemo(
    () => pipelineModelGroups.flatMap((group) => group.models),
    [pipelineModelGroups],
  );

  const preferredModelOrder = ["diar_streaming_sortformer_4spk-v2.1"];

  const resolvedSelectedModel = (() => {
    if (
      selectedModel &&
      diarizationModels.some((model) => model.variant === selectedModel)
    ) {
      return selectedModel;
    }

    for (const variant of preferredModelOrder) {
      const readyPreferred = diarizationModels.find(
        (model) => model.variant === variant && model.status === "ready",
      );
      if (readyPreferred) {
        return readyPreferred.variant;
      }
    }

    const readyModel = diarizationModels.find(
      (model) => model.status === "ready",
    );
    if (readyModel) {
      return readyModel.variant;
    }

    for (const variant of preferredModelOrder) {
      const preferred = diarizationModels.find(
        (model) => model.variant === variant,
      );
      if (preferred) {
        return preferred.variant;
      }
    }

    return diarizationModels[0]?.variant ?? null;
  })();

  const selectedModelInfo =
    diarizationModels.find(
      (model) => model.variant === resolvedSelectedModel,
    ) ?? null;
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
    const targetModel = diarizationModels.find(
      (model) => model.variant === modalIntentModel,
    );
    if (targetModel?.status === "ready") {
      closeModelModal();
    }
  }, [
    autoCloseOnIntentReady,
    closeModelModal,
    diarizationModels,
    isModelModalOpen,
    modalIntentModel,
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
        return "bg-white/10 border-white/20 text-gray-300";
      case "loading":
      case "downloading":
        return "bg-amber-500/15 border-amber-500/40 text-amber-300";
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
    canSelectModel: boolean,
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
      if (canSelectModel) {
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
      return (
        <button className="btn btn-secondary text-xs" disabled>
          <CheckCircle2 className="w-3.5 h-3.5" />
          Loaded
        </button>
      );
    }

    return null;
  };

  const activeReadyModelVariant =
    diarizationModels.find((model) => model.status === "ready")?.variant ??
    null;

  const modelOptions = diarizationModels.map((model) => ({
    value: model.variant,
    label: model.variant,
    statusLabel: getStatusLabel(model.status),
    isReady: model.status === "ready",
  }));

  const handleModelSelect = (variant: string) => {
    const model = diarizationModels.find((m) => m.variant === variant);
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

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-white">Diarization</h1>
      </div>

      <DiarizationPlayground
        selectedModel={resolvedSelectedModel}
        selectedModelReady={selectedModelReady}
        modelLabel={selectedModelInfo?.variant ?? null}
        modelOptions={modelOptions}
        onSelectModel={handleModelSelect}
        onOpenModelManager={openModelManager}
        onModelRequired={() => {
          setModalIntentModel(resolvedSelectedModel);
          setAutoCloseOnIntentReady(true);
          setIsModelModalOpen(true);
          onError("Select and load a diarization model to start.");
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
                    Diarization Models
                  </h2>
                  <p className="text-xs text-gray-500 mt-1">
                    Manage pipeline models for /v1/audio/diarizations.
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
                ) : pipelineModels.length === 0 ? (
                  <div className="text-sm text-gray-400 py-4">
                    No diarization pipeline models available for this route.
                  </div>
                ) : (
                  pipelineModelGroups.map((group) => (
                    <section key={group.key} className="space-y-2">
                      <div className="px-1">
                        <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-400">
                          {group.title}
                        </h3>
                        <p className="text-[11px] text-gray-500 mt-0.5">
                          {group.description}
                        </p>
                      </div>
                      {group.models.length === 0 ? (
                        <div className="rounded-lg border border-[#2a2a2a] bg-[#141414] p-3 text-xs text-gray-500">
                          No {group.title.toLowerCase()} models found.
                        </div>
                      ) : (
                        group.models.map((model) => {
                          const isSelected =
                            resolvedSelectedModel === model.variant;
                          const isIntent = modalIntentModel === model.variant;
                          const isActiveModel =
                            activeReadyModelVariant === model.variant;
                          const progressValue = downloadProgress[model.variant];
                          const progress =
                            progressValue?.percent ??
                            model.download_progress ??
                            0;

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
                                    {isActiveModel && group.selectable && (
                                      <span className="inline-flex items-center gap-1 text-[11px] text-gray-300">
                                        <CheckCircle2 className="w-3 h-3" />
                                        Active
                                      </span>
                                    )}
                                  </div>
                                </div>

                                <div className="flex items-center gap-2">
                                  {renderPrimaryAction(
                                    model,
                                    isActiveModel,
                                    group.selectable,
                                  )}
                                  {model.status === "ready" && (
                                    <button
                                      onClick={(event) => {
                                        event.stopPropagation();
                                        onUnload(model.variant);
                                      }}
                                      className="btn btn-secondary text-xs"
                                    >
                                      Unload
                                    </button>
                                  )}
                                  {model.status !== "downloading" &&
                                    (pendingDeleteVariant === model.variant ? (
                                      <>
                                        <button
                                          onClick={(event) => {
                                            event.stopPropagation();
                                            setPendingDeleteVariant(null);
                                          }}
                                          className="btn btn-secondary text-xs"
                                        >
                                          Cancel
                                        </button>
                                        <button
                                          onClick={(event) => {
                                            event.stopPropagation();
                                            onDelete(model.variant);
                                            setPendingDeleteVariant(null);
                                          }}
                                          className="btn btn-danger text-xs"
                                        >
                                          Confirm Delete
                                        </button>
                                      </>
                                    ) : (
                                      <button
                                        onClick={(event) => {
                                          event.stopPropagation();
                                          setPendingDeleteVariant(
                                            model.variant,
                                          );
                                        }}
                                        className="btn btn-danger text-xs"
                                      >
                                        <Trash2 className="w-3.5 h-3.5" />
                                        Delete
                                      </button>
                                    ))}
                                </div>
                              </div>

                              {model.status === "downloading" && (
                                <div className="mt-3">
                                  <div className="h-1.5 rounded-full bg-[#252525] overflow-hidden">
                                    <div
                                      className="h-full bg-white transition-all duration-300"
                                      style={{ width: `${progress}%` }}
                                    />
                                  </div>
                                  <div className="mt-1 text-[11px] text-gray-500">
                                    {progress.toFixed(1)}%
                                  </div>
                                </div>
                              )}
                            </div>
                          );
                        })
                      )}
                    </section>
                  ))
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
