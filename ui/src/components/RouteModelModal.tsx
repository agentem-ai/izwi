import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  Loader2,
  Play,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ModelInfo } from "../api";
import { MODEL_DETAILS } from "../pages/MyModelsPage";
import { withQwen3Prefix } from "../utils/modelDisplay";

interface RouteModelSection {
  key: string;
  title?: string;
  description?: string;
  models: ModelInfo[];
}

interface RouteModelModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  description: string;
  models: ModelInfo[];
  loading: boolean;
  selectedVariant: string | null;
  intentVariant?: string | null;
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
  onUseModel: (variant: string) => void;
  emptyMessage?: string;
  sections?: RouteModelSection[];
  canUseModel?: (variant: string) => boolean;
  getModelLabel?: (variant: string) => string;
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

function getCategoryLabel(variant: string): string | null {
  const category = MODEL_DETAILS[variant]?.category;
  if (!category) {
    return null;
  }

  if (category === "tts") return "Text to Speech";
  if (category === "asr") return "Transcription";
  return "Chat";
}

function defaultModelLabel(variant: string): string {
  const details = MODEL_DETAILS[variant];
  if (!details) {
    return variant;
  }
  return withQwen3Prefix(details.shortName, variant);
}

function getModelSizeLabel(
  model: ModelInfo,
  progress: {
    percent: number;
    currentFile: string;
    status: string;
    downloadedBytes: number;
    totalBytes: number;
  } | undefined,
): string {
  if (progress && progress.totalBytes > 0) {
    return formatBytes(progress.totalBytes);
  }
  if (model.size_bytes !== null) {
    return formatBytes(model.size_bytes);
  }
  const knownSize = MODEL_DETAILS[model.variant]?.size;
  if (knownSize) {
    return knownSize;
  }
  return "Size unknown";
}

function requiresManualDownload(variant: string): boolean {
  return variant === "Gemma-3-1b-it";
}

export function RouteModelModal({
  isOpen,
  onClose,
  title,
  description,
  models,
  loading,
  selectedVariant,
  intentVariant,
  downloadProgress,
  onDownload,
  onCancelDownload,
  onLoad,
  onUnload,
  onDelete,
  onUseModel,
  emptyMessage = "No models are available for this route.",
  sections,
  canUseModel,
  getModelLabel,
}: RouteModelModalProps) {
  const [deleteTargetVariant, setDeleteTargetVariant] = useState<string | null>(
    null,
  );

  useEffect(() => {
    if (!isOpen) {
      setDeleteTargetVariant(null);
    }
  }, [isOpen]);

  const modalSections = useMemo(() => {
    if (sections && sections.length > 0) {
      return sections;
    }
    return [{ key: "models", models }];
  }, [models, sections]);

  const orderedModels = useMemo(
    () => modalSections.flatMap((section) => section.models),
    [modalSections],
  );

  const activeReadyModelVariant =
    orderedModels.find(
      (model) =>
        model.status === "ready" &&
        (canUseModel ? canUseModel(model.variant) : true),
    )?.variant ?? null;

  const deleteTargetModel = deleteTargetVariant
    ? orderedModels.find((model) => model.variant === deleteTargetVariant) ?? null
    : null;

  const resolveModelLabel = (variant: string): string => {
    if (getModelLabel) {
      return getModelLabel(variant);
    }
    return defaultModelLabel(variant);
  };

  const destructiveDeleteButtonClass =
    "flex items-center gap-1.5 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-2.5 py-1.5 text-xs font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]";

  const handleConfirmDelete = () => {
    if (!deleteTargetModel) {
      return;
    }
    onDelete(deleteTargetModel.variant);
    setDeleteTargetVariant(null);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 z-50 bg-black/70 p-4 backdrop-blur-sm sm:p-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            initial={{ y: 16, opacity: 0, scale: 0.98 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            exit={{ y: 16, opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.2 }}
            className="mx-auto flex max-h-[90vh] max-w-4xl flex-col overflow-hidden rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)]"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between gap-3 border-b border-[var(--border-muted)] px-4 py-4 sm:px-5">
              <div>
                <h2 className="text-base font-semibold text-[var(--text-primary)]">
                  {title}
                </h2>
                <p className="mt-1 text-xs text-[var(--text-muted)]">
                  {description}
                </p>
              </div>
              <button
                className="flex items-center gap-1 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1.5 text-xs text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
                onClick={onClose}
              >
                <X className="h-3.5 w-3.5" />
                Close
              </button>
            </div>

            <div className="max-h-[calc(90vh-88px)] overflow-y-auto px-4 py-4 sm:px-5">
              {loading ? (
                <div className="flex items-center gap-2 py-4 text-sm text-[var(--text-muted)]">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading models...
                </div>
              ) : orderedModels.length === 0 ? (
                <div className="py-4 text-sm text-[var(--text-muted)]">
                  {emptyMessage}
                </div>
              ) : (
                <div className="space-y-4">
                  {modalSections.map((section) => (
                    <section key={section.key} className="space-y-2">
                      {section.title && (
                        <div className="px-1">
                          <h3 className="text-xs font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                            {section.title}
                          </h3>
                          {section.description && (
                            <p className="mt-0.5 text-[11px] text-[var(--text-subtle)]">
                              {section.description}
                            </p>
                          )}
                        </div>
                      )}

                      {section.models.length === 0 ? (
                        <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3 text-xs text-[var(--text-subtle)]">
                          No models in this group.
                        </div>
                      ) : (
                        section.models.map((model) => {
                          const isSelected = selectedVariant === model.variant;
                          const isIntent = intentVariant === model.variant;
                          const isActiveModel =
                            activeReadyModelVariant === model.variant;
                          const progressValue = downloadProgress[model.variant];
                          const progress =
                            progressValue?.percent ?? model.download_progress ?? 0;
                          const canSelect = canUseModel
                            ? canUseModel(model.variant)
                            : true;
                          const categoryLabel = getCategoryLabel(model.variant);
                          const modelSizeLabel = getModelSizeLabel(
                            model,
                            progressValue,
                          );

                          return (
                            <div
                              key={model.variant}
                              className={clsx(
                                "rounded-xl border p-3 transition-colors sm:p-4",
                                isIntent
                                  ? "border-[var(--border-strong)] bg-[var(--bg-surface-2)]"
                                  : isSelected
                                    ? "border-[var(--border-strong)] bg-[var(--bg-surface-1)]"
                                    : "border-[var(--border-muted)] bg-[var(--bg-surface-1)]",
                              )}
                            >
                              <div className="flex flex-col gap-3 md:flex-row md:items-center">
                                <div className="min-w-0 flex-1">
                                  <div className="flex flex-wrap items-center gap-2">
                                    {model.status === "downloading" ||
                                    model.status === "loading" ? (
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
                                      {resolveModelLabel(model.variant)}
                                    </h3>
                                    <span
                                      className={clsx(
                                        "rounded border px-2 py-0.5 text-[11px] font-medium",
                                        getStatusBadgeClass(model.status),
                                      )}
                                    >
                                      {getStatusLabel(model.status)}
                                    </span>
                                    {isActiveModel && canSelect && (
                                      <span className="inline-flex items-center gap-1 text-[11px] text-[var(--text-secondary)]">
                                        <CheckCircle2 className="h-3 w-3" />
                                        Active
                                      </span>
                                    )}
                                  </div>

                                  <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-[var(--text-subtle)]">
                                    {categoryLabel && <span>{categoryLabel}</span>}
                                    {categoryLabel && <span aria-hidden>•</span>}
                                    <span>{modelSizeLabel}</span>
                                    <span aria-hidden>•</span>
                                    <span className="truncate">{model.variant}</span>
                                  </div>

                                  {model.status === "downloading" && (
                                    <div className="mt-2">
                                      <div className="h-1.5 w-full max-w-[260px] overflow-hidden rounded-full bg-[var(--bg-surface-3)]">
                                        <div
                                          className="h-full rounded-full bg-[var(--accent-solid)] transition-all duration-300"
                                          style={{ width: `${progress}%` }}
                                        />
                                      </div>
                                      <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                                        Downloading {Math.round(progress)}%
                                        {progressValue &&
                                          progressValue.totalBytes > 0 && (
                                            <>
                                              {" "}
                                              (
                                              {formatBytes(
                                                progressValue.downloadedBytes,
                                              )} / {formatBytes(progressValue.totalBytes)})
                                            </>
                                          )}
                                      </div>
                                      {progressValue?.currentFile && (
                                        <div className="mt-0.5 truncate text-[11px] text-[var(--text-subtle)]">
                                          {progressValue.currentFile}
                                        </div>
                                      )}
                                    </div>
                                  )}
                                </div>

                                <div className="flex flex-wrap items-center gap-1.5">
                                  {model.status === "downloading" &&
                                    onCancelDownload && (
                                      <button
                                        onClick={() => onCancelDownload(model.variant)}
                                        className="flex items-center gap-1 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-2.5 py-1.5 text-xs font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                                      >
                                        <X className="h-3.5 w-3.5" />
                                        Cancel
                                      </button>
                                    )}

                                  {(model.status === "not_downloaded" ||
                                    model.status === "error") &&
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

                                  {model.status === "downloaded" && (
                                    <button
                                      onClick={() => onLoad(model.variant)}
                                      className="flex items-center gap-1.5 rounded-md bg-[var(--accent-solid)] px-3 py-1.5 text-xs font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90"
                                    >
                                      <Play className="h-3.5 w-3.5" />
                                      Load
                                    </button>
                                  )}

                                  {model.status === "loading" && (
                                    <button
                                      className="flex items-center gap-1.5 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-muted)]"
                                      disabled
                                    >
                                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                      Loading
                                    </button>
                                  )}

                                  {model.status === "ready" && canSelect && (
                                    isSelected ? (
                                      <button
                                        className="flex items-center gap-1.5 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-muted)]"
                                        disabled
                                      >
                                        <CheckCircle2 className="h-3.5 w-3.5" />
                                        Selected
                                      </button>
                                    ) : (
                                      <button
                                        onClick={() => {
                                          onUseModel(model.variant);
                                          onClose();
                                        }}
                                        className="flex items-center gap-1.5 rounded-md bg-[var(--accent-solid)] px-3 py-1.5 text-xs font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90"
                                      >
                                        <CheckCircle2 className="h-3.5 w-3.5" />
                                        Use model
                                      </button>
                                    )
                                  )}

                                  {model.status === "ready" && (
                                    <button
                                      onClick={() => onUnload(model.variant)}
                                      className="flex items-center gap-1.5 rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-surface-3)]"
                                    >
                                      <Square className="h-3.5 w-3.5" />
                                      Unload
                                    </button>
                                  )}

                                  {(model.status === "downloaded" ||
                                    model.status === "ready") && (
                                    <button
                                      onClick={() =>
                                        setDeleteTargetVariant(model.variant)
                                      }
                                      className={destructiveDeleteButtonClass}
                                    >
                                      <Trash2 className="h-3.5 w-3.5" />
                                      Delete
                                    </button>
                                  )}
                                </div>
                              </div>
                            </div>
                          );
                        })
                      )}
                    </section>
                  ))}
                </div>
              )}
            </div>
          </motion.div>

          <AnimatePresence>
            {deleteTargetModel && (
              <motion.div
                className="fixed inset-0 z-[60] bg-black/75 p-4 backdrop-blur-sm"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setDeleteTargetVariant(null)}
              >
                <motion.div
                  initial={{ y: 10, opacity: 0, scale: 0.98 }}
                  animate={{ y: 0, opacity: 1, scale: 1 }}
                  exit={{ y: 10, opacity: 0, scale: 0.98 }}
                  transition={{ duration: 0.16 }}
                  className="mx-auto mt-[18vh] max-w-md rounded-xl border border-[var(--danger-border)] bg-[var(--bg-surface-1)] p-5"
                  onClick={(event) => event.stopPropagation()}
                >
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
                      <AlertTriangle className="h-4 w-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                        Delete model?
                      </h3>
                      <p className="mt-1 text-sm text-[var(--text-muted)]">
                        This removes
                        <span className="mx-1 font-medium text-[var(--text-primary)]">
                          {resolveModelLabel(deleteTargetModel.variant)}
                        </span>
                        from local storage.
                      </p>
                      <p className="mt-2 truncate text-xs text-[var(--text-subtle)]">
                        {deleteTargetModel.variant}
                      </p>
                    </div>
                  </div>

                  <div className="mt-5 flex items-center justify-end gap-2">
                    <button
                      onClick={() => setDeleteTargetVariant(null)}
                      className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-surface-3)]"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleConfirmDelete}
                      className="flex items-center gap-1.5 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-1.5 text-xs font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                      Delete model
                    </button>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
