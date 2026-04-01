import { useEffect, useMemo, useRef } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { ArrowLeft, Settings2 } from "lucide-react";

import { type ModelInfo } from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { StatusBadge } from "@/components/ui/status-badge";
import {
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";
import { TextToSpeechHistoryTable } from "@/features/text-to-speech/components/TextToSpeechHistoryTable";
import { useTextToSpeechHistory } from "@/features/text-to-speech/hooks/useTextToSpeechHistory";
import { useTextToSpeechRecord } from "@/features/text-to-speech/hooks/useTextToSpeechRecord";
import {
  formatSpeechCreatedAt,
  normalizeSpeechProcessingStatus,
  speechProcessingStatusLabel,
} from "@/features/text-to-speech/support";

interface TextToSpeechPageProps {
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

function statusToneFor(
  status: ReturnType<typeof normalizeSpeechProcessingStatus>,
): "neutral" | "warning" | "success" | "danger" {
  switch (status) {
    case "pending":
      return "neutral";
    case "processing":
      return "warning";
    case "failed":
      return "danger";
    case "ready":
    default:
      return "success";
  }
}

export function TextToSpeechPage({
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
}: TextToSpeechPageProps) {
  const { recordId } = useParams<{ recordId: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const appliedQueryModelRef = useRef(false);
  const {
    routeModels,
    resolvedSelectedModel,
    isModelModalOpen,
    intentVariant,
    closeModelModal,
    openModelManager,
  } = useRouteModelSelection({
    models,
    selectedModel,
    onSelect,
    modelFilter: (variant) => {
      const match = models.find((model) => model.variant === variant);
      const capabilities = match?.speech_capabilities;
      return Boolean(
        capabilities &&
          !variant.includes("Tokenizer") &&
          (capabilities.supports_builtin_voices ||
            capabilities.supports_reference_voice ||
            capabilities.supports_voice_description),
      );
    },
    resolveSelectedModel: (availableRouteModels, currentModel) =>
      resolvePreferredRouteModel({
        models: availableRouteModels,
        selectedModel: currentModel,
        preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
      }),
  });

  const {
    records,
    loading: historyLoading,
    error: historyError,
    refresh: refreshHistory,
  } = useTextToSpeechHistory();
  const {
    record,
    loading: recordLoading,
    error: recordError,
    refresh: refreshRecord,
  } = useTextToSpeechRecord(recordId);

  useEffect(() => {
    if (appliedQueryModelRef.current || routeModels.length === 0) {
      return;
    }

    const requestedModel = searchParams.get("model");
    if (
      requestedModel &&
      routeModels.some((model) => model.variant === requestedModel)
    ) {
      onSelect(requestedModel);
    }

    appliedQueryModelRef.current = true;
  }, [onSelect, routeModels, searchParams]);

  const detailDescription = useMemo(() => {
    if (!record) {
      return "Inspect status and generation details for this text-to-speech record.";
    }

    const processingStatus = normalizeSpeechProcessingStatus(
      record.processing_status,
      record.processing_error,
    );

    switch (processingStatus) {
      case "pending":
        return "This generation is queued.";
      case "processing":
        return "This generation is actively rendering audio.";
      case "failed":
        return "This generation failed before audio became available.";
      case "ready":
      default:
        return "Inspect status and generation details for this text-to-speech record.";
    }
  }, [record]);

  return (
    <PageShell>
      {recordId ? (
        <>
          <PageHeader
            title="Text-to-Speech Record"
            description={detailDescription}
            actions={
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 gap-2"
                onClick={openModelManager}
              >
                <Settings2 className="h-4 w-4" />
                Models
              </Button>
            }
          />

          <Card className="mb-6 border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5 sm:p-6">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="mb-4 h-10 gap-2 rounded-full px-4"
              onClick={() => navigate("/text-to-speech")}
            >
              <ArrowLeft className="h-4 w-4" />
              Back to history
            </Button>

            {recordLoading ? (
              <p className="text-sm text-[var(--text-muted)]">Loading record...</p>
            ) : null}

            {recordError ? (
              <div className="rounded-xl border border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]">
                <div className="flex items-center justify-between gap-3">
                  <p>{recordError}</p>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => void refreshRecord()}
                  >
                    Retry
                  </Button>
                </div>
              </div>
            ) : null}

            {!recordLoading && !recordError && record ? (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <StatusBadge
                    tone={statusToneFor(
                      normalizeSpeechProcessingStatus(
                        record.processing_status,
                        record.processing_error,
                      ),
                    )}
                  >
                    {speechProcessingStatusLabel(
                      normalizeSpeechProcessingStatus(
                        record.processing_status,
                        record.processing_error,
                      ),
                    )}
                  </StatusBadge>
                  <span className="text-xs text-[var(--text-muted)]">
                    {formatSpeechCreatedAt(record.created_at)}
                  </span>
                </div>
                <div className="grid gap-3 sm:grid-cols-2">
                  <div>
                    <div className="text-xs uppercase tracking-[0.12em] text-[var(--text-muted)]">
                      Model
                    </div>
                    <div className="mt-1 text-sm text-[var(--text-primary)]">
                      {record.model_id || "Unknown"}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs uppercase tracking-[0.12em] text-[var(--text-muted)]">
                      Voice
                    </div>
                    <div className="mt-1 text-sm text-[var(--text-primary)]">
                      {record.saved_voice_id ||
                        record.speaker ||
                        "No voice metadata"}
                    </div>
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-[0.12em] text-[var(--text-muted)]">
                    Input
                  </div>
                  <p className="mt-1 whitespace-pre-wrap rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3 text-sm leading-6 text-[var(--text-primary)]">
                    {record.input_text}
                  </p>
                </div>
              </div>
            ) : null}
          </Card>
        </>
      ) : (
        <>
          <PageHeader
            title="Text to Speech"
            description="Monitor queued, processing, and completed speech generations in one operational history table."
            actions={
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 gap-2"
                onClick={openModelManager}
              >
                <Settings2 className="h-4 w-4" />
                Models
              </Button>
            }
          />

          <TextToSpeechHistoryTable
            records={records}
            loading={historyLoading}
            error={historyError}
            onRefresh={() => void refreshHistory()}
            onOpenRecord={(nextRecordId) => {
              navigate(`/text-to-speech/${nextRecordId}`);
            }}
          />
        </>
      )}

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title="Text-to-Speech Models"
        description="Manage built-in voice, saved-voice, and voice-direction TTS models for this route."
        models={routeModels}
        loading={loading}
        selectedVariant={resolvedSelectedModel}
        intentVariant={intentVariant}
        downloadProgress={downloadProgress}
        onDownload={onDownload}
        onCancelDownload={onCancelDownload}
        onLoad={onLoad}
        onUnload={onUnload}
        onDelete={onDelete}
        onUseModel={onSelect}
        emptyMessage="Load a compatible TTS model with built-in voices, saved voices, or voice-direction prompts to generate speech."
      />
    </PageShell>
  );
}
