import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  api,
  type DiarizationRecord,
  type DiarizationRecordRerunRequest,
  type ModelInfo,
} from "@/api";
import { DiarizationPlayground } from "@/components/DiarizationPlayground";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { DiarizationHistoryTable } from "@/features/diarization/components/DiarizationHistoryTable";
import { DiarizationRecordDetail } from "@/features/diarization/components/DiarizationRecordDetail";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useDiarizationHistory } from "@/features/diarization/hooks/useDiarizationHistory";
import { useDiarizationRecord } from "@/features/diarization/hooks/useDiarizationRecord";
import { summarizeDiarizationRecord } from "@/features/diarization/historySummary";
import { Settings2 } from "lucide-react";

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
  models: ModelInfo[];
}

function resolvePreferredVariant(
  availableModels: ModelInfo[],
  preferredOrder: string[],
): string | null {
  for (const variant of preferredOrder) {
    const readyPreferred = availableModels.find(
      (model) => model.variant === variant && model.status === "ready",
    );
    if (readyPreferred) {
      return readyPreferred.variant;
    }
  }

  const readyModel = availableModels.find((model) => model.status === "ready");
  if (readyModel) {
    return readyModel.variant;
  }

  for (const variant of preferredOrder) {
    const preferred = availableModels.find((model) => model.variant === variant);
    if (preferred) {
      return preferred.variant;
    }
  }

  return availableModels[0]?.variant ?? null;
}

function isDiarizationVariant(variant: string): boolean {
  const normalized = variant.toLowerCase();
  return normalized.includes("sortformer") || normalized.includes("diar");
}

function isPipelineAsrVariant(variant: string): boolean {
  return variant.startsWith("Parakeet-TDT-");
}

function isPipelineAlignerVariant(variant: string): boolean {
  return variant === "Qwen3-ForcedAligner-0.6B";
}

function isPipelineLlmVariant(variant: string): boolean {
  return variant === "Qwen3.5-4B";
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
  const { recordId } = useParams<{ recordId: string }>();
  const navigate = useNavigate();
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [modalIntentModel, setModalIntentModel] = useState<string | null>(null);
  const [autoCloseOnIntentReady, setAutoCloseOnIntentReady] = useState(false);
  const [isPipelineLoadAllRequested, setIsPipelineLoadAllRequested] =
    useState(false);
  const loadAllDownloadRequestedRef = useRef(new Set<string>());
  const loadAllLoadRequestedRef = useRef(new Set<string>());
  const [latestRecord, setLatestRecord] = useState<DiarizationRecord | null>(
    null,
  );

  const diarizationModels = useMemo(
    () =>
      models
        .filter((model) => isDiarizationVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models],
  );

  const asrPipelineModels = useMemo(
    () =>
      models
        .filter((model) => isPipelineAsrVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models],
  );

  const alignerPipelineModels = useMemo(
    () =>
      models
        .filter((model) => isPipelineAlignerVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models],
  );

  const llmPipelineModels = useMemo(
    () =>
      models
        .filter((model) => isPipelineLlmVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models],
  );

  const pipelineModelGroups = useMemo<DiarizationModelGroup[]>(
    () => [
      {
        key: "diarization",
        title: "Diarization",
        description: "Speaker segmentation model used by this route.",
        models: diarizationModels,
      },
      {
        key: "asr",
        title: "ASR",
        description: "Transcript generation model in the diarization pipeline.",
        models: asrPipelineModels,
      },
      {
        key: "aligner",
        title: "Forced Aligner",
        description: "Word timing alignment model for speaker attribution.",
        models: alignerPipelineModels,
      },
      {
        key: "llm",
        title: "Refiner + Summary",
        description:
          "LLM used for transcript refinement and diarization summaries.",
        models: llmPipelineModels,
      },
    ],
    [
      asrPipelineModels,
      alignerPipelineModels,
      diarizationModels,
      llmPipelineModels,
    ],
  );

  const pipelineModels = useMemo(
    () => pipelineModelGroups.flatMap((group) => group.models),
    [pipelineModelGroups],
  );

  const preferredDiarizationModelOrder = ["diar_streaming_sortformer_4spk-v2.1"];
  const preferredAsrModelOrder = ["Parakeet-TDT-0.6B-v3"];
  const preferredAlignerModelOrder = ["Qwen3-ForcedAligner-0.6B"];
  const preferredLlmModelOrder = ["Qwen3.5-4B"];

  const resolvedSelectedModel =
    selectedModel &&
    diarizationModels.some((model) => model.variant === selectedModel)
      ? selectedModel
      : resolvePreferredVariant(diarizationModels, preferredDiarizationModelOrder);

  const selectedModelInfo =
    diarizationModels.find(
      (model) => model.variant === resolvedSelectedModel,
    ) ?? null;
  const selectedModelReady = selectedModelInfo?.status === "ready";

  const resolvedAsrModel = resolvePreferredVariant(
    asrPipelineModels,
    preferredAsrModelOrder,
  );
  const resolvedAlignerModel = resolvePreferredVariant(
    alignerPipelineModels,
    preferredAlignerModelOrder,
  );
  const resolvedLlmModel = resolvePreferredVariant(
    llmPipelineModels,
    preferredLlmModelOrder,
  );
  const resolvedSummaryModel = resolvedLlmModel;

  const asrModelReady =
    resolvedAsrModel != null &&
    asrPipelineModels.some(
      (model) => model.variant === resolvedAsrModel && model.status === "ready",
    );
  const alignerModelReady =
    resolvedAlignerModel != null &&
    alignerPipelineModels.some(
      (model) =>
        model.variant === resolvedAlignerModel && model.status === "ready",
    );
  const llmModelReady =
    resolvedLlmModel != null &&
    llmPipelineModels.some(
      (model) => model.variant === resolvedLlmModel && model.status === "ready",
    );
  const summaryModelStatus = useMemo(() => {
    if (!resolvedSummaryModel) {
      return null;
    }
    return (
      llmPipelineModels.find(
        (model) => model.variant === resolvedSummaryModel,
      )?.status ?? null
    );
  }, [llmPipelineModels, resolvedSummaryModel]);
  const summaryModelReady = llmModelReady;
  const summaryModelRequirementMessage = useMemo(() => {
    const modelName = resolvedSummaryModel || "Qwen3.5-4B";
    switch (summaryModelStatus) {
      case "downloaded":
        return `Load ${modelName} in Diarization Models to generate summaries.`;
      case "downloading":
        return `${modelName} is downloading. Wait for download to complete, then generate summaries.`;
      case "loading":
        return `${modelName} is loading. Wait for it to become ready, then generate summaries.`;
      case "not_downloaded":
      case "error":
      default:
        return `Download and load ${modelName} in Diarization Models to generate summaries.`;
    }
  }, [resolvedSummaryModel, summaryModelStatus]);
  const pipelineModelsReady = asrModelReady && alignerModelReady;

  const targetPipelineVariants = useMemo(
    () =>
      [
        resolvedSelectedModel,
        resolvedAsrModel,
        resolvedAlignerModel,
        resolvedLlmModel,
      ].filter((variant): variant is string => !!variant),
    [
      resolvedAlignerModel,
      resolvedAsrModel,
      resolvedLlmModel,
      resolvedSelectedModel,
    ],
  );

  const targetPipelineModels = useMemo(
    () =>
      targetPipelineVariants
        .map(
          (variant) =>
            pipelineModels.find((model) => model.variant === variant) ?? null,
        )
        .filter((model): model is ModelInfo => model !== null),
    [pipelineModels, targetPipelineVariants],
  );

  const pipelineAllLoaded =
    targetPipelineModels.length > 0 &&
    targetPipelineModels.every((model) => model.status === "ready");

  const pipelineLoadAllBusy =
    isPipelineLoadAllRequested ||
    targetPipelineModels.some(
      (model) => model.status === "downloading" || model.status === "loading",
    );
  const {
    records,
    loading: historyLoading,
    error: historyError,
    refresh: refreshHistory,
  } = useDiarizationHistory();
  const {
    record,
    loading: recordLoading,
    error: recordError,
    refresh: refreshRecord,
  } = useDiarizationRecord(recordId);
  const detailAudioUrl = useMemo(
    () => (recordId ? api.diarizationRecordAudioUrl(recordId) : null),
    [recordId],
  );
  const visibleHistoryRecords = useMemo(() => {
    const nextRecords = latestRecord
      ? [summarizeDiarizationRecord(latestRecord), ...records]
      : records;
    return nextRecords.filter(
      (recordSummary, index, list) =>
        list.findIndex((candidate) => candidate.id === recordSummary.id) ===
        index,
    );
  }, [latestRecord, records]);

  const closeModelModal = () => {
    setIsModelModalOpen(false);
    setAutoCloseOnIntentReady(false);
  };

  useEffect(() => {
    if (!isModelModalOpen || !modalIntentModel || !autoCloseOnIntentReady) {
      return;
    }
    const targetModel = pipelineModels.find(
      (model) => model.variant === modalIntentModel,
    );
    if (targetModel?.status === "ready") {
      closeModelModal();
    }
  }, [
    autoCloseOnIntentReady,
    pipelineModels,
    isModelModalOpen,
    modalIntentModel,
  ]);

  const openModelManager = () => {
    setModalIntentModel(null);
    setAutoCloseOnIntentReady(false);
    setIsModelModalOpen(true);
  };

  const openModelManagerForPipeline = () => {
    const missingPipelineVariant =
      (!asrModelReady && resolvedAsrModel) ||
      (!alignerModelReady && resolvedAlignerModel) ||
      resolvedSelectedModel;
    setModalIntentModel(missingPipelineVariant);
    setAutoCloseOnIntentReady(true);
    setIsModelModalOpen(true);
  };

  const handleToggleLoadAllPipeline = () => {
    if (pipelineAllLoaded) {
      for (const model of targetPipelineModels) {
        if (model.status === "ready") {
          onUnload(model.variant);
        }
      }
      return;
    }

    loadAllDownloadRequestedRef.current.clear();
    loadAllLoadRequestedRef.current.clear();
    setIsPipelineLoadAllRequested(true);
  };

  useEffect(() => {
    if (!isPipelineLoadAllRequested) {
      return;
    }

    if (targetPipelineModels.length === 0) {
      setIsPipelineLoadAllRequested(false);
      return;
    }

    let allReady = true;
    let encounteredError = false;

    for (const model of targetPipelineModels) {
      if (model.status === "ready") {
        continue;
      }

      allReady = false;

      if (
        model.status === "error" &&
        loadAllDownloadRequestedRef.current.has(model.variant)
      ) {
        encounteredError = true;
      }

      if (
        (model.status === "not_downloaded" || model.status === "error") &&
        !loadAllDownloadRequestedRef.current.has(model.variant)
      ) {
        loadAllDownloadRequestedRef.current.add(model.variant);
        onDownload(model.variant);
      }

      if (
        model.status === "downloaded" &&
        !loadAllLoadRequestedRef.current.has(model.variant)
      ) {
        loadAllLoadRequestedRef.current.add(model.variant);
        onLoad(model.variant);
      }
    }

    if (allReady || encounteredError) {
      setIsPipelineLoadAllRequested(false);
      loadAllDownloadRequestedRef.current.clear();
      loadAllLoadRequestedRef.current.clear();
    }
  }, [isPipelineLoadAllRequested, onDownload, onLoad, targetPipelineModels]);

  const handleOpenRecord = useCallback(
    (nextRecordId: string) => {
      navigate(`/diarization/${nextRecordId}`);
    },
    [navigate],
  );

  const handleCloseRecord = useCallback(() => {
    navigate("/diarization");
  }, [navigate]);

  const handleOpenModels = useCallback(() => {
    openModelManager();
  }, [openModelManager]);

  const handleDeleteRecord = useCallback(
    async (targetRecordId: string) => {
      await api.deleteDiarizationRecord(targetRecordId);
      await refreshHistory();
      if (recordId === targetRecordId) {
        navigate("/diarization", { replace: true });
      }
      if (latestRecord?.id === targetRecordId) {
        setLatestRecord(null);
      }
    },
    [latestRecord?.id, navigate, recordId, refreshHistory],
  );

  const handleSaveSpeakerCorrections = useCallback(
    async (
      targetRecordId: string,
      speakerNameOverrides: Record<string, string>,
    ) => {
      await api.updateDiarizationRecord(targetRecordId, {
        speaker_name_overrides: speakerNameOverrides,
      });
      await Promise.all([
        refreshHistory(),
        recordId === targetRecordId ? refreshRecord() : Promise.resolve(),
      ]);
    },
    [recordId, refreshHistory, refreshRecord],
  );

  const handleRerunRecord = useCallback(
    async (
      targetRecordId: string,
      request: DiarizationRecordRerunRequest,
    ) => {
      const rerunRecord = await api.rerunDiarizationRecord(targetRecordId, request);
      await refreshHistory();
      if (latestRecord?.id === targetRecordId) {
        setLatestRecord(rerunRecord);
      }
      navigate(`/diarization/${rerunRecord.id}`);
    },
    [latestRecord?.id, navigate, refreshHistory],
  );

  const handleRegenerateSummary = useCallback(
    async (targetRecordId: string) => {
      if (!summaryModelReady) {
        openModelManager();
        onError(summaryModelRequirementMessage);
        throw new Error(summaryModelRequirementMessage);
      }
      await api.regenerateDiarizationSummary(targetRecordId);
      await Promise.all([
        refreshHistory(),
        recordId === targetRecordId ? refreshRecord() : Promise.resolve(),
      ]);
    },
    [
      onError,
      openModelManager,
      recordId,
      refreshHistory,
      refreshRecord,
      summaryModelReady,
      summaryModelRequirementMessage,
    ],
  );

  return (
    <PageShell>
      {recordId ? (
        <>
          <PageHeader
            title="Diarization Record"
            description="Inspect transcript output, speaker corrections, and quality reruns for this saved diarization record."
            actions={
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 gap-2"
                onClick={handleOpenModels}
              >
                <Settings2 className="h-4 w-4" />
                Models
              </Button>
            }
          />

          <DiarizationRecordDetail
            record={record}
            audioUrl={detailAudioUrl}
            loading={recordLoading}
            error={recordError}
            summaryModelGuidance={
              summaryModelReady ? null : summaryModelRequirementMessage
            }
            onBack={handleCloseRecord}
            onDelete={handleDeleteRecord}
            onSaveSpeakerCorrections={handleSaveSpeakerCorrections}
            onRerun={handleRerunRecord}
            onRegenerateSummary={handleRegenerateSummary}
          />
        </>
      ) : (
        <>
          <PageHeader
            title="Diarization"
            description="Capture speaker-separated transcripts, tune diarization quality, and review saved records from one workspace."
            actions={
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 gap-2"
                onClick={handleOpenModels}
              >
                <Settings2 className="h-4 w-4" />
                Models
              </Button>
            }
          />

          <DiarizationPlayground
            selectedModel={resolvedSelectedModel}
            selectedModelReady={selectedModelReady}
            onOpenModelManager={openModelManager}
            onTogglePipelineLoadAll={handleToggleLoadAllPipeline}
            pipelineAllLoaded={pipelineAllLoaded}
            pipelineLoadAllBusy={pipelineLoadAllBusy}
            onModelRequired={() => {
              setModalIntentModel(resolvedSelectedModel);
              setAutoCloseOnIntentReady(true);
              setIsModelModalOpen(true);
              onError("Select and load a diarization model to start.");
            }}
            pipelineAsrModelId={resolvedAsrModel}
            pipelineAlignerModelId={resolvedAlignerModel}
            pipelineLlmModelId={resolvedLlmModel}
            pipelineLlmModelReady={llmModelReady}
            pipelineModelsReady={pipelineModelsReady}
            summaryModelId={resolvedSummaryModel}
            summaryModelReady={summaryModelReady}
            summaryModelStatus={summaryModelStatus}
            onSummaryModelRequired={() => {
              openModelManager();
              onError(summaryModelRequirementMessage);
            }}
            onLatestRecordChange={setLatestRecord}
            onPipelineModelsRequired={() => {
              openModelManagerForPipeline();
              onError("Load ASR and forced aligner models before diarization.");
            }}
          />

          <section className="space-y-3">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <h2 className="text-xl font-semibold tracking-tight text-[var(--text-primary)]">
                  History
                </h2>
                <p className="mt-1 text-sm text-[var(--text-muted)]">
                  Open saved diarization runs on their own review pages.
                </p>
              </div>
            </div>

            <DiarizationHistoryTable
              records={visibleHistoryRecords}
              loading={historyLoading}
              error={historyError}
              onRefresh={() => void refreshHistory()}
              onOpenRecord={handleOpenRecord}
            />
          </section>
        </>
      )}

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title="Diarization Models"
        description="Manage diarization pipeline models for /v1/diarizations."
        models={pipelineModels}
        sections={pipelineModelGroups}
        loading={loading}
        selectedVariant={null}
        intentVariant={modalIntentModel}
        selectionMode="manage"
        downloadProgress={downloadProgress}
        onDownload={onDownload}
        onCancelDownload={onCancelDownload}
        onLoad={onLoad}
        onUnload={onUnload}
        onDelete={onDelete}
        onUseModel={onSelect}
        emptyMessage="No diarization pipeline models available for this route."
      />
    </PageShell>
  );
}
