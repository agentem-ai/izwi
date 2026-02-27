import { useEffect, useMemo, useState } from "react";
import { ModelInfo } from "../api";
import { DiarizationPlayground } from "../components/DiarizationPlayground";
import { PageHeader, PageShell } from "../components/PageShell";
import { RouteModelModal } from "../components/RouteModelModal";

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

function isDiarizationVariant(variant: string): boolean {
  const normalized = variant.toLowerCase();
  return normalized.includes("sortformer") || normalized.includes("diar");
}

function isPipelineAsrVariant(variant: string): boolean {
  return variant === "Qwen3-ASR-0.6B" || variant.startsWith("Qwen3-ASR-0.6B-");
}

function isPipelineAlignerVariant(variant: string): boolean {
  return (
    variant === "Qwen3-ForcedAligner-0.6B" ||
    variant === "Qwen3-ForcedAligner-0.6B-4bit"
  );
}

function isPipelineLlmVariant(variant: string): boolean {
  return variant === "Qwen3-1.7B" || variant === "Qwen3-1.7B-4bit";
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
        models: models
          .filter((model) => isDiarizationVariant(model.variant))
          .sort((a, b) => a.variant.localeCompare(b.variant)),
      },
      {
        key: "asr",
        title: "ASR",
        description: "Transcript generation model in the diarization pipeline.",
        models: models
          .filter((model) => isPipelineAsrVariant(model.variant))
          .sort((a, b) => a.variant.localeCompare(b.variant)),
      },
      {
        key: "aligner",
        title: "Forced Aligner",
        description: "Word timing alignment model for speaker attribution.",
        models: models
          .filter((model) => isPipelineAlignerVariant(model.variant))
          .sort((a, b) => a.variant.localeCompare(b.variant)),
      },
      {
        key: "llm",
        title: "Transcript Refiner",
        description: "LLM used to polish final diarized transcript output.",
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
    diarizationModels,
    isModelModalOpen,
    modalIntentModel,
  ]);

  const openModelManager = () => {
    setModalIntentModel(resolvedSelectedModel);
    setAutoCloseOnIntentReady(false);
    setIsModelModalOpen(true);
  };

  const modelOptions = diarizationModels.map((model) => ({
    value: model.variant,
    label: model.variant,
    statusLabel: getStatusLabel(model.status),
    isReady: model.status === "ready",
  }));

  const handleModelSelect = (variant: string) => {
    const model = diarizationModels.find((entry) => entry.variant === variant);
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
    <PageShell>
      <PageHeader
        title="Diarization"
        description="Separate speakers from audio streams and review timestamped transcript segments."
      />

      <DiarizationPlayground
        selectedModel={resolvedSelectedModel}
        selectedModelReady={selectedModelReady}
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

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title="Diarization Models"
        description="Manage pipeline models for /v1/diarization/records."
        models={pipelineModels}
        sections={pipelineModelGroups}
        canUseModel={isDiarizationVariant}
        loading={loading}
        selectedVariant={resolvedSelectedModel}
        intentVariant={modalIntentModel}
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
