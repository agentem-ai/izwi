import { useEffect, useMemo, useState } from "react";
import { ModelInfo } from "../api";
import { RouteModelModal } from "../components/RouteModelModal";
import { VoiceClonePlayground } from "../components/VoiceClonePlayground";
import { VIEW_CONFIGS } from "../types";

interface VoiceCloningPageProps {
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

export function VoiceCloningPage({
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
}: VoiceCloningPageProps) {
  const viewConfig = VIEW_CONFIGS["voice-clone"];
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [modalIntentModel, setModalIntentModel] = useState<string | null>(null);
  const [autoCloseOnIntentReady, setAutoCloseOnIntentReady] = useState(false);

  const routeModels = useMemo(
    () =>
      models
        .filter((model) => !model.variant.includes("Tokenizer"))
        .filter((model) => viewConfig.modelFilter(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models, viewConfig],
  );

  const preferredModelOrder = [
    "Qwen3-TTS-12Hz-0.6B-Base-4bit",
    "Qwen3-TTS-12Hz-0.6B-Base-8bit",
    "Qwen3-TTS-12Hz-0.6B-Base-bf16",
    "Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen3-TTS-12Hz-1.7B-Base-4bit",
    "Qwen3-TTS-12Hz-1.7B-Base",
  ];

  const resolvedSelectedModel = (() => {
    if (selectedModel && routeModels.some((m) => m.variant === selectedModel)) {
      return selectedModel;
    }

    for (const variant of preferredModelOrder) {
      const readyPreferred = routeModels.find(
        (model) => model.variant === variant && model.status === "ready",
      );
      if (readyPreferred) {
        return readyPreferred.variant;
      }
    }

    const readyModel = routeModels.find((model) => model.status === "ready");
    if (readyModel) {
      return readyModel.variant;
    }

    for (const variant of preferredModelOrder) {
      const preferred = routeModels.find((model) => model.variant === variant);
      if (preferred) {
        return preferred.variant;
      }
    }

    return routeModels[0]?.variant ?? null;
  })();

  const selectedModelInfo =
    routeModels.find((model) => model.variant === resolvedSelectedModel) ?? null;
  const selectedModelReady = selectedModelInfo?.status === "ready";

  useEffect(() => {
    if (!isModelModalOpen || !modalIntentModel || !autoCloseOnIntentReady) {
      return;
    }
    const targetModel = routeModels.find(
      (model) => model.variant === modalIntentModel,
    );
    if (targetModel?.status === "ready") {
      setIsModelModalOpen(false);
      setAutoCloseOnIntentReady(false);
    }
  }, [autoCloseOnIntentReady, isModelModalOpen, modalIntentModel, routeModels]);

  const closeModelModal = () => {
    setIsModelModalOpen(false);
    setAutoCloseOnIntentReady(false);
  };

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

  const modelOptions = routeModels.map((model) => ({
    value: model.variant,
    label: model.variant,
    statusLabel: getStatusLabel(model.status),
    isReady: model.status === "ready",
  }));

  const handleModelSelect = (variant: string) => {
    const model = routeModels.find((m) => m.variant === variant);
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
        <h1 className="text-xl font-semibold text-white">Voice Cloning</h1>
      </div>

      <VoiceClonePlayground
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
          onError("Select and load a Base model to clone voices.");
        }}
      />

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title="Voice Cloning Models"
        description="Manage Base models for this route."
        models={routeModels}
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
        emptyMessage={viewConfig.emptyStateDescription}
      />
    </div>
  );
}
