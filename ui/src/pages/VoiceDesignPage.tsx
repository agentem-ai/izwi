import { useEffect, useMemo, useState } from "react";
import { ModelInfo } from "../api";
import { RouteModelModal } from "../components/RouteModelModal";
import { VoiceDesignPlayground } from "../components/VoiceDesignPlayground";
import { VIEW_CONFIGS } from "../types";

interface VoiceDesignPageProps {
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

export function VoiceDesignPage({
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
}: VoiceDesignPageProps) {
  const viewConfig = VIEW_CONFIGS["voice-design"];
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [modalIntentModel, setModalIntentModel] = useState<string | null>(null);

  const routeModels = useMemo(
    () =>
      models
        .filter((model) => !model.variant.includes("Tokenizer"))
        .filter((model) => viewConfig.modelFilter(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models, viewConfig],
  );

  const preferredModelOrder = [
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
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
    if (!isModelModalOpen || !modalIntentModel) {
      return;
    }
    const targetModel = routeModels.find(
      (model) => model.variant === modalIntentModel,
    );
    if (targetModel?.status === "ready") {
      setIsModelModalOpen(false);
    }
  }, [isModelModalOpen, modalIntentModel, routeModels]);

  const openModelManager = () => {
    setModalIntentModel(resolvedSelectedModel);
    setIsModelModalOpen(true);
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-white">Voice Design</h1>
      </div>

      <VoiceDesignPlayground
        selectedModel={resolvedSelectedModel}
        selectedModelReady={selectedModelReady}
        modelLabel={selectedModelInfo?.variant ?? null}
        onOpenModelManager={openModelManager}
        onModelRequired={() => {
          setModalIntentModel(resolvedSelectedModel);
          setIsModelModalOpen(true);
          onError("Select and load a VoiceDesign model to continue.");
        }}
      />

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={() => setIsModelModalOpen(false)}
        title="Voice Design Models"
        description="Manage VoiceDesign models for this route."
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
