import { motion } from "framer-motion";
import { ModelInfo } from "../api";
import { ModelManager } from "../components/ModelManager";
import { TranscriptionPlayground } from "../components/TranscriptionPlayground";
import { VIEW_CONFIGS } from "../types";

interface TranscriptionPageProps {
  models: ModelInfo[];
  selectedModel: string | null;
  loading: boolean;
  downloadProgress: Record<string, number>;
  onDownload: (variant: string) => void;
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
  onLoad,
  onUnload,
  onDelete,
  onSelect,
  onError,
}: TranscriptionPageProps) {
  const viewConfig = VIEW_CONFIGS["transcription"];

  const relevantSelectedModel = (() => {
    if (!selectedModel) return null;
    if (viewConfig.modelFilter(selectedModel)) {
      return selectedModel;
    }
    const readyModel = models.find(
      (m) => m.status === "ready" && viewConfig.modelFilter(m.variant)
    );
    return readyModel?.variant || null;
  })();

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-white">Transcription</h1>
        <p className="text-sm text-gray-500 mt-1">
          Speech-to-text with Qwen3-ASR supporting 52 languages
        </p>
      </div>

      <div className="grid lg:grid-cols-[320px,1fr] gap-6">
        {/* Models sidebar */}
        <div className="card p-4">
          <div className="mb-3">
            <h2 className="text-sm font-medium text-white">Models</h2>
            <p className="text-xs text-gray-500 mt-0.5">Qwen3-ASR compatible</p>
          </div>

          {loading ? (
            <div className="flex flex-col items-center justify-center py-12 gap-2">
              <motion.div
                className="w-6 h-6 border-2 border-white border-t-transparent rounded-full"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
              <p className="text-xs text-gray-400">Loading...</p>
            </div>
          ) : (
            <ModelManager
              models={models}
              selectedModel={relevantSelectedModel}
              onDownload={onDownload}
              onLoad={onLoad}
              onUnload={onUnload}
              onDelete={onDelete}
              onSelect={onSelect}
              downloadProgress={downloadProgress}
              modelFilter={viewConfig.modelFilter}
              emptyStateTitle={viewConfig.emptyStateTitle}
              emptyStateDescription={viewConfig.emptyStateDescription}
            />
          )}
        </div>

        {/* Playground */}
        <div>
          <TranscriptionPlayground
            selectedModel={relevantSelectedModel}
            onModelRequired={() => onError("Please load a Qwen3-ASR model first")}
          />
        </div>
      </div>
    </div>
  );
}
