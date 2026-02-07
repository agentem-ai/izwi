import { motion } from "framer-motion";
import { ModelInfo } from "../api";
import { ModelManager } from "../components/ModelManager";
import { ChatPlayground } from "../components/ChatPlayground";
import { VIEW_CONFIGS } from "../types";

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
  const viewConfig = VIEW_CONFIGS["chat"];

  const relevantSelectedModel = (() => {
    if (!selectedModel) return null;
    if (viewConfig.modelFilter(selectedModel)) {
      return selectedModel;
    }
    const readyModel = models.find(
      (m) => m.status === "ready" && viewConfig.modelFilter(m.variant),
    );
    return readyModel?.variant || null;
  })();

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-white">Chat</h1>
        <p className="text-sm text-gray-500 mt-1">
          Text-to-text conversation with Qwen3-0.6B-4bit
        </p>
      </div>

      <div className="grid lg:grid-cols-[320px,1fr] gap-4 lg:gap-6">
        <div className="card p-3 lg:p-4">
          <div className="mb-3">
            <h2 className="text-sm font-medium text-white">Models</h2>
            <p className="text-xs text-gray-500 mt-0.5">Chat compatible</p>
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
              onCancelDownload={onCancelDownload}
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

        <div>
          <ChatPlayground
            selectedModel={relevantSelectedModel}
            onModelRequired={() => onError("Please load Qwen3-0.6B-4bit first")}
          />
        </div>
      </div>
    </div>
  );
}
