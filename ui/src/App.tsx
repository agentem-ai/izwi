import { useState, useEffect, useCallback } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { api, ModelInfo } from "./api";
import { Layout } from "./components/Layout";
import {
  TextToSpeechPage,
  VoiceCloningPage,
  VoiceDesignPage,
  TranscriptionPage,
  MyModelsPage,
} from "./pages";

function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<
    Record<string, number>
  >({});

  const loadModels = useCallback(async () => {
    try {
      const response = await api.listModels();
      setModels(response.models);

      // Auto-select first ready model
      const readyModel = response.models.find((m) => m.status === "ready");
      if (readyModel && !selectedModel) {
        setSelectedModel(readyModel.variant);
      }
    } catch (err) {
      console.error("Failed to load models:", err);
    }
  }, [selectedModel]);

  useEffect(() => {
    const init = async () => {
      setLoading(true);
      await loadModels();
      setLoading(false);
    };
    init();

    // Poll for model status updates
    const interval = setInterval(loadModels, 5000);
    return () => clearInterval(interval);
  }, [loadModels]);

  const handleDownload = async (variant: string) => {
    try {
      setModels((prev) =>
        prev.map((m) =>
          m.variant === variant ? { ...m, status: "downloading" as const } : m,
        ),
      );

      // Simulate progress updates (real implementation would use SSE/WebSocket)
      const progressInterval = setInterval(() => {
        setDownloadProgress((prev) => {
          const current = prev[variant] || 0;
          if (current >= 95) {
            clearInterval(progressInterval);
            return prev;
          }
          return {
            ...prev,
            [variant]: Math.min(current + Math.random() * 15, 95),
          };
        });
      }, 500);

      await api.downloadModel(variant);

      clearInterval(progressInterval);
      setDownloadProgress((prev) => ({ ...prev, [variant]: 100 }));

      await loadModels();

      // Clear progress after a delay
      setTimeout(() => {
        setDownloadProgress((prev) => {
          const { [variant]: _, ...rest } = prev;
          return rest;
        });
      }, 1000);
    } catch (err) {
      console.error("Download failed:", err);
      setError("Failed to download model. Please try again.");
      await loadModels();
    }
  };

  const handleLoad = async (variant: string) => {
    try {
      setModels((prev) =>
        prev.map((m) =>
          m.variant === variant ? { ...m, status: "loading" as const } : m,
        ),
      );

      await api.loadModel(variant);
      await loadModels();
      setSelectedModel(variant);
    } catch (err) {
      console.error("Load failed:", err);
      setError("Failed to load model. Please try again.");
      await loadModels();
    }
  };

  const handleUnload = async (variant: string) => {
    try {
      await api.unloadModel(variant);
      await loadModels();
      if (selectedModel === variant) {
        setSelectedModel(null);
      }
    } catch (err) {
      console.error("Unload failed:", err);
    }
  };

  const handleDelete = async (variant: string) => {
    try {
      await api.deleteModel(variant);
      await loadModels();
      if (selectedModel === variant) {
        setSelectedModel(null);
      }
    } catch (err) {
      console.error("Delete failed:", err);
      setError("Failed to delete model. Please try again.");
    }
  };

  const readyModelsCount = models.filter((m) => m.status === "ready").length;

  const pageProps = {
    models,
    selectedModel,
    loading,
    downloadProgress,
    onDownload: handleDownload,
    onLoad: handleLoad,
    onUnload: handleUnload,
    onDelete: handleDelete,
    onSelect: setSelectedModel,
    onError: setError,
  };

  return (
    <BrowserRouter>
      <Routes>
        <Route
          element={
            <Layout
              error={error}
              onErrorDismiss={() => setError(null)}
              readyModelsCount={readyModelsCount}
            />
          }
        >
          <Route
            path="/text-to-speech"
            element={<TextToSpeechPage {...pageProps} />}
          />
          <Route
            path="/voice-cloning"
            element={<VoiceCloningPage {...pageProps} />}
          />
          <Route
            path="/voice-design"
            element={<VoiceDesignPage {...pageProps} />}
          />
          <Route
            path="/transcription"
            element={<TranscriptionPage {...pageProps} />}
          />
          <Route
            path="/my-models"
            element={
              <MyModelsPage
                models={models}
                loading={loading}
                downloadProgress={downloadProgress}
                onDownload={handleDownload}
                onLoad={handleLoad}
                onUnload={handleUnload}
                onDelete={handleDelete}
              />
            }
          />
          <Route path="/" element={<Navigate to="/text-to-speech" replace />} />
          <Route path="*" element={<Navigate to="/text-to-speech" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
