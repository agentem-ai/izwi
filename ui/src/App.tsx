import { useState, useEffect, useCallback, useRef } from "react";
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
    Record<string, { percent: number; currentFile: string; status: string }>
  >({});

  // Use ref to track polling state and active downloads
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const activeDownloadsRef = useRef<Set<string>>(new Set());
  const eventSourcesRef = useRef<Record<string, EventSource>>({});

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
  }, []);

  // Smart polling: only poll when there are active operations, use ref to prevent duplicates
  useEffect(() => {
    const hasActiveOperations = models.some(
      (m) => m.status === "downloading" || m.status === "loading",
    );

    // Clear existing polling if no active operations
    if (!hasActiveOperations) {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      return;
    }

    // Only start polling if not already polling
    if (!pollingRef.current) {
      pollingRef.current = setInterval(loadModels, 3000); // Increased to 3s to reduce server load
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [models, loadModels]);

  // Connect to SSE for real-time download progress
  const connectDownloadStream = useCallback(
    (variant: string) => {
      // Close existing connection if any
      if (eventSourcesRef.current[variant]) {
        eventSourcesRef.current[variant].close();
      }

      const eventSource = new EventSource(
        `${api.baseUrl}/models/${variant}/download/progress`,
      );
      eventSourcesRef.current[variant] = eventSource;

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setDownloadProgress((prev) => ({
            ...prev,
            [variant]: {
              percent: data.percent,
              currentFile: data.current_file,
              status: data.status,
            },
          }));

          // Close connection when complete
          if (data.status === "completed" || data.status === "error") {
            eventSource.close();
            delete eventSourcesRef.current[variant];
            activeDownloadsRef.current.delete(variant);

            // Refresh models after completion
            loadModels();

            // Clear progress after delay
            setTimeout(() => {
              setDownloadProgress((prev) => {
                const { [variant]: _, ...rest } = prev;
                return rest;
              });
            }, 2000);
          }
        } catch (err) {
          console.error("Failed to parse progress event:", err);
        }
      };

      eventSource.onerror = (err) => {
        console.error("SSE error:", err);
        eventSource.close();
        delete eventSourcesRef.current[variant];
      };
    },
    [loadModels],
  );

  const handleDownload = async (variant: string) => {
    try {
      // Prevent duplicate downloads
      if (activeDownloadsRef.current.has(variant)) {
        return;
      }
      activeDownloadsRef.current.add(variant);

      setModels((prev) =>
        prev.map((m) =>
          m.variant === variant ? { ...m, status: "downloading" as const } : m,
        ),
      );

      // Start download
      const response = await api.downloadModel(variant);

      if (response.status === "started" || response.status === "downloading") {
        // Connect to SSE for progress updates
        connectDownloadStream(variant);
      } else {
        // Download already complete or not started
        activeDownloadsRef.current.delete(variant);
        await loadModels();
      }
    } catch (err: any) {
      console.error("Download failed:", err);
      activeDownloadsRef.current.delete(variant);
      setError(err.message || "Failed to download model. Please try again.");

      // Close SSE connection on error
      if (eventSourcesRef.current[variant]) {
        eventSourcesRef.current[variant].close();
        delete eventSourcesRef.current[variant];
      }

      await loadModels();
    }
  };

  const handleCancelDownload = async (variant: string) => {
    try {
      // Close SSE connection
      if (eventSourcesRef.current[variant]) {
        eventSourcesRef.current[variant].close();
        delete eventSourcesRef.current[variant];
      }

      activeDownloadsRef.current.delete(variant);

      await api.cancelDownload(variant);

      // Update UI immediately
      setDownloadProgress((prev) => {
        const { [variant]: _, ...rest } = prev;
        return rest;
      });

      setModels((prev) =>
        prev.map((m) =>
          m.variant === variant
            ? {
                ...m,
                status: "not_downloaded" as const,
                download_progress: null,
              }
            : m,
        ),
      );

      await loadModels();
    } catch (err: any) {
      console.error("Cancel failed:", err);
      setError(err.message || "Failed to cancel download.");
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
      // Refresh models after load completes
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
      // Refresh models after unload
      await loadModels();
      if (selectedModel === variant) {
        setSelectedModel(null);
      }
    } catch (err) {
      console.error("Unload failed:", err);
      setError("Failed to unload model. Please try again.");
    }
  };

  const handleDelete = async (variant: string) => {
    try {
      await api.deleteModel(variant);
      // Refresh models after delete
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
    onCancelDownload: handleCancelDownload,
    onLoad: handleLoad,
    onUnload: handleUnload,
    onDelete: handleDelete,
    onSelect: setSelectedModel,
    onError: setError,
    onRefresh: loadModels,
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
                onCancelDownload={handleCancelDownload}
                onLoad={handleLoad}
                onUnload={handleUnload}
                onDelete={handleDelete}
                onRefresh={loadModels}
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
