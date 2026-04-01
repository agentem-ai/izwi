import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Loader2,
  Mic2,
  Sparkles,
  Trash2,
} from "lucide-react";
import type { ModelInfo, SavedVoiceSummary } from "@/api";
import { api } from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { StatePanel } from "@/components/ui/state-panel";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { WorkspacePanel } from "@/components/ui/workspace";
import { getSpeakerProfilesForVariant, isLfm25AudioVariant } from "@/types";
import {
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";
import { VoiceLibraryTable } from "@/features/voices/components/VoiceLibraryTable";
import { type VoiceLibraryItem } from "@/features/voices/types";
import { cn } from "@/lib/utils";

interface VoicesPageProps {
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
  embedded?: boolean;
}

type VoiceLibraryTab = "all" | "saved" | "built-in";

const BUILT_IN_PREVIEW_TEXT = {
  chinese: "你好，欢迎使用 Izwi 的内置语音预览。",
  japanese: "こんにちは。Izwi の音声プレビューです。",
  korean: "안녕하세요. Izwi 음성 미리보기입니다.",
  spanish: "Hola. Esta es una muestra de voz de Izwi.",
  french: "Bonjour. Ceci est un apercu vocal Izwi.",
  hindi: "Namaste. Yeh Izwi ki awaaz preview hai.",
  italian: "Ciao. Questa e una voce di anteprima Izwi.",
  portuguese: "Ola. Esta e uma amostra de voz do Izwi.",
  english: "Hello. This is an Izwi built-in voice preview.",
} as const;

function formatRelativeDate(timestampMs: number): string {
  const value = new Date(timestampMs);
  if (Number.isNaN(value.getTime())) {
    return "Unknown date";
  }
  return value.toLocaleDateString([], {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function savedVoiceSourceLabel(
  source: SavedVoiceSummary["source_route_kind"],
): string {
  switch (source) {
    case "voice_cloning":
      return "Cloned voice";
    case "voice_design":
      return "Designed voice";
    default:
      return "Saved voice";
  }
}

function previewTextForLanguage(language: string): string {
  const normalized = language.toLowerCase();
  if (normalized.includes("chinese")) return BUILT_IN_PREVIEW_TEXT.chinese;
  if (normalized.includes("japanese")) return BUILT_IN_PREVIEW_TEXT.japanese;
  if (normalized.includes("korean")) return BUILT_IN_PREVIEW_TEXT.korean;
  if (normalized.includes("spanish")) return BUILT_IN_PREVIEW_TEXT.spanish;
  if (normalized.includes("french")) return BUILT_IN_PREVIEW_TEXT.french;
  if (normalized.includes("hindi")) return BUILT_IN_PREVIEW_TEXT.hindi;
  if (normalized.includes("italian")) return BUILT_IN_PREVIEW_TEXT.italian;
  if (normalized.includes("portuguese"))
    return BUILT_IN_PREVIEW_TEXT.portuguese;
  return BUILT_IN_PREVIEW_TEXT.english;
}

export function VoicesPage({
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
  embedded = false,
}: VoicesPageProps) {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<VoiceLibraryTab>("saved");
  const [savedVoices, setSavedVoices] = useState<SavedVoiceSummary[]>([]);
  const [savedVoicesLoading, setSavedVoicesLoading] = useState(true);
  const [savedVoicesError, setSavedVoicesError] = useState<string | null>(null);
  const [deletingVoiceId, setDeletingVoiceId] = useState<string | null>(null);
  const [previewLoadingVoiceId, setPreviewLoadingVoiceId] = useState<
    string | null
  >(null);
  const [previewUrls, setPreviewUrls] = useState<Record<string, string>>({});
  const previewUrlsRef = useRef<Record<string, string>>({});

  const {
    routeModels,
    resolvedSelectedModel,
    selectedModelInfo,
    selectedModelReady,
    isModelModalOpen,
    intentVariant,
    closeModelModal,
    requestModel,
    handleModelSelect,
  } = useRouteModelSelection({
    models,
    selectedModel,
    onSelect,
    modelFilter: (variant) => {
      const match = models.find((model) => model.variant === variant);
      return (
        match?.speech_capabilities?.supports_builtin_voices === true &&
        !isLfm25AudioVariant(variant) &&
        !variant.includes("Tokenizer")
      );
    },
    resolveSelectedModel: (routeModels, currentModel) =>
      resolvePreferredRouteModel({
        models: routeModels,
        selectedModel: currentModel,
        preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
      }),
  });

  useEffect(() => {
    previewUrlsRef.current = previewUrls;
  }, [previewUrls]);

  useEffect(() => {
    return () => {
      Object.values(previewUrlsRef.current).forEach((url) => {
        if (url.startsWith("blob:")) {
          URL.revokeObjectURL(url);
        }
      });
    };
  }, []);

  const loadSavedVoices = async () => {
    setSavedVoicesLoading(true);
    setSavedVoicesError(null);
    try {
      const records = await api.listSavedVoices();
      setSavedVoices(records);
    } catch (error) {
      setSavedVoicesError(
        error instanceof Error ? error.message : "Failed to load saved voices.",
      );
    } finally {
      setSavedVoicesLoading(false);
    }
  };

  useEffect(() => {
    void loadSavedVoices();
  }, []);

  const builtInVoices = useMemo(
    () => getSpeakerProfilesForVariant(resolvedSelectedModel),
    [resolvedSelectedModel],
  );

  const handleUseSavedVoice = (voiceId: string) => {
    navigate(`/text-to-speech?voiceId=${encodeURIComponent(voiceId)}`);
  };

  const handleUseBuiltInVoice = (speaker: string) => {
    const params = new URLSearchParams();
    params.set("speaker", speaker);
    if (resolvedSelectedModel) {
      params.set("model", resolvedSelectedModel);
    }
    navigate(`/text-to-speech?${params.toString()}`);
  };

  const handleDeleteVoice = async (voiceId: string) => {
    setDeletingVoiceId(voiceId);
    try {
      await api.deleteSavedVoice(voiceId);
      setSavedVoices((current) =>
        current.filter((voice) => voice.id !== voiceId),
      );
    } catch (error) {
      onError(
        error instanceof Error
          ? error.message
          : "Failed to delete saved voice.",
      );
    } finally {
      setDeletingVoiceId(null);
    }
  };

  const handlePreviewBuiltInVoice = async (
    voiceId: string,
    language: string,
  ) => {
    if (!resolvedSelectedModel) {
      requestModel();
      onError("Select and load a built-in voice model to generate previews.");
      return;
    }
    if (previewUrls[voiceId]) {
      return;
    }
    if (!selectedModelReady) {
      requestModel(resolvedSelectedModel);
      onError("Load the selected voice model before generating a preview.");
      return;
    }

    setPreviewLoadingVoiceId(voiceId);
    try {
      const result = await api.generateTTSWithStats({
        model_id: resolvedSelectedModel,
        text: previewTextForLanguage(language),
        speaker: voiceId,
      });
      const url = URL.createObjectURL(result.audioBlob);
      setPreviewUrls((current) => ({ ...current, [voiceId]: url }));
    } catch (error) {
      onError(
        error instanceof Error
          ? error.message
          : "Failed to generate built-in voice preview.",
      );
    } finally {
      setPreviewLoadingVoiceId(null);
    }
  };

  const savedVoiceItems: VoiceLibraryItem[] = savedVoices.map(
    (voice) => ({
      id: voice.id,
      name: voice.name,
      categoryLabel: savedVoiceSourceLabel(voice.source_route_kind),
      description: voice.reference_text_preview,
      meta: [
        `${voice.reference_text_chars} chars`,
        formatRelativeDate(voice.updated_at || voice.created_at),
      ],
      previewUrl: api.savedVoiceAudioUrl(voice.id),
      actions: (
        <>
          <Button
            size="sm"
            className="h-8 rounded-[var(--radius-pill)] px-3.5 text-xs font-semibold"
            onClick={(event) => {
              event.stopPropagation();
              handleUseSavedVoice(voice.id);
            }}
          >
            <Mic2 className="h-4 w-4" />
            Use in TTS
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-8 rounded-[var(--radius-pill)] border-[var(--border-strong)] bg-[var(--bg-surface-1)]/72 px-3.5 text-xs font-semibold"
            onClick={(event) => {
              event.stopPropagation();
              void handleDeleteVoice(voice.id);
            }}
            disabled={deletingVoiceId === voice.id}
          >
            {deletingVoiceId === voice.id ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Trash2 className="h-4 w-4" />
            )}
            Delete
          </Button>
        </>
      ),
    }),
  );

  const builtInVoiceItems: VoiceLibraryItem[] = builtInVoices.map(
    (voice) => ({
      id: voice.id,
      name: voice.name,
      categoryLabel: "Built-in voice",
      description: voice.description,
      meta: [voice.language, selectedModelInfo?.variant ?? "Model preview"],
      previewUrl: previewUrls[voice.id] ?? null,
      previewLoading: previewLoadingVoiceId === voice.id,
      previewMessage: previewUrls[voice.id]
        ? null
        : previewLoadingVoiceId === voice.id
          ? "Generating a preview sample for this speaker."
          : "Generate a preview sample to audition this built-in voice.",
      actions: (
        <>
          <Button
            size="sm"
            onClick={(event) => {
              event.stopPropagation();
              handleUseBuiltInVoice(voice.id);
            }}
            className="h-8 rounded-[var(--radius-pill)] px-3.5 text-xs font-semibold"
          >
            <Mic2 className="h-4 w-4" />
            Use in TTS
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={(event) => {
              event.stopPropagation();
              void handlePreviewBuiltInVoice(voice.id, voice.language);
            }}
            disabled={previewLoadingVoiceId === voice.id}
            className="h-8 rounded-[var(--radius-pill)] border-[var(--border-strong)] bg-[var(--bg-surface-1)]/72 px-3.5 text-xs font-semibold"
          >
            {previewLoadingVoiceId === voice.id ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            Preview
          </Button>
        </>
      ),
    }),
  );

  const totalSavedVoices = savedVoices.length;
  const totalBuiltInVoices = builtInVoices.length;
  const allVoiceItems = [...savedVoiceItems, ...builtInVoiceItems];
  const activeItems =
    activeTab === "all"
      ? allVoiceItems
      : activeTab === "saved"
        ? savedVoiceItems
        : builtInVoiceItems;
  const showSavedVoiceError =
    savedVoicesError && (activeTab === "saved" || activeTab === "all");

  const workspaceContent = (
    <div className="flex flex-col">
      <Tabs
        value={activeTab}
        onValueChange={(value) => setActiveTab(value as VoiceLibraryTab)}
        className="w-full"
      >
        <WorkspacePanel className={cn("p-5 sm:p-6", embedded && "p-4 sm:p-5")}>
          <div className="flex flex-col gap-4 border-b border-[var(--border-muted)] pb-4">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <TabsList className="grid h-10 w-full max-w-[30rem] grid-cols-3 overflow-hidden rounded-[var(--radius-pill)] border-[var(--border-strong)] bg-[var(--bg-surface-2)] p-[2px] shadow-none">
                <TabsTrigger
                  value="all"
                  className="h-full rounded-[var(--radius-pill)] px-3 text-[13px] font-semibold text-[var(--text-muted)] data-[state=active]:bg-[var(--bg-surface-1)] data-[state=active]:text-[var(--text-primary)] data-[state=active]:shadow-none"
                >
                  All Voices
                </TabsTrigger>
                <TabsTrigger
                  value="saved"
                  className="h-full gap-2 rounded-[var(--radius-pill)] px-3 text-[13px] font-semibold text-[var(--text-muted)] data-[state=active]:bg-[var(--bg-surface-1)] data-[state=active]:text-[var(--text-primary)] data-[state=active]:shadow-none"
                >
                  <span>My Voices</span>
                  <span className="rounded-full border border-current/20 px-2 py-0.5 text-[10px] font-semibold">
                    {totalSavedVoices}
                  </span>
                </TabsTrigger>
                <TabsTrigger
                  value="built-in"
                  className="h-full gap-2 rounded-[var(--radius-pill)] px-3 text-[13px] font-semibold text-[var(--text-muted)] data-[state=active]:bg-[var(--bg-surface-1)] data-[state=active]:text-[var(--text-primary)] data-[state=active]:shadow-none"
                >
                  <span>Built-in Voices</span>
                  <span className="rounded-full border border-current/20 px-2 py-0.5 text-[10px] font-semibold">
                    {totalBuiltInVoices}
                  </span>
                </TabsTrigger>
              </TabsList>
            </div>

          </div>

          {showSavedVoiceError ? (
            <div className="mt-4">
              <StatePanel
                title="Saved voices unavailable"
                description={savedVoicesError}
                tone="danger"
              />
            </div>
          ) : null}

          <VoiceLibraryTable
            items={activeItems}
            emptyTitle={
              activeTab === "all"
                ? savedVoicesLoading
                  ? "Loading voices"
                  : "No voices yet"
                : activeTab === "saved"
                  ? savedVoicesLoading
                    ? "Loading saved voices"
                    : "No saved voices yet"
                  : "No built-in voices available"
            }
            emptyDescription={
              activeTab === "all"
                ? routeModels.length === 0
                  ? "Save a voice or load a supported built-in voice model to populate the library."
                  : "Save a voice from voice cloning or voice design to build out your library."
                : activeTab === "saved"
                  ? savedVoicesLoading
                    ? "Fetching your reusable cloned and designed voices."
                    : "Save a result from voice cloning or voice design to build a reusable voice library."
                  : routeModels.length === 0
                    ? "Load a CustomVoice or Kokoro model to browse built-in voices."
                    : "Try a different built-in voice model."
            }
            className="mt-5"
            compact={embedded}
          />
        </WorkspacePanel>
      </Tabs>
    </div>
  );

  const routeModelModal = (
    <RouteModelModal
      isOpen={isModelModalOpen}
      onClose={closeModelModal}
      title="Built-in Voice Models"
      description="Manage the voice models that expose built-in speaker libraries."
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
      onUseModel={handleModelSelect}
      emptyMessage="Load a CustomVoice or Kokoro model to browse built-in voices."
    />
  );

  if (embedded) {
    return (
      <>
        {workspaceContent}
        {routeModelModal}
      </>
    );
  }

  return (
    <PageShell>
      <PageHeader
        title="Voices"
        description="Manage, browse, and use your saved, cloned, and designed voices for text-to-speech."
      />
      {workspaceContent}
      {routeModelModal}
    </PageShell>
  );
}
