import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Check,
  Loader2,
  Mic2,
  Plus,
  RefreshCw,
  Search,
  Sparkles,
  Trash2,
} from "lucide-react";
import type { ModelInfo, SavedVoiceSummary } from "@/api";
import { api } from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import type { VoicePickerItem } from "@/components/VoicePicker";
import {
  VOICE_ROUTE_META_COPY_CLASS,
} from "@/components/voiceRouteTypography";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { StatePanel } from "@/components/ui/state-panel";
import { StatusBadge } from "@/components/ui/status-badge";
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
  onAddNewVoice?: () => void;
}

type SavedVoiceFilter = "all" | "voice_cloning" | "voice_design";
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
  onAddNewVoice,
}: VoicesPageProps) {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<VoiceLibraryTab>("saved");
  const [search, setSearch] = useState("");
  const [savedVoiceFilter, setSavedVoiceFilter] =
    useState<SavedVoiceFilter>("all");
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
    openModelManager,
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

  const filteredSavedVoices = useMemo(() => {
    const normalizedQuery = search.trim().toLowerCase();
    return savedVoices.filter((voice) => {
      const matchesSource =
        savedVoiceFilter === "all" ||
        voice.source_route_kind === savedVoiceFilter;
      if (!matchesSource) {
        return false;
      }
      if (!normalizedQuery) {
        return true;
      }
      return (
        voice.name.toLowerCase().includes(normalizedQuery) ||
        voice.reference_text_preview.toLowerCase().includes(normalizedQuery)
      );
    });
  }, [savedVoices, savedVoiceFilter, search]);

  const filteredBuiltInVoices = useMemo(() => {
    const normalizedQuery = search.trim().toLowerCase();
    return builtInVoices.filter((voice) => {
      if (!normalizedQuery) {
        return true;
      }
      return (
        voice.name.toLowerCase().includes(normalizedQuery) ||
        voice.language.toLowerCase().includes(normalizedQuery) ||
        voice.description.toLowerCase().includes(normalizedQuery)
      );
    });
  }, [builtInVoices, search]);

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

  const savedVoiceItems: VoicePickerItem[] = filteredSavedVoices.map(
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

  const builtInVoiceItems: VoicePickerItem[] = filteredBuiltInVoices.map(
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
  const clonedVoiceCount = savedVoices.filter(
    (voice) => voice.source_route_kind === "voice_cloning",
  ).length;
  const designedVoiceCount = savedVoices.filter(
    (voice) => voice.source_route_kind === "voice_design",
  ).length;
  const totalBuiltInVoices = builtInVoices.length;
  const allVoiceItems = [...savedVoiceItems, ...builtInVoiceItems];
  const activeItems =
    activeTab === "all"
      ? allVoiceItems
      : activeTab === "saved"
        ? savedVoiceItems
        : builtInVoiceItems;
  const activeResultCount =
    activeTab === "all"
      ? filteredSavedVoices.length + filteredBuiltInVoices.length
      : activeTab === "saved"
        ? filteredSavedVoices.length
        : filteredBuiltInVoices.length;
  const activeResultsLabel =
    activeResultCount === 1 ? "1 result" : `${activeResultCount} results`;
  const showBuiltInMeta = activeTab === "built-in" || activeTab === "all";
  const showSavedVoiceError =
    savedVoicesError && (activeTab === "saved" || activeTab === "all");

  const handleAddNewVoice = () => {
    if (onAddNewVoice) {
      onAddNewVoice();
      return;
    }
    navigate("/voice-design");
  };

  const workspaceContent = (
    <div className="flex flex-col">
      <Tabs
        value={activeTab}
        onValueChange={(value) => setActiveTab(value as VoiceLibraryTab)}
        className="w-full"
      >
        <WorkspacePanel className="p-5 sm:p-6">
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

              <div className="flex flex-wrap items-center gap-2 text-xs">
                <StatusBadge>{activeResultsLabel.toUpperCase()}</StatusBadge>
                <StatusBadge>SAVED {totalSavedVoices}</StatusBadge>
                <StatusBadge>CLONED {clonedVoiceCount}</StatusBadge>
                <StatusBadge>DESIGNED {designedVoiceCount}</StatusBadge>
                <StatusBadge>BUILT-IN {totalBuiltInVoices}</StatusBadge>
                {savedVoiceFilter !== "all" &&
                (activeTab === "saved" || activeTab === "all") ? (
                  <StatusBadge>
                    {savedVoiceFilter === "voice_cloning" ? "CLONED" : "DESIGNED"}
                  </StatusBadge>
                ) : null}
                {showBuiltInMeta && resolvedSelectedModel ? (
                  <StatusBadge>{resolvedSelectedModel}</StatusBadge>
                ) : null}
                {showBuiltInMeta ? (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={openModelManager}
                    className="h-8 bg-[var(--bg-surface-1)] text-xs"
                  >
                    Model
                  </Button>
                ) : null}
              </div>
            </div>

            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              {(activeTab === "saved" || activeTab === "all") ? (
                <div className="flex flex-wrap items-center gap-2">
                  <div role="radiogroup" className="flex flex-wrap gap-2">
                    {(
                      [
                        ["all", "All"],
                        ["voice_cloning", "Cloned"],
                        ["voice_design", "Designed"],
                      ] as const
                    ).map(([value, label]) => (
                      <button
                        key={value}
                        type="button"
                        role="radio"
                        aria-checked={savedVoiceFilter === value}
                        onClick={() => setSavedVoiceFilter(value)}
                        className={cn(
                          "inline-flex h-8 items-center gap-2 rounded-full border px-3 text-xs font-semibold transition-colors",
                          savedVoiceFilter === value
                            ? "border-[var(--text-primary)] bg-[var(--text-primary)] text-[var(--text-on-accent)]"
                            : "border-[var(--border-strong)] bg-[var(--bg-surface-1)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-2)]",
                        )}
                      >
                        <Check className="h-3.5 w-3.5" />
                        {label}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div />
              )}

              <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row sm:items-center">
                <div className="relative w-full sm:w-[20rem]">
                  <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--text-muted)]" />
                  <Input
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                    placeholder="Search voices by name or notes"
                    className="pl-9"
                  />
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => void loadSavedVoices()}
                  disabled={savedVoicesLoading}
                  className="h-9"
                >
                  {savedVoicesLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                  Refresh
                </Button>
                <Button
                  onClick={handleAddNewVoice}
                  className="h-9 rounded-[var(--radius-pill)] text-sm"
                >
                  <Plus className="h-4 w-4" />
                  Add New Voice
                </Button>
              </div>
            </div>
          </div>

          {showBuiltInMeta ? (
            <div
              className={cn(
                VOICE_ROUTE_META_COPY_CLASS,
                "mt-3 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3.5 py-2.5",
              )}
            >
              {selectedModelReady
                ? `Built-in previews are using ${selectedModelInfo?.variant ?? "the selected model"}.`
                : "Select and load a supported voice model to generate built-in previews."}
            </div>
          ) : null}

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
                  : "Save a voice from voice cloning or voice design, or clear your search/filter to browse more voices."
                : activeTab === "saved"
                  ? savedVoicesLoading
                    ? "Fetching your reusable cloned and designed voices."
                    : "Save a result from voice cloning or voice design to build a reusable voice library."
                  : routeModels.length === 0
                    ? "Load a CustomVoice or Kokoro model to browse built-in voices."
                    : "Try a different built-in voice model or search term."
            }
            className="mt-5"
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
