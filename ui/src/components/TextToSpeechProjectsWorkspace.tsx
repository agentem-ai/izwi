import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Download,
  FileAudio,
  FilePlus2,
  Loader2,
  PencilLine,
  Play,
  Settings2,
  Trash2,
  Upload,
  Waves,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  api,
  type ModelInfo,
  type SavedVoiceSummary,
  type TtsProjectRecord,
  type TtsProjectSummary,
  type TtsProjectVoiceMode,
} from "@/api";
import {
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  VOICE_CLONING_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { VoicePicker, type VoicePickerItem } from "@/components/VoicePicker";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useDownloadIndicator } from "@/utils/useDownloadIndicator";
import { getSpeakerProfilesForVariant } from "@/types";

interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface TextToSpeechProjectsWorkspaceProps {
  selectedModel: string | null;
  selectedModelInfo: ModelInfo | null;
  availableModels: ModelInfo[];
  modelOptions?: ModelOption[];
  onSelectModel?: (variant: string) => void;
  onOpenModelManager?: () => void;
  onModelRequired: () => void;
  onError: (message: string) => void;
}

const SAVED_VOICE_RENDERER_PREFERRED_MODELS = [
  ...VOICE_CLONING_PREFERRED_MODELS,
  "LFM2.5-Audio-1.5B-4bit",
  "LFM2.5-Audio-1.5B",
] as const;

function formatRelativeDate(timestampMs: number): string {
  const value = new Date(timestampMs);
  if (Number.isNaN(value.getTime())) {
    return "Unknown";
  }
  return value.toLocaleDateString([], {
    month: "short",
    day: "numeric",
  });
}

function sourceLabel(source: SavedVoiceSummary["source_route_kind"]): string {
  return source === "voice_cloning" ? "Cloned voice" : "Designed voice";
}

function projectAudioFilename(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug ? `${slug}.wav` : "tts-project.wav";
}

export function TextToSpeechProjectsWorkspace({
  selectedModel,
  selectedModelInfo,
  availableModels,
  modelOptions = [],
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
  onError,
}: TextToSpeechProjectsWorkspaceProps) {
  const [projects, setProjects] = useState<TtsProjectSummary[]>([]);
  const [projectsLoading, setProjectsLoading] = useState(false);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [selectedProject, setSelectedProject] = useState<TtsProjectRecord | null>(
    null,
  );
  const [projectLoading, setProjectLoading] = useState(false);
  const [savedVoices, setSavedVoices] = useState<SavedVoiceSummary[]>([]);
  const [savedVoicesLoading, setSavedVoicesLoading] = useState(false);
  const [savedVoicesError, setSavedVoicesError] = useState<string | null>(null);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectText, setNewProjectText] = useState("");
  const [newProjectFilename, setNewProjectFilename] = useState("");
  const [creatingProject, setCreatingProject] = useState(false);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [workspaceStatus, setWorkspaceStatus] = useState<{
    tone: "success" | "error";
    message: string;
  } | null>(null);
  const [projectName, setProjectName] = useState("");
  const [projectModelId, setProjectModelId] = useState(selectedModel ?? "");
  const [projectVoiceMode, setProjectVoiceMode] =
    useState<TtsProjectVoiceMode>("built_in");
  const [projectSpeaker, setProjectSpeaker] = useState("Vivian");
  const [projectSavedVoiceId, setProjectSavedVoiceId] = useState("");
  const [projectSpeed, setProjectSpeed] = useState(1);
  const [savingProject, setSavingProject] = useState(false);
  const [savingSegmentId, setSavingSegmentId] = useState<string | null>(null);
  const [renderingSegmentId, setRenderingSegmentId] = useState<string | null>(
    null,
  );
  const [renderingAll, setRenderingAll] = useState(false);
  const [deletingProject, setDeletingProject] = useState(false);
  const [segmentDrafts, setSegmentDrafts] = useState<Record<string, string>>({});
  const fileInputRef = useRef<HTMLInputElement>(null);
  const {
    downloadState,
    downloadMessage,
    isDownloading,
    beginDownload,
    completeDownload,
    failDownload,
  } = useDownloadIndicator();

  const currentProjectModelInfo = useMemo(
    () =>
      availableModels.find((model) => model.variant === projectModelId) ?? null,
    [availableModels, projectModelId],
  );
  const currentProjectCapabilities =
    currentProjectModelInfo?.speech_capabilities ?? null;
  const supportsBuiltInVoices =
    currentProjectCapabilities?.supports_builtin_voices ?? false;
  const supportsSavedVoices =
    currentProjectCapabilities?.supports_reference_voice ?? false;

  const builtInCompatibleModels = useMemo(
    () =>
      availableModels.filter(
        (model) => model.speech_capabilities?.supports_builtin_voices,
      ),
    [availableModels],
  );
  const savedVoiceCompatibleModels = useMemo(
    () =>
      availableModels.filter(
        (model) => model.speech_capabilities?.supports_reference_voice,
      ),
    [availableModels],
  );

  const compatibleModelOptions = useMemo(() => {
    return modelOptions.filter((option) => {
      const model = availableModels.find((candidate) => candidate.variant === option.value);
      const capabilities = model?.speech_capabilities;
      if (!capabilities) {
        return false;
      }
      return projectVoiceMode === "saved"
        ? capabilities.supports_reference_voice
        : capabilities.supports_builtin_voices;
    });
  }, [availableModels, modelOptions, projectVoiceMode]);

  const availableSpeakers = useMemo(
    () =>
      supportsBuiltInVoices ? getSpeakerProfilesForVariant(projectModelId) : [],
    [projectModelId, supportsBuiltInVoices],
  );

  const selectedProjectSummary = useMemo(
    () =>
      projects.find((project) => project.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  );

  const projectDirty = useMemo(() => {
    if (!selectedProject) {
      return false;
    }
    return (
      projectName.trim() !== selectedProject.name ||
      projectModelId !== (selectedProject.model_id ?? "") ||
      projectVoiceMode !== selectedProject.voice_mode ||
      projectSpeaker !== (selectedProject.speaker ?? "") ||
      projectSavedVoiceId !== (selectedProject.saved_voice_id ?? "") ||
      Number(projectSpeed.toFixed(2)) !==
        Number((selectedProject.speed ?? 1).toFixed(2))
    );
  }, [
    projectModelId,
    projectName,
    projectSavedVoiceId,
    projectSpeaker,
    projectSpeed,
    projectVoiceMode,
    selectedProject,
  ]);

  const loadProjects = useCallback(async () => {
    setProjectsLoading(true);
    try {
      const records = await api.listTtsProjects();
      setProjects(records);
      setSelectedProjectId((current) => {
        if (current && records.some((project) => project.id === current)) {
          return current;
        }
        return records[0]?.id ?? null;
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to load TTS projects.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setProjectsLoading(false);
    }
  }, [onError]);

  const loadSavedVoices = useCallback(async () => {
    setSavedVoicesLoading(true);
    setSavedVoicesError(null);
    try {
      setSavedVoices(await api.listSavedVoices());
    } catch (err) {
      setSavedVoicesError(
        err instanceof Error ? err.message : "Failed to load saved voices.",
      );
    } finally {
      setSavedVoicesLoading(false);
    }
  }, []);

  const loadProject = useCallback(
    async (projectId: string) => {
      setProjectLoading(true);
      try {
        setSelectedProject(await api.getTtsProject(projectId));
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to load the TTS project.";
        setWorkspaceError(message);
        onError(message);
      } finally {
        setProjectLoading(false);
      }
    },
    [onError],
  );

  useEffect(() => {
    void loadProjects();
    void loadSavedVoices();
  }, [loadProjects, loadSavedVoices]);

  useEffect(() => {
    if (!selectedProjectId) {
      setSelectedProject(null);
      return;
    }
    void loadProject(selectedProjectId);
  }, [loadProject, selectedProjectId]);

  useEffect(() => {
    if (!selectedProject) {
      setProjectName("");
      setProjectModelId(selectedModel ?? "");
      setProjectVoiceMode("built_in");
      setProjectSpeaker("Vivian");
      setProjectSavedVoiceId("");
      setProjectSpeed(1);
      setSegmentDrafts({});
      return;
    }

    setProjectName(selectedProject.name);
    setProjectModelId(selectedProject.model_id ?? selectedModel ?? "");
    setProjectVoiceMode(selectedProject.voice_mode);
    setProjectSpeaker(selectedProject.speaker ?? "Vivian");
    setProjectSavedVoiceId(selectedProject.saved_voice_id ?? "");
    setProjectSpeed(selectedProject.speed ?? 1);
    setSegmentDrafts(
      Object.fromEntries(
        selectedProject.segments.map((segment) => [segment.id, segment.text]),
      ),
    );
  }, [selectedModel, selectedProject]);

  useEffect(() => {
    if (projectVoiceMode === "saved" && !supportsSavedVoices) {
      const nextModel = resolvePreferredRouteModel({
        models: savedVoiceCompatibleModels,
        selectedModel: projectModelId || selectedModel,
        preferredVariants: SAVED_VOICE_RENDERER_PREFERRED_MODELS,
      });
      if (nextModel && nextModel !== projectModelId) {
        setProjectModelId(nextModel);
      }
    }
  }, [
    projectModelId,
    projectVoiceMode,
    savedVoiceCompatibleModels,
    selectedModel,
    supportsSavedVoices,
  ]);

  useEffect(() => {
    if (projectVoiceMode === "built_in" && !supportsBuiltInVoices) {
      const nextModel = resolvePreferredRouteModel({
        models: builtInCompatibleModels,
        selectedModel: projectModelId || selectedModel,
        preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
      });
      if (nextModel && nextModel !== projectModelId) {
        setProjectModelId(nextModel);
      }
    }
  }, [
    builtInCompatibleModels,
    projectModelId,
    projectVoiceMode,
    selectedModel,
    supportsBuiltInVoices,
  ]);

  useEffect(() => {
    if (
      projectVoiceMode === "built_in" &&
      availableSpeakers.length > 0 &&
      !availableSpeakers.some((voice) => voice.id === projectSpeaker)
    ) {
      setProjectSpeaker(availableSpeakers[0]?.id ?? "Vivian");
    }
  }, [availableSpeakers, projectSpeaker, projectVoiceMode]);

  useEffect(() => {
    if (projectVoiceMode === "saved" && !projectSavedVoiceId && savedVoices.length > 0) {
      setProjectSavedVoiceId(savedVoices[0]?.id ?? "");
    }
  }, [projectSavedVoiceId, projectVoiceMode, savedVoices]);

  useEffect(() => {
    if (projectModelId && projectModelId !== selectedModel) {
      onSelectModel?.(projectModelId);
    }
  }, [onSelectModel, projectModelId, selectedModel]);

  const builtInVoiceItems: VoicePickerItem[] = availableSpeakers.map((voice) => ({
    id: voice.id,
    name: voice.name,
    categoryLabel: currentProjectModelInfo?.variant ?? "Built-in voice",
    description: voice.description,
    meta: [voice.language],
    previewMessage: "Use project render to audition this voice across the full script.",
    selected: projectVoiceMode === "built_in" && projectSpeaker === voice.id,
    onSelect: () => {
      setProjectVoiceMode("built_in");
      setProjectSpeaker(voice.id);
      setWorkspaceStatus(null);
    },
  }));

  const savedVoiceItems: VoicePickerItem[] = savedVoices.map((voice) => ({
    id: voice.id,
    name: voice.name,
    categoryLabel: sourceLabel(voice.source_route_kind),
    description: voice.reference_text_preview,
    meta: [
      `${voice.reference_text_chars} chars`,
      formatRelativeDate(voice.updated_at || voice.created_at),
    ],
    previewUrl: api.savedVoiceAudioUrl(voice.id),
    selected: projectVoiceMode === "saved" && projectSavedVoiceId === voice.id,
    onSelect: () => {
      setProjectVoiceMode("saved");
      setProjectSavedVoiceId(voice.id);
      setWorkspaceStatus(null);
    },
  }));

  const handleImportFile = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    try {
      const text = await file.text();
      setNewProjectText(text);
      setNewProjectFilename(file.name);
      if (!newProjectName.trim()) {
        setNewProjectName(file.name.replace(/\.[^.]+$/, ""));
      }
      setWorkspaceStatus({
        tone: "success",
        message: `Imported ${file.name}. Review the text, then create the project.`,
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to read the selected file.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      event.target.value = "";
    }
  };

  const resolveNewProjectDefaults = useCallback(() => {
    const currentVariant = selectedModelInfo?.variant ?? selectedModel ?? "";
    if (
      selectedModelInfo?.speech_capabilities?.supports_builtin_voices &&
      getSpeakerProfilesForVariant(currentVariant).length > 0
    ) {
      return {
        modelId: currentVariant,
        voiceMode: "built_in" as const,
        speaker: getSpeakerProfilesForVariant(currentVariant)[0]?.id ?? "Vivian",
        savedVoiceId: undefined,
      };
    }

    if (
      selectedModelInfo?.speech_capabilities?.supports_reference_voice &&
      savedVoices.length > 0
    ) {
      return {
        modelId: currentVariant,
        voiceMode: "saved" as const,
        speaker: undefined,
        savedVoiceId: savedVoices[0]?.id,
      };
    }

    const builtInModel = resolvePreferredRouteModel({
      models: builtInCompatibleModels,
      selectedModel,
      preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
    });
    if (builtInModel) {
      return {
        modelId: builtInModel,
        voiceMode: "built_in" as const,
        speaker: getSpeakerProfilesForVariant(builtInModel)[0]?.id ?? "Vivian",
        savedVoiceId: undefined,
      };
    }

    const savedModel = resolvePreferredRouteModel({
      models: savedVoiceCompatibleModels,
      selectedModel,
      preferredVariants: SAVED_VOICE_RENDERER_PREFERRED_MODELS,
    });
    if (savedModel && savedVoices.length > 0) {
      return {
        modelId: savedModel,
        voiceMode: "saved" as const,
        speaker: undefined,
        savedVoiceId: savedVoices[0]?.id,
      };
    }

    return null;
  }, [
    builtInCompatibleModels,
    savedVoiceCompatibleModels,
    savedVoices,
    selectedModel,
    selectedModelInfo,
  ]);

  const handleCreateProject = async () => {
    if (!newProjectText.trim()) {
      const message = "Paste or import the script before creating a project.";
      setWorkspaceError(message);
      onError(message);
      return;
    }

    const defaults = resolveNewProjectDefaults();
    if (!defaults) {
      onModelRequired();
      const message =
        "Load a built-in voice model or saved-voice renderer before creating a TTS project.";
      setWorkspaceError(message);
      onError(message);
      return;
    }

    try {
      setCreatingProject(true);
      setWorkspaceError(null);
      setWorkspaceStatus(null);

      const project = await api.createTtsProject({
        name: newProjectName.trim() || undefined,
        source_filename: newProjectFilename || undefined,
        source_text: newProjectText,
        model_id: defaults.modelId,
        voice_mode: defaults.voiceMode,
        speaker: defaults.speaker,
        saved_voice_id: defaults.savedVoiceId,
        speed: 1,
      });

      setSelectedProjectId(project.id);
      setSelectedProject(project);
      setNewProjectName("");
      setNewProjectText("");
      setNewProjectFilename("");
      setWorkspaceStatus({
        tone: "success",
        message: `Created project "${project.name}" with ${project.segments.length} segments.`,
      });
      await loadProjects();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to create the TTS project.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setCreatingProject(false);
    }
  };

  const persistProjectSettings = useCallback(async () => {
    if (!selectedProject) {
      return null;
    }
    if (!projectModelId) {
      onModelRequired();
      return null;
    }
    if (projectVoiceMode === "built_in" && !projectSpeaker) {
      const message = "Choose a built-in voice before saving project settings.";
      setWorkspaceError(message);
      onError(message);
      return null;
    }
    if (projectVoiceMode === "saved" && !projectSavedVoiceId) {
      const message = "Choose a saved voice before saving project settings.";
      setWorkspaceError(message);
      onError(message);
      return null;
    }
    if (!projectDirty) {
      return selectedProject;
    }

    try {
      setSavingProject(true);
      const project = await api.updateTtsProject(selectedProject.id, {
        name: projectName.trim(),
        model_id: projectModelId,
        voice_mode: projectVoiceMode,
        speaker: projectVoiceMode === "built_in" ? projectSpeaker : undefined,
        saved_voice_id:
          projectVoiceMode === "saved" ? projectSavedVoiceId : undefined,
        speed: projectSpeed,
      });
      setSelectedProject(project);
      setWorkspaceStatus({
        tone: "success",
        message: "Project settings saved.",
      });
      await loadProjects();
      return project;
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to save project settings.";
      setWorkspaceError(message);
      onError(message);
      return null;
    } finally {
      setSavingProject(false);
    }
  }, [
    loadProjects,
    onError,
    onModelRequired,
    projectDirty,
    projectModelId,
    projectName,
    projectSavedVoiceId,
    projectSpeaker,
    projectSpeed,
    projectVoiceMode,
    selectedProject,
  ]);

  const handleSaveSegment = async (segmentId: string) => {
    if (!selectedProject) {
      return;
    }
    const text = segmentDrafts[segmentId]?.trim() ?? "";
    if (!text) {
      const message = "Segment text cannot be empty.";
      setWorkspaceError(message);
      onError(message);
      return;
    }
    try {
      setSavingSegmentId(segmentId);
      const project = await api.updateTtsProjectSegment(selectedProject.id, segmentId, {
        text,
      });
      setSelectedProject(project);
      setWorkspaceStatus({
        tone: "success",
        message: "Segment text saved. Re-render to refresh the audio.",
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to save the segment.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setSavingSegmentId(null);
    }
  };

  const handleRenderSegment = async (segmentId: string) => {
    if (!selectedProject) {
      return;
    }
    const project = await persistProjectSettings();
    if (!project) {
      return;
    }
    try {
      setRenderingSegmentId(segmentId);
      const updated = await api.renderTtsProjectSegment(project.id, segmentId);
      setSelectedProject(updated);
      setWorkspaceStatus({
        tone: "success",
        message: "Segment rendered and attached to the project.",
      });
      await loadProjects();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to render the segment.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setRenderingSegmentId(null);
    }
  };

  const handleRenderAll = async () => {
    if (!selectedProject) {
      return;
    }
    const project = await persistProjectSettings();
    if (!project) {
      return;
    }
    try {
      setRenderingAll(true);
      let current = project;
      for (const segment of current.segments) {
        current = await api.renderTtsProjectSegment(current.id, segment.id);
        setSelectedProject(current);
      }
      setWorkspaceStatus({
        tone: "success",
        message: `Rendered ${current.segments.length} project segments.`,
      });
      await loadProjects();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed while rendering the project.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setRenderingAll(false);
    }
  };

  const handleExport = async () => {
    if (!selectedProject || isDownloading) {
      return;
    }
    beginDownload();
    try {
      await api.downloadAudioFile(
        api.ttsProjectAudioUrl(selectedProject.id, { download: true }),
        projectAudioFilename(selectedProject.name),
      );
      completeDownload();
    } catch (err) {
      failDownload(err);
    }
  };

  const handleDeleteProject = async () => {
    if (!selectedProject || deletingProject) {
      return;
    }
    if (typeof window !== "undefined") {
      const confirmed = window.confirm(
        `Delete the project "${selectedProject.name}"?`,
      );
      if (!confirmed) {
        return;
      }
    }
    try {
      setDeletingProject(true);
      await api.deleteTtsProject(selectedProject.id);
      setWorkspaceStatus({
        tone: "success",
        message: `Deleted project "${selectedProject.name}".`,
      });
      setSelectedProject(null);
      setSelectedProjectId(null);
      await loadProjects();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to delete the project.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setDeletingProject(false);
    }
  };

  return (
    <div className="grid gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
      <div className="space-y-4">
        <Card>
          <CardContent className="space-y-4 p-5">
            <div className="space-y-1">
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                New project
              </div>
              <h3 className="text-lg font-semibold text-foreground">
                Import and split a script
              </h3>
              <p className="text-sm text-muted-foreground">
                Create a reusable narration project with editable segments and merged export.
              </p>
            </div>

            <Input
              value={newProjectName}
              onChange={(event) => setNewProjectName(event.target.value)}
              placeholder="Optional project name"
            />
            <Textarea
              value={newProjectText}
              onChange={(event) => setNewProjectText(event.target.value)}
              rows={8}
              placeholder="Paste the script you want to split into renderable segments..."
            />

            <input
              ref={fileInputRef}
              type="file"
              accept=".txt,.md,text/plain"
              className="hidden"
              onChange={handleImportFile}
            />
            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="h-4 w-4" />
                Import text file
              </Button>
              <Button onClick={handleCreateProject} disabled={creatingProject}>
                {creatingProject ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <FilePlus2 className="h-4 w-4" />
                    Create project
                  </>
                )}
              </Button>
            </div>
            {newProjectFilename ? (
              <div className="rounded-xl border border-border/70 bg-muted/35 px-3 py-2 text-xs text-muted-foreground">
                Imported file: {newProjectFilename}
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-4 p-5">
            <div className="space-y-1">
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                Projects
              </div>
              <h3 className="text-lg font-semibold text-foreground">
                Saved scripts
              </h3>
            </div>

            <div className="space-y-2">
              {projectsLoading ? (
                <div className="rounded-xl border border-dashed border-border/70 bg-muted/25 px-3 py-6 text-center text-sm text-muted-foreground">
                  Loading projects...
                </div>
              ) : projects.length === 0 ? (
                <div className="rounded-xl border border-dashed border-border/70 bg-muted/25 px-3 py-6 text-center text-sm text-muted-foreground">
                  No TTS projects yet.
                </div>
              ) : (
                projects.map((project) => (
                  <button
                    key={project.id}
                    type="button"
                    onClick={() => setSelectedProjectId(project.id)}
                    className={cn(
                      "w-full rounded-2xl border px-3 py-3 text-left transition-colors",
                      project.id === selectedProjectId
                        ? "border-primary/50 bg-primary/5"
                        : "border-border/75 bg-card/70 hover:border-primary/30",
                    )}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="truncate text-sm font-semibold text-foreground">
                          {project.name}
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          {project.rendered_segment_count}/{project.segment_count} segments rendered
                        </div>
                      </div>
                      <div className="rounded-full border border-border/80 bg-muted/55 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                        {formatRelativeDate(project.updated_at)}
                      </div>
                    </div>
                  </button>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="space-y-4">
        {workspaceError ? (
          <div className="flex items-start gap-2 rounded-xl border border-destructive/45 bg-destructive/5 px-3 py-3 text-sm text-destructive">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <p>{workspaceError}</p>
          </div>
        ) : null}

        {workspaceStatus ? (
          <div
            className={cn(
              "flex items-center gap-2 rounded-xl border px-3 py-2 text-sm",
              workspaceStatus.tone === "success"
                ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-700"
                : "border-destructive/35 bg-destructive/5 text-destructive",
            )}
          >
            {workspaceStatus.tone === "success" ? (
              <CheckCircle2 className="h-4 w-4" />
            ) : (
              <AlertCircle className="h-4 w-4" />
            )}
            {workspaceStatus.message}
          </div>
        ) : null}

        {downloadState !== "idle" && downloadMessage ? (
          <div
            className={cn(
              "flex items-center gap-2 rounded-xl border px-3 py-2 text-sm",
              downloadState === "downloading" &&
                "border-amber-500/30 bg-amber-500/10 text-amber-700",
              downloadState === "success" &&
                "border-emerald-500/25 bg-emerald-500/10 text-emerald-700",
              downloadState === "error" &&
                "border-destructive/35 bg-destructive/5 text-destructive",
            )}
          >
            {downloadState === "downloading" ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : downloadState === "success" ? (
              <CheckCircle2 className="h-4 w-4" />
            ) : (
              <AlertCircle className="h-4 w-4" />
            )}
            {downloadMessage}
          </div>
        ) : null}

        {!selectedProject ? (
          <Card>
            <CardContent className="flex min-h-[420px] flex-col items-center justify-center gap-4 p-6 text-center">
              <div className="rounded-2xl border border-border/70 bg-muted/45 p-4">
                <FileAudio className="h-6 w-6 text-muted-foreground" />
              </div>
              <div className="space-y-1">
                <p className="text-base font-semibold text-foreground">
                  Select or create a TTS project
                </p>
                <p className="max-w-md text-sm text-muted-foreground">
                  Projects keep script segments, rendering progress, and merged export in one place.
                </p>
              </div>
            </CardContent>
          </Card>
        ) : (
          <>
            <Card>
              <CardContent className="space-y-5 p-5">
                <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                  <div className="space-y-1">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                      Project settings
                    </div>
                    <h2 className="text-lg font-semibold text-foreground">
                      {selectedProject.name}
                    </h2>
                    <p className="text-sm text-muted-foreground">
                      Apply a global voice and model, then render segments individually or all at once.
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      variant="outline"
                      onClick={() => void persistProjectSettings()}
                      disabled={!projectDirty || savingProject}
                    >
                      {savingProject ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Settings2 className="h-4 w-4" />
                      )}
                      Save settings
                    </Button>
                    <Button
                      variant="outline"
                      onClick={handleExport}
                      disabled={isDownloading}
                    >
                      {isDownloading ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Download className="h-4 w-4" />
                      )}
                      Export WAV
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => void handleRenderAll()}
                      disabled={renderingAll}
                    >
                      {renderingAll ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Waves className="h-4 w-4" />
                      )}
                      Render all
                    </Button>
                    <Button
                      variant="ghost"
                      onClick={() => void handleDeleteProject()}
                      disabled={deletingProject}
                    >
                      {deletingProject ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
                      Delete
                    </Button>
                  </div>
                </div>

                <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_220px]">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                        Project name
                      </label>
                      <Input
                        value={projectName}
                        onChange={(event) => setProjectName(event.target.value)}
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                        Render model
                      </label>
                      <select
                        value={projectModelId}
                        onChange={(event) => setProjectModelId(event.target.value)}
                        className="flex h-10 w-full rounded-lg border border-input/85 bg-background/70 px-3 text-sm shadow-sm transition-[border-color,box-shadow,background-color] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/35 focus-visible:border-ring/50"
                      >
                        {compatibleModelOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label} · {option.statusLabel}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-border/75 bg-muted/25 p-4">
                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                      Progress
                    </div>
                    <div className="mt-2 text-2xl font-semibold text-foreground">
                      {selectedProjectSummary?.rendered_segment_count ?? 0}/
                      {selectedProjectSummary?.segment_count ?? selectedProject.segments.length}
                    </div>
                    <div className="mt-1 text-sm text-muted-foreground">
                      rendered segments
                    </div>
                    {onOpenModelManager ? (
                      <Button
                        variant="outline"
                        size="sm"
                        className="mt-4"
                        onClick={onOpenModelManager}
                      >
                        <Settings2 className="h-4 w-4" />
                        Models
                      </Button>
                    ) : null}
                  </div>
                </div>

                <div className="space-y-4 rounded-2xl border border-border/75 bg-muted/25 p-4">
                  <Tabs
                    value={projectVoiceMode}
                    onValueChange={(value) =>
                      setProjectVoiceMode(value as TtsProjectVoiceMode)
                    }
                    className="space-y-4"
                  >
                    <TabsList>
                      <TabsTrigger value="built_in">Built-in</TabsTrigger>
                      <TabsTrigger value="saved">My voices</TabsTrigger>
                    </TabsList>
                  </Tabs>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium text-foreground">Speed</span>
                      <span className="text-muted-foreground">{projectSpeed.toFixed(2)}x</span>
                    </div>
                    <Slider
                      value={[projectSpeed]}
                      min={0.5}
                      max={1.5}
                      step={0.05}
                      onValueChange={([value]) => setProjectSpeed(value ?? 1)}
                    />
                  </div>

                  {projectVoiceMode === "saved" ? (
                    <>
                      {savedVoicesError ? (
                        <div className="rounded-xl border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs text-destructive">
                          {savedVoicesError}
                        </div>
                      ) : null}
                      <VoicePicker
                        items={savedVoiceItems}
                        emptyTitle={
                          savedVoicesLoading
                            ? "Loading saved voices"
                            : "No saved voices yet"
                        }
                        emptyDescription="Save a voice from cloning or design before using it in a TTS project."
                      />
                    </>
                  ) : (
                    <VoicePicker
                      items={builtInVoiceItems}
                      emptyTitle="No built-in voices available"
                      emptyDescription="Load a built-in voice model to assign a speaker for this project."
                    />
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="space-y-4 p-5">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                      Segments
                    </div>
                    <h3 className="text-lg font-semibold text-foreground">
                      Review and render script blocks
                    </h3>
                  </div>
                  {projectLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                  ) : null}
                </div>

                <div className="space-y-4">
                  {selectedProject.segments.map((segment) => {
                    const draft = segmentDrafts[segment.id] ?? segment.text;
                    const segmentDirty = draft !== segment.text;
                    const isSaving = savingSegmentId === segment.id;
                    const isRendering = renderingSegmentId === segment.id;

                    return (
                      <div
                        key={segment.id}
                        className="rounded-2xl border border-border/75 bg-card/70 p-4"
                      >
                        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                          <div className="space-y-1">
                            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                              Segment {segment.position + 1}
                            </div>
                            <div className="text-sm text-muted-foreground">
                              {segment.input_chars} chars
                              {segment.audio_duration_secs
                                ? ` · ${segment.audio_duration_secs.toFixed(1)}s audio`
                                : ""}
                            </div>
                          </div>

                          <div className="flex flex-wrap items-center gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => void handleSaveSegment(segment.id)}
                              disabled={!segmentDirty || isSaving}
                            >
                              {isSaving ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <PencilLine className="h-4 w-4" />
                              )}
                              Save text
                            </Button>
                            <Button
                              size="sm"
                              onClick={() => void handleRenderSegment(segment.id)}
                              disabled={isRendering}
                            >
                              {isRendering ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <Play className="h-4 w-4" />
                              )}
                              Render
                            </Button>
                          </div>
                        </div>

                        <Textarea
                          className="mt-4"
                          value={draft}
                          onChange={(event) =>
                            setSegmentDrafts((current) => ({
                              ...current,
                              [segment.id]: event.target.value,
                            }))
                          }
                        />

                        {segment.speech_record_id ? (
                          <div className="mt-4 space-y-2">
                            <audio
                              src={api.textToSpeechRecordAudioUrl(segment.speech_record_id)}
                              controls
                              preload="none"
                              className="h-10 w-full"
                            />
                            <div className="text-xs text-muted-foreground">
                              Linked generation: {segment.speech_record_id}
                            </div>
                          </div>
                        ) : (
                          <div className="mt-4 rounded-xl border border-dashed border-border/75 bg-muted/25 px-3 py-2 text-xs text-muted-foreground">
                            Render this segment to attach audio and include it in the merged export.
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}
