import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { Settings2 } from "lucide-react";

import { api, type ModelInfo, type SpeechHistoryRecord } from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import {
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";
import { TextToSpeechHistoryTable } from "@/features/text-to-speech/components/TextToSpeechHistoryTable";
import { NewTextToSpeechModal } from "@/features/text-to-speech/components/NewTextToSpeechModal";
import { TextToSpeechRecordDetail } from "@/features/text-to-speech/components/TextToSpeechRecordDetail";
import { useTextToSpeechHistory } from "@/features/text-to-speech/hooks/useTextToSpeechHistory";
import { useTextToSpeechRecord } from "@/features/text-to-speech/hooks/useTextToSpeechRecord";
import {
  normalizeSpeechProcessingStatus,
  resolveSpeechVoiceLabel,
} from "@/features/text-to-speech/support";

interface TextToSpeechPageProps {
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

export function TextToSpeechPage({
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
}: TextToSpeechPageProps) {
  const { recordId } = useParams<{ recordId: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const appliedQueryModelRef = useRef(false);
  const autoOpenCreateRef = useRef(false);
  const [isNewTextToSpeechModalOpen, setIsNewTextToSpeechModalOpen] =
    useState(false);
  const [streamingRecord, setStreamingRecord] =
    useState<SpeechHistoryRecord | null>(null);
  const [recordActionError, setRecordActionError] = useState<string | null>(
    null,
  );
  const [recordDeletePending, setRecordDeletePending] = useState(false);
  const [savedVoiceNameById, setSavedVoiceNameById] = useState<
    Record<string, string>
  >({});
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
  } = useRouteModelSelection({
    models,
    selectedModel,
    onSelect,
    modelFilter: (variant) => {
      const match = models.find((model) => model.variant === variant);
      const capabilities = match?.speech_capabilities;
      return Boolean(
        capabilities &&
          !variant.includes("Tokenizer") &&
          (capabilities.supports_builtin_voices ||
            capabilities.supports_reference_voice ||
            capabilities.supports_voice_description),
      );
    },
    resolveSelectedModel: (availableRouteModels, currentModel) =>
      resolvePreferredRouteModel({
        models: availableRouteModels,
        selectedModel: currentModel,
        preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
      }),
  });

  const {
    records,
    loading: historyLoading,
    error: historyError,
    refresh: refreshHistory,
  } = useTextToSpeechHistory();
  const {
    record,
    loading: recordLoading,
    error: recordError,
    refresh: refreshRecord,
  } = useTextToSpeechRecord(recordId);

  const refreshSavedVoiceNames = useCallback(async () => {
    try {
      const voices = await api.listSavedVoices();
      setSavedVoiceNameById(
        voices.reduce<Record<string, string>>((acc, voice) => {
          acc[voice.id] = voice.name;
          return acc;
        }, {}),
      );
    } catch {
      setSavedVoiceNameById({});
    }
  }, []);

  useEffect(() => {
    if (appliedQueryModelRef.current || routeModels.length === 0) {
      return;
    }

    const requestedModel = searchParams.get("model");
    if (
      requestedModel &&
      routeModels.some((model) => model.variant === requestedModel)
    ) {
      onSelect(requestedModel);
    }

    appliedQueryModelRef.current = true;
  }, [onSelect, routeModels, searchParams]);

  useEffect(() => {
    void refreshSavedVoiceNames();
  }, [refreshSavedVoiceNames]);

  useEffect(() => {
    if (!recordId) {
      setRecordActionError(null);
      return;
    }

    setRecordActionError(null);
    setStreamingRecord((current) =>
      current?.id === recordId ? current : null,
    );
  }, [recordId]);

  useEffect(() => {
    if (!record || !streamingRecord || record.id !== streamingRecord.id) {
      return;
    }

    const processingStatus = normalizeSpeechProcessingStatus(
      record.processing_status,
      record.processing_error,
    );
    if (processingStatus === "ready" || processingStatus === "failed") {
      setStreamingRecord(null);
    }
  }, [record, streamingRecord]);

  useEffect(() => {
    if (recordId || autoOpenCreateRef.current) {
      return;
    }

    if (searchParams.get("voiceId") || searchParams.get("speaker")) {
      setIsNewTextToSpeechModalOpen(true);
      autoOpenCreateRef.current = true;
    }
  }, [recordId, searchParams]);

  const visibleRecord = useMemo(() => {
    if (!recordId) {
      return null;
    }
    if (!streamingRecord || streamingRecord.id !== recordId) {
      return record;
    }
    if (!record) {
      return streamingRecord;
    }

    return {
      ...record,
      processing_status: streamingRecord.processing_status,
      processing_error:
        streamingRecord.processing_error ?? record.processing_error,
      model_id: streamingRecord.model_id || record.model_id,
      speaker: streamingRecord.speaker || record.speaker,
      language: streamingRecord.language || record.language,
      saved_voice_id: streamingRecord.saved_voice_id || record.saved_voice_id,
      speed: streamingRecord.speed ?? record.speed,
      input_text: streamingRecord.input_text || record.input_text,
      voice_description:
        streamingRecord.voice_description || record.voice_description,
      reference_text: streamingRecord.reference_text || record.reference_text,
      generation_time_ms:
        streamingRecord.generation_time_ms || record.generation_time_ms,
      audio_duration_secs:
        record.audio_duration_secs ?? streamingRecord.audio_duration_secs,
      rtf: record.rtf ?? streamingRecord.rtf,
      tokens_generated:
        record.tokens_generated ?? streamingRecord.tokens_generated,
      audio_filename: record.audio_filename || streamingRecord.audio_filename,
      audio_mime_type: record.audio_mime_type || streamingRecord.audio_mime_type,
    };
  }, [record, recordId, streamingRecord]);

  const detailDescription = useMemo(() => {
    if (!visibleRecord) {
      return "Inspect status and generation details for this text-to-speech record.";
    }

    const processingStatus = normalizeSpeechProcessingStatus(
      visibleRecord.processing_status,
      visibleRecord.processing_error,
    );

    switch (processingStatus) {
      case "pending":
        return "This generation is queued.";
      case "processing":
        return "This generation is actively rendering audio.";
      case "failed":
        return "This generation failed before audio became available.";
      case "ready":
      default:
        return "Inspect status and generation details for this text-to-speech record.";
    }
  }, [visibleRecord]);

  const visibleVoiceLabel = useMemo(
    () =>
      visibleRecord
        ? resolveSpeechVoiceLabel({
            savedVoiceId: visibleRecord.saved_voice_id,
            speaker: visibleRecord.speaker,
            modelId: visibleRecord.model_id,
            savedVoiceNameById,
          })
        : null,
    [savedVoiceNameById, visibleRecord],
  );

  const detailAudioUrl = useMemo(
    () =>
      recordId ? api.textToSpeechRecordAudioUrl(recordId) : null,
    [recordId],
  );

  const handleDetailDelete = async () => {
    if (!recordId || recordDeletePending) {
      return;
    }

    setRecordDeletePending(true);
    setRecordActionError(null);
    try {
      await api.deleteTextToSpeechRecord(recordId);
      setStreamingRecord(null);
      await refreshHistory();
      navigate("/text-to-speech", { replace: true });
    } catch (err) {
      setRecordActionError(
        err instanceof Error ? err.message : "Failed to delete generation.",
      );
    } finally {
      setRecordDeletePending(false);
    }
  };

  return (
    <PageShell>
      {recordId ? (
        <>
          <PageHeader
            title="Text-to-Speech Record"
            description={detailDescription}
            actions={
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 gap-2"
                onClick={openModelManager}
              >
                <Settings2 className="h-4 w-4" />
                Models
              </Button>
            }
          />

          <TextToSpeechRecordDetail
            record={visibleRecord}
            audioUrl={detailAudioUrl}
            voiceLabel={visibleVoiceLabel}
            loading={recordLoading}
            error={recordError}
            deleteError={recordActionError}
            deletePending={recordDeletePending}
            onBack={() => navigate("/text-to-speech")}
            onDelete={() => void handleDetailDelete()}
          />
        </>
      ) : (
        <>
          <PageHeader
            title="Text to Speech"
            description="Monitor queued, processing, and completed speech generations in one operational history table."
            actions={
              <>
                <Button
                  type="button"
                  size="sm"
                  className="h-9 gap-2"
                  onClick={() => setIsNewTextToSpeechModalOpen(true)}
                >
                  New generation
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-9 gap-2"
                  onClick={openModelManager}
                >
                  <Settings2 className="h-4 w-4" />
                  Models
                </Button>
              </>
            }
          />

          <TextToSpeechHistoryTable
            records={records}
            savedVoiceNameById={savedVoiceNameById}
            loading={historyLoading}
            error={historyError}
            onRefresh={() => void refreshHistory()}
            onOpenRecord={(nextRecordId) => {
              navigate(`/text-to-speech/${nextRecordId}`);
            }}
          />
        </>
      )}

      <NewTextToSpeechModal
        isOpen={isNewTextToSpeechModalOpen}
        onClose={() => setIsNewTextToSpeechModalOpen(false)}
        selectedModel={resolvedSelectedModel}
        selectedModelInfo={selectedModelInfo}
        selectedModelReady={selectedModelReady}
        initialSavedVoiceId={searchParams.get("voiceId")}
        initialSpeaker={searchParams.get("speaker")}
        onModelRequired={() => {
          requestModel();
          onError(
            "Select and load a compatible TTS model before creating a text-to-speech job.",
          );
        }}
        onCreated={async (createdRecord) => {
          setRecordActionError(null);
          setStreamingRecord(createdRecord);
          navigate(`/text-to-speech/${createdRecord.id}`);
          void refreshHistory();
        }}
        onStreamingStart={() => {
          setStreamingRecord((current) =>
            current
              ? {
                  ...current,
                  processing_status: "processing",
                }
              : current,
          );
        }}
        onStreamingFinal={(finalRecord) => {
          setStreamingRecord(finalRecord);
          void refreshHistory();
        }}
        onStreamingError={() => {
          void refreshRecord();
        }}
      />

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title="Text-to-Speech Models"
        description="Manage built-in voice, saved-voice, and voice-direction TTS models for this route."
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
        onUseModel={onSelect}
        emptyMessage="Load a compatible TTS model with built-in voices, saved voices, or voice-direction prompts to generate speech."
      />
    </PageShell>
  );
}
