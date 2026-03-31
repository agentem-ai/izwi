import { type DiarizationRecord, type DiarizationRecordSummary } from "@/api";
import {
  formattedTranscriptFromRecord,
  previewTranscript,
  transcriptEntriesFromRecord,
} from "@/utils/diarizationTranscript";
import {
  buildDiarizationSummaryPreview,
  normalizeDiarizationSummaryStatus,
} from "@/utils/diarizationSummary";

export function summarizeDiarizationRecord(
  record: DiarizationRecord,
): DiarizationRecordSummary {
  const entries = transcriptEntriesFromRecord(record);
  const preview = previewTranscript(
    entries,
    record.transcript ?? "",
    record.raw_transcript ?? "",
  );
  const formatted = formattedTranscriptFromRecord(record);

  return {
    id: record.id,
    created_at: record.created_at,
    model_id: record.model_id,
    processing_status: record.processing_status,
    processing_error: record.processing_error,
    speaker_count: record.speaker_count ?? 0,
    corrected_speaker_count:
      record.corrected_speaker_count ?? record.speaker_count ?? 0,
    speaker_name_override_count: Object.keys(
      record.speaker_name_overrides ?? {},
    ).length,
    duration_secs: record.duration_secs,
    processing_time_ms: record.processing_time_ms,
    rtf: record.rtf,
    audio_mime_type: record.audio_mime_type,
    audio_filename: record.audio_filename,
    transcript_preview: preview || "No transcript",
    transcript_chars: Array.from(formatted).length,
    summary_status: normalizeDiarizationSummaryStatus(
      record.summary_status,
      record.summary_text,
      record.summary_error,
    ),
    summary_preview: buildDiarizationSummaryPreview(record.summary_text),
    summary_chars: Array.from(record.summary_text ?? "").length,
  };
}
