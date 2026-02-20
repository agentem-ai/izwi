//! Persistent diarization history storage backed by SQLite.

use anyhow::{anyhow, Context};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::task;

const DEFAULT_LIST_LIMIT: usize = 200;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationSegmentRecord {
    pub speaker: String,
    pub start: f32,
    pub end: f32,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationWordRecord {
    pub word: String,
    pub speaker: String,
    pub start: f32,
    pub end: f32,
    pub speaker_confidence: Option<f32>,
    pub overlaps_segment: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationUtteranceRecord {
    pub speaker: String,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub word_start: usize,
    pub word_end: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiarizationRecordSummary {
    pub id: String,
    pub created_at: u64,
    pub model_id: Option<String>,
    pub speaker_count: usize,
    pub duration_secs: Option<f64>,
    pub processing_time_ms: f64,
    pub rtf: Option<f64>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub transcript_preview: String,
    pub transcript_chars: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiarizationRecord {
    pub id: String,
    pub created_at: u64,
    pub model_id: Option<String>,
    pub asr_model_id: Option<String>,
    pub aligner_model_id: Option<String>,
    pub llm_model_id: Option<String>,
    pub min_speakers: Option<usize>,
    pub max_speakers: Option<usize>,
    pub min_speech_duration_ms: Option<f64>,
    pub min_silence_duration_ms: Option<f64>,
    pub enable_llm_refinement: bool,
    pub processing_time_ms: f64,
    pub duration_secs: Option<f64>,
    pub rtf: Option<f64>,
    pub speaker_count: usize,
    pub alignment_coverage: Option<f64>,
    pub unattributed_words: usize,
    pub llm_refined: bool,
    pub asr_text: String,
    pub raw_transcript: String,
    pub transcript: String,
    pub segments: Vec<DiarizationSegmentRecord>,
    pub words: Vec<DiarizationWordRecord>,
    pub utterances: Vec<DiarizationUtteranceRecord>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StoredDiarizationAudio {
    pub audio_bytes: Vec<u8>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewDiarizationRecord {
    pub model_id: Option<String>,
    pub asr_model_id: Option<String>,
    pub aligner_model_id: Option<String>,
    pub llm_model_id: Option<String>,
    pub min_speakers: Option<usize>,
    pub max_speakers: Option<usize>,
    pub min_speech_duration_ms: Option<f64>,
    pub min_silence_duration_ms: Option<f64>,
    pub enable_llm_refinement: bool,
    pub processing_time_ms: f64,
    pub duration_secs: Option<f64>,
    pub rtf: Option<f64>,
    pub speaker_count: usize,
    pub alignment_coverage: Option<f64>,
    pub unattributed_words: usize,
    pub llm_refined: bool,
    pub asr_text: String,
    pub raw_transcript: String,
    pub transcript: String,
    pub segments: Vec<DiarizationSegmentRecord>,
    pub words: Vec<DiarizationWordRecord>,
    pub utterances: Vec<DiarizationUtteranceRecord>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub audio_bytes: Vec<u8>,
}

#[derive(Clone)]
pub struct DiarizationStore {
    db_path: PathBuf,
}

impl DiarizationStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = resolve_db_path();

        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create diarization database directory: {}",
                    parent.display()
                )
            })?;
        }

        let conn = open_connection(&db_path).with_context(|| {
            format!("Failed to open diarization database: {}", db_path.display())
        })?;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS diarization_records (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                model_id TEXT NULL,
                asr_model_id TEXT NULL,
                aligner_model_id TEXT NULL,
                llm_model_id TEXT NULL,
                min_speakers INTEGER NULL,
                max_speakers INTEGER NULL,
                min_speech_duration_ms REAL NULL,
                min_silence_duration_ms REAL NULL,
                enable_llm_refinement INTEGER NOT NULL,
                processing_time_ms REAL NOT NULL,
                duration_secs REAL NULL,
                rtf REAL NULL,
                speaker_count INTEGER NOT NULL,
                alignment_coverage REAL NULL,
                unattributed_words INTEGER NOT NULL,
                llm_refined INTEGER NOT NULL,
                asr_text TEXT NOT NULL,
                raw_transcript TEXT NOT NULL,
                transcript TEXT NOT NULL,
                segments_json TEXT NOT NULL,
                words_json TEXT NOT NULL,
                utterances_json TEXT NOT NULL,
                audio_mime_type TEXT NOT NULL,
                audio_filename TEXT NULL,
                audio_bytes BLOB NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_diarization_records_created_at
                ON diarization_records(created_at DESC);
            "#,
        )
        .context("Failed to initialize diarization database schema")?;

        Ok(Self { db_path })
    }

    pub async fn list_records(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<DiarizationRecordSummary>> {
        self.run_blocking(move |db_path| {
            let conn = open_connection(&db_path)?;
            let list_limit = i64::try_from(limit.clamp(1, 500).max(1)).unwrap_or(200);

            let mut stmt = conn.prepare(
                r#"
                SELECT
                    id,
                    created_at,
                    model_id,
                    speaker_count,
                    duration_secs,
                    processing_time_ms,
                    rtf,
                    audio_mime_type,
                    audio_filename,
                    transcript
                FROM diarization_records
                ORDER BY created_at DESC, id DESC
                LIMIT ?1
                "#,
            )?;

            let rows = stmt.query_map(params![list_limit], |row| {
                let transcript: String = row.get(9)?;
                Ok(DiarizationRecordSummary {
                    id: row.get(0)?,
                    created_at: i64_to_u64(row.get(1)?),
                    model_id: row.get(2)?,
                    speaker_count: row
                        .get::<_, Option<i64>>(3)?
                        .and_then(i64_to_usize)
                        .unwrap_or(0),
                    duration_secs: row.get(4)?,
                    processing_time_ms: row.get(5)?,
                    rtf: row.get(6)?,
                    audio_mime_type: row.get(7)?,
                    audio_filename: row.get(8)?,
                    transcript_preview: transcript_preview(transcript.as_str()),
                    transcript_chars: transcript.chars().count(),
                })
            })?;

            let mut records = Vec::new();
            for row in rows {
                records.push(row?);
            }
            Ok(records)
        })
        .await
    }

    pub async fn get_record(&self, record_id: String) -> anyhow::Result<Option<DiarizationRecord>> {
        self.run_blocking(move |db_path| {
            let conn = open_connection(&db_path)?;
            let record = fetch_record_without_audio(&conn, &record_id)?;
            Ok(record)
        })
        .await
    }

    pub async fn get_audio(
        &self,
        record_id: String,
    ) -> anyhow::Result<Option<StoredDiarizationAudio>> {
        self.run_blocking(move |db_path| {
            let conn = open_connection(&db_path)?;
            let audio = conn
                .query_row(
                    r#"
                    SELECT audio_bytes, audio_mime_type, audio_filename
                    FROM diarization_records
                    WHERE id = ?1
                    "#,
                    params![record_id],
                    |row| {
                        Ok(StoredDiarizationAudio {
                            audio_bytes: row.get(0)?,
                            audio_mime_type: row.get(1)?,
                            audio_filename: row.get(2)?,
                        })
                    },
                )
                .optional()?;
            Ok(audio)
        })
        .await
    }

    pub async fn create_record(
        &self,
        record: NewDiarizationRecord,
    ) -> anyhow::Result<DiarizationRecord> {
        self.run_blocking(move |db_path| {
            let conn = open_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let record_id = format!("dir_{}", uuid::Uuid::new_v4().simple());

            let model_id = sanitize_optional_text(record.model_id.as_deref(), 160);
            let asr_model_id = sanitize_optional_text(record.asr_model_id.as_deref(), 160);
            let aligner_model_id = sanitize_optional_text(record.aligner_model_id.as_deref(), 160);
            let llm_model_id = sanitize_optional_text(record.llm_model_id.as_deref(), 160);
            let min_speakers = record
                .min_speakers
                .and_then(|value| i64::try_from(value).ok());
            let max_speakers = record
                .max_speakers
                .and_then(|value| i64::try_from(value).ok());
            let min_speech_duration_ms = record
                .min_speech_duration_ms
                .filter(|value| value.is_finite() && *value >= 0.0);
            let min_silence_duration_ms = record
                .min_silence_duration_ms
                .filter(|value| value.is_finite() && *value >= 0.0);
            let processing_time_ms = if record.processing_time_ms.is_finite() {
                record.processing_time_ms.max(0.0)
            } else {
                0.0
            };
            let duration_secs = record
                .duration_secs
                .filter(|value| value.is_finite() && *value >= 0.0);
            let rtf = record
                .rtf
                .filter(|value| value.is_finite() && *value >= 0.0);
            let speaker_count = i64::try_from(record.speaker_count).unwrap_or(0);
            let alignment_coverage = record
                .alignment_coverage
                .filter(|value| value.is_finite() && *value >= 0.0);
            let unattributed_words = i64::try_from(record.unattributed_words).unwrap_or(0);
            let asr_text = sanitize_required_text(record.asr_text.as_str(), 40_000);
            let raw_transcript = sanitize_required_text(record.raw_transcript.as_str(), 100_000);
            let transcript = sanitize_required_text(record.transcript.as_str(), 100_000);
            let audio_mime_type = sanitize_audio_mime_type(record.audio_mime_type.as_str());
            let audio_filename = sanitize_optional_text(record.audio_filename.as_deref(), 260);

            if record.audio_bytes.is_empty() {
                return Err(anyhow!("Audio payload cannot be empty"));
            }

            let segments_json =
                serde_json::to_string(&record.segments).context("Failed serializing segments")?;
            let words_json =
                serde_json::to_string(&record.words).context("Failed serializing words")?;
            let utterances_json = serde_json::to_string(&record.utterances)
                .context("Failed serializing utterances")?;

            conn.execute(
                r#"
                INSERT INTO diarization_records (
                    id,
                    created_at,
                    model_id,
                    asr_model_id,
                    aligner_model_id,
                    llm_model_id,
                    min_speakers,
                    max_speakers,
                    min_speech_duration_ms,
                    min_silence_duration_ms,
                    enable_llm_refinement,
                    processing_time_ms,
                    duration_secs,
                    rtf,
                    speaker_count,
                    alignment_coverage,
                    unattributed_words,
                    llm_refined,
                    asr_text,
                    raw_transcript,
                    transcript,
                    segments_json,
                    words_json,
                    utterances_json,
                    audio_mime_type,
                    audio_filename,
                    audio_bytes
                )
                VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                    ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27
                )
                "#,
                params![
                    &record_id,
                    now,
                    model_id,
                    asr_model_id,
                    aligner_model_id,
                    llm_model_id,
                    min_speakers,
                    max_speakers,
                    min_speech_duration_ms,
                    min_silence_duration_ms,
                    if record.enable_llm_refinement {
                        1_i64
                    } else {
                        0_i64
                    },
                    processing_time_ms,
                    duration_secs,
                    rtf,
                    speaker_count,
                    alignment_coverage,
                    unattributed_words,
                    if record.llm_refined { 1_i64 } else { 0_i64 },
                    asr_text,
                    raw_transcript,
                    transcript,
                    segments_json,
                    words_json,
                    utterances_json,
                    audio_mime_type,
                    audio_filename,
                    record.audio_bytes
                ],
            )?;

            let created = fetch_record_without_audio(&conn, &record_id)?
                .ok_or_else(|| anyhow!("Failed to fetch created diarization record"))?;
            Ok(created)
        })
        .await
    }

    pub async fn delete_record(&self, record_id: String) -> anyhow::Result<bool> {
        self.run_blocking(move |db_path| {
            let conn = open_connection(&db_path)?;
            let changed = conn.execute(
                "DELETE FROM diarization_records WHERE id = ?1",
                params![record_id],
            )?;
            Ok(changed > 0)
        })
        .await
    }

    async fn run_blocking<F, T>(&self, task_fn: F) -> anyhow::Result<T>
    where
        F: FnOnce(PathBuf) -> anyhow::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let db_path = self.db_path.clone();
        task::spawn_blocking(move || task_fn(db_path))
            .await
            .map_err(|err| anyhow!("Diarization storage worker failed: {err}"))?
    }
}

fn fetch_record_without_audio(
    conn: &Connection,
    record_id: &str,
) -> anyhow::Result<Option<DiarizationRecord>> {
    let record = conn
        .query_row(
            r#"
            SELECT
                id,
                created_at,
                model_id,
                asr_model_id,
                aligner_model_id,
                llm_model_id,
                min_speakers,
                max_speakers,
                min_speech_duration_ms,
                min_silence_duration_ms,
                enable_llm_refinement,
                processing_time_ms,
                duration_secs,
                rtf,
                speaker_count,
                alignment_coverage,
                unattributed_words,
                llm_refined,
                asr_text,
                raw_transcript,
                transcript,
                segments_json,
                words_json,
                utterances_json,
                audio_mime_type,
                audio_filename
            FROM diarization_records
            WHERE id = ?1
            "#,
            params![record_id],
            map_diarization_record,
        )
        .optional()?;
    Ok(record)
}

fn map_diarization_record(row: &Row<'_>) -> rusqlite::Result<DiarizationRecord> {
    let min_speakers = row.get::<_, Option<i64>>(6)?.and_then(i64_to_usize);
    let max_speakers = row.get::<_, Option<i64>>(7)?.and_then(i64_to_usize);
    let speaker_count = row
        .get::<_, Option<i64>>(14)?
        .and_then(i64_to_usize)
        .unwrap_or(0);
    let unattributed_words = row
        .get::<_, Option<i64>>(16)?
        .and_then(i64_to_usize)
        .unwrap_or(0);
    let segments_raw: String = row.get(21)?;
    let words_raw: String = row.get(22)?;
    let utterances_raw: String = row.get(23)?;
    let segments: Vec<DiarizationSegmentRecord> = parse_json_vec(Some(segments_raw));
    let words: Vec<DiarizationWordRecord> = parse_json_vec(Some(words_raw));
    let utterances: Vec<DiarizationUtteranceRecord> = parse_json_vec(Some(utterances_raw));

    Ok(DiarizationRecord {
        id: row.get(0)?,
        created_at: i64_to_u64(row.get(1)?),
        model_id: row.get(2)?,
        asr_model_id: row.get(3)?,
        aligner_model_id: row.get(4)?,
        llm_model_id: row.get(5)?,
        min_speakers,
        max_speakers,
        min_speech_duration_ms: row.get(8)?,
        min_silence_duration_ms: row.get(9)?,
        enable_llm_refinement: row.get::<_, i64>(10)? > 0,
        processing_time_ms: row.get(11)?,
        duration_secs: row.get(12)?,
        rtf: row.get(13)?,
        speaker_count,
        alignment_coverage: row.get(15)?,
        unattributed_words,
        llm_refined: row.get::<_, i64>(17)? > 0,
        asr_text: row.get(18)?,
        raw_transcript: row.get(19)?,
        transcript: row.get(20)?,
        segments,
        words,
        utterances,
        audio_mime_type: row.get(24)?,
        audio_filename: row.get(25)?,
    })
}

fn parse_json_vec<T>(raw: Option<String>) -> Vec<T>
where
    T: for<'de> Deserialize<'de>,
{
    raw.and_then(|value| serde_json::from_str::<Vec<T>>(value.as_str()).ok())
        .unwrap_or_default()
}

fn resolve_db_path() -> PathBuf {
    if let Ok(raw_path) = std::env::var("IZWI_CHAT_DB_PATH") {
        let trimmed = raw_path.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }

    if let Some(mut dir) = dirs::data_local_dir() {
        dir.push("izwi");
        dir.push("chat.sqlite3");
        return dir;
    }

    PathBuf::from("data/chat.sqlite3")
}

fn open_connection(path: &Path) -> anyhow::Result<Connection> {
    let conn = Connection::open(path)
        .with_context(|| format!("Unable to open SQLite database at {}", path.display()))?;
    conn.busy_timeout(Duration::from_secs(3))
        .context("Failed to configure SQLite busy timeout")?;
    conn.pragma_update(None, "journal_mode", "WAL")
        .context("Failed to enable SQLite WAL journal mode")?;
    conn.pragma_update(None, "foreign_keys", "ON")
        .context("Failed to enable SQLite foreign key constraints")?;
    Ok(conn)
}

fn transcript_preview(content: &str) -> String {
    let normalized = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return "No transcript".to_string();
    }
    truncate_string(&normalized, 180)
}

fn sanitize_required_text(raw: &str, max_chars: usize) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        " ".to_string()
    } else {
        truncate_string(trimmed, max_chars)
    }
}

fn sanitize_optional_text(raw: Option<&str>, max_chars: usize) -> Option<String> {
    let normalized = raw
        .unwrap_or("")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if normalized.is_empty() {
        None
    } else {
        Some(truncate_string(&normalized, max_chars))
    }
}

fn sanitize_audio_mime_type(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        "audio/wav".to_string()
    } else {
        truncate_string(trimmed, 80)
    }
}

fn truncate_string(input: &str, max_chars: usize) -> String {
    let mut result = String::new();
    for (idx, ch) in input.chars().enumerate() {
        if idx >= max_chars {
            break;
        }
        result.push(ch);
    }
    if input.chars().count() > max_chars {
        result.push_str("...");
    }
    result
}

fn now_unix_millis_i64() -> i64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_millis() as i64
}

fn i64_to_u64(value: i64) -> u64 {
    if value.is_negative() {
        0
    } else {
        value as u64
    }
}

fn i64_to_usize(value: i64) -> Option<usize> {
    if value.is_negative() {
        None
    } else {
        usize::try_from(value).ok()
    }
}

#[allow(dead_code)]
pub const fn default_list_limit() -> usize {
    DEFAULT_LIST_LIMIT
}
