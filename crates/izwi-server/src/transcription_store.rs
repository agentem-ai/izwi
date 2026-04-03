//! Persistent transcription history storage backed by SQLite.

use anyhow::{anyhow, Context};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::task;

use crate::{
    ids::new_uuid,
    storage_layout::{self, MediaGroup},
};

const DEFAULT_LIST_LIMIT: usize = 200;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionSummaryStatus {
    NotRequested,
    Pending,
    Ready,
    Failed,
}

impl Default for TranscriptionSummaryStatus {
    fn default() -> Self {
        Self::NotRequested
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionProcessingStatus {
    Pending,
    Processing,
    Ready,
    Failed,
}

impl Default for TranscriptionProcessingStatus {
    fn default() -> Self {
        Self::Pending
    }
}

impl TranscriptionProcessingStatus {
    fn as_db_value(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Processing => "processing",
            Self::Ready => "ready",
            Self::Failed => "failed",
        }
    }
}

impl TranscriptionSummaryStatus {
    fn as_db_value(self) -> &'static str {
        match self {
            Self::NotRequested => "not_requested",
            Self::Pending => "pending",
            Self::Ready => "ready",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionWordRecord {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegmentRecord {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub word_start: usize,
    pub word_end: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionRecordSummary {
    pub id: String,
    pub created_at: u64,
    pub model_id: Option<String>,
    pub language: Option<String>,
    pub processing_status: TranscriptionProcessingStatus,
    pub processing_error: Option<String>,
    pub duration_secs: Option<f64>,
    pub processing_time_ms: f64,
    pub rtf: Option<f64>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub transcription_preview: String,
    pub transcription_chars: usize,
    pub summary_status: TranscriptionSummaryStatus,
    pub summary_preview: Option<String>,
    pub summary_chars: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionRecord {
    pub id: String,
    pub created_at: u64,
    pub model_id: Option<String>,
    pub aligner_model_id: Option<String>,
    pub language: Option<String>,
    pub processing_status: TranscriptionProcessingStatus,
    pub processing_error: Option<String>,
    pub duration_secs: Option<f64>,
    pub processing_time_ms: f64,
    pub rtf: Option<f64>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub transcription: String,
    pub segments: Vec<TranscriptionSegmentRecord>,
    pub words: Vec<TranscriptionWordRecord>,
    pub summary_status: TranscriptionSummaryStatus,
    pub summary_model_id: Option<String>,
    pub summary_text: Option<String>,
    pub summary_error: Option<String>,
    pub summary_updated_at: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct StoredTranscriptionAudio {
    pub audio_bytes: Vec<u8>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewTranscriptionRecord {
    pub model_id: Option<String>,
    pub aligner_model_id: Option<String>,
    pub language: Option<String>,
    pub processing_status: TranscriptionProcessingStatus,
    pub processing_error: Option<String>,
    pub duration_secs: Option<f64>,
    pub processing_time_ms: f64,
    pub rtf: Option<f64>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub audio_bytes: Vec<u8>,
    pub transcription: String,
    pub segments: Vec<TranscriptionSegmentRecord>,
    pub words: Vec<TranscriptionWordRecord>,
    pub summary_status: TranscriptionSummaryStatus,
    pub summary_model_id: Option<String>,
    pub summary_text: Option<String>,
    pub summary_error: Option<String>,
    pub summary_updated_at: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct UpdateTranscriptionSummary {
    pub status: TranscriptionSummaryStatus,
    pub model_id: Option<String>,
    pub text: Option<String>,
    pub error: Option<String>,
    pub updated_at: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct CompleteTranscriptionRecord {
    pub model_id: Option<String>,
    pub aligner_model_id: Option<String>,
    pub language: Option<String>,
    pub duration_secs: Option<f64>,
    pub processing_time_ms: f64,
    pub rtf: Option<f64>,
    pub transcription: String,
    pub segments: Vec<TranscriptionSegmentRecord>,
    pub words: Vec<TranscriptionWordRecord>,
    pub summary_status: TranscriptionSummaryStatus,
    pub summary_model_id: Option<String>,
    pub summary_text: Option<String>,
    pub summary_error: Option<String>,
    pub summary_updated_at: Option<u64>,
}

#[derive(Clone)]
pub struct TranscriptionStore {
    db_path: PathBuf,
    media_root: PathBuf,
}

impl TranscriptionStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();

        Self::initialize_at(db_path, media_root)
    }

    fn initialize_at(db_path: PathBuf, media_root: PathBuf) -> anyhow::Result<Self> {
        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare transcription storage layout")?;

        let conn = storage_layout::open_sqlite_connection(&db_path).with_context(|| {
            format!(
                "Failed to open transcription database: {}",
                db_path.display()
            )
        })?;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS transcription_records (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                model_id TEXT NULL,
                aligner_model_id TEXT NULL,
                language TEXT NULL,
                processing_status TEXT NOT NULL DEFAULT 'ready',
                processing_error TEXT NULL,
                duration_secs REAL NULL,
                processing_time_ms REAL NOT NULL,
                rtf REAL NULL,
                audio_mime_type TEXT NOT NULL,
                audio_filename TEXT NULL,
                audio_storage_path TEXT NOT NULL,
                transcription TEXT NOT NULL,
                segments_json TEXT NOT NULL DEFAULT '[]',
                words_json TEXT NOT NULL DEFAULT '[]',
                summary_status TEXT NOT NULL DEFAULT 'not_requested',
                summary_model_id TEXT NULL,
                summary_text TEXT NULL,
                summary_error TEXT NULL,
                summary_updated_at INTEGER NULL
            );

            CREATE INDEX IF NOT EXISTS idx_transcription_records_created_at
                ON transcription_records(created_at DESC);
            "#,
        )
        .context("Failed to initialize transcription database schema")?;
        ensure_transcription_records_aligner_model_id_column(&conn)?;
        ensure_transcription_records_processing_status_column(&conn)?;
        ensure_transcription_records_processing_error_column(&conn)?;
        ensure_transcription_records_segments_json_column(&conn)?;
        ensure_transcription_records_words_json_column(&conn)?;
        ensure_transcription_records_summary_status_column(&conn)?;
        ensure_transcription_records_summary_model_id_column(&conn)?;
        ensure_transcription_records_summary_text_column(&conn)?;
        ensure_transcription_records_summary_error_column(&conn)?;
        ensure_transcription_records_summary_updated_at_column(&conn)?;

        Ok(Self {
            db_path,
            media_root,
        })
    }

    pub async fn list_records(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<TranscriptionRecordSummary>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let list_limit = i64::try_from(limit.clamp(1, 500).max(1)).unwrap_or(200);

            let mut stmt = conn.prepare(
                r#"
                SELECT
                    id,
                    created_at,
                    model_id,
                    language,
                    processing_status,
                    processing_error,
                    duration_secs,
                    processing_time_ms,
                    rtf,
                    audio_mime_type,
                    audio_filename,
                    transcription,
                    summary_status,
                    summary_text
                FROM transcription_records
                ORDER BY created_at DESC, id DESC
                LIMIT ?1
                "#,
            )?;

            let rows = stmt.query_map(params![list_limit], |row| {
                let processing_status = parse_processing_status(row.get::<_, Option<String>>(4)?);
                let processing_error: Option<String> = row.get(5)?;
                let transcription: String = row.get(11)?;
                let summary_status = parse_summary_status(row.get::<_, Option<String>>(12)?);
                let summary_text: Option<String> = row.get(13)?;
                Ok(TranscriptionRecordSummary {
                    id: row.get(0)?,
                    created_at: i64_to_u64(row.get(1)?),
                    model_id: row.get(2)?,
                    language: row.get(3)?,
                    processing_status,
                    processing_error: processing_error.clone(),
                    duration_secs: row.get(6)?,
                    processing_time_ms: row.get(7)?,
                    rtf: row.get(8)?,
                    audio_mime_type: row.get(9)?,
                    audio_filename: row.get(10)?,
                    transcription_preview: transcription_preview(
                        processing_status,
                        processing_error.as_deref(),
                        &transcription,
                    ),
                    transcription_chars: transcription.chars().count(),
                    summary_status,
                    summary_preview: summary_preview(summary_text.as_deref()),
                    summary_chars: summary_text
                        .as_ref()
                        .map(|text| text.chars().count())
                        .unwrap_or(0),
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

    pub async fn get_record(
        &self,
        record_id: String,
    ) -> anyhow::Result<Option<TranscriptionRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let record = fetch_record_without_audio(&conn, &record_id)?;
            Ok(record)
        })
        .await
    }

    pub async fn get_audio(
        &self,
        record_id: String,
    ) -> anyhow::Result<Option<StoredTranscriptionAudio>> {
        let media_root = self.media_root.clone();
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let audio = conn
                .query_row(
                    r#"
                    SELECT audio_storage_path, audio_mime_type, audio_filename
                    FROM transcription_records
                    WHERE id = ?1
                    "#,
                    params![record_id],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, Option<String>>(2)?,
                        ))
                    },
                )
                .optional()?;

            let Some((audio_storage_path, audio_mime_type, audio_filename)) = audio else {
                return Ok(None);
            };

            let audio_bytes =
                storage_layout::read_media_file(&media_root, audio_storage_path.as_str())?;

            Ok(Some(StoredTranscriptionAudio {
                audio_bytes,
                audio_mime_type,
                audio_filename,
            }))
        })
        .await
    }

    pub async fn create_record(
        &self,
        record: NewTranscriptionRecord,
    ) -> anyhow::Result<TranscriptionRecord> {
        let media_root = self.media_root.clone();
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let record_id = new_uuid();

            let model_id = sanitize_optional_text(record.model_id.as_deref(), 160);
            let aligner_model_id = sanitize_optional_text(record.aligner_model_id.as_deref(), 160);
            let language = sanitize_optional_text(record.language.as_deref(), 80);
            let processing_error =
                sanitize_optional_text(record.processing_error.as_deref(), 1_000);
            let processing_status = normalize_processing_status(
                record.processing_status,
                processing_error.as_deref(),
            );
            let duration_secs = record.duration_secs.filter(|v| v.is_finite() && *v >= 0.0);
            let processing_time_ms = if record.processing_time_ms.is_finite() {
                record.processing_time_ms.max(0.0)
            } else {
                0.0
            };
            let rtf = record.rtf.filter(|v| v.is_finite() && *v >= 0.0);
            let audio_mime_type = sanitize_audio_mime_type(record.audio_mime_type.as_str());
            let audio_filename = sanitize_optional_text(record.audio_filename.as_deref(), 260);
            let transcription = sanitize_required_text(record.transcription.as_str(), 100_000);
            let segments = sanitize_segments(record.segments);
            let words = sanitize_words(record.words);
            let summary_model_id = sanitize_optional_text(record.summary_model_id.as_deref(), 160);
            let summary_text = sanitize_optional_text(record.summary_text.as_deref(), 20_000);
            let summary_error = sanitize_optional_text(record.summary_error.as_deref(), 1_000);
            let summary_status = normalize_summary_status(
                record.summary_status,
                summary_text.as_deref(),
                summary_error.as_deref(),
            );
            let summary_updated_at = normalize_optional_timestamp_i64(record.summary_updated_at)
                .or_else(|| {
                    if summary_status == TranscriptionSummaryStatus::NotRequested {
                        None
                    } else {
                        Some(now)
                    }
                });

            if record.audio_bytes.is_empty() {
                return Err(anyhow!("Audio payload cannot be empty"));
            }

            let audio_storage_path = storage_layout::persist_audio_file(
                &media_root,
                MediaGroup::Uploads,
                "transcription",
                &record_id,
                audio_filename.as_deref(),
                audio_mime_type.as_str(),
                &record.audio_bytes,
            )?;

            let segments_json =
                serde_json::to_string(&segments).context("Failed serializing segments")?;
            let words_json = serde_json::to_string(&words).context("Failed serializing words")?;

            if let Err(err) = conn.execute(
                r#"
                INSERT INTO transcription_records (
                    id,
                    created_at,
                    model_id,
                    aligner_model_id,
                    language,
                    processing_status,
                    processing_error,
                    duration_secs,
                    processing_time_ms,
                    rtf,
                    audio_mime_type,
                    audio_filename,
                    audio_storage_path,
                    transcription,
                    segments_json,
                    words_json,
                    summary_status,
                    summary_model_id,
                    summary_text,
                    summary_error,
                    summary_updated_at
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)
                "#,
                params![
                    &record_id,
                    now,
                    model_id,
                    aligner_model_id,
                    language,
                    processing_status.as_db_value(),
                    processing_error,
                    duration_secs,
                    processing_time_ms,
                    rtf,
                    audio_mime_type,
                    audio_filename,
                    audio_storage_path,
                    transcription,
                    segments_json,
                    words_json,
                    summary_status.as_db_value(),
                    summary_model_id,
                    summary_text,
                    summary_error,
                    summary_updated_at,
                ],
            ) {
                let _ = storage_layout::delete_media_file(
                    &media_root,
                    Some(audio_storage_path.as_str()),
                );
                return Err(err).context("Failed to insert transcription record");
            }

            let created = fetch_record_without_audio(&conn, &record_id)?
                .ok_or_else(|| anyhow!("Failed to fetch created transcription record"))?;
            Ok(created)
        })
        .await
    }

    pub async fn update_summary(
        &self,
        record_id: String,
        update: UpdateTranscriptionSummary,
    ) -> anyhow::Result<Option<TranscriptionRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();

            let summary_model_id = sanitize_optional_text(update.model_id.as_deref(), 160);
            let summary_text = sanitize_optional_text(update.text.as_deref(), 20_000);
            let summary_error = sanitize_optional_text(update.error.as_deref(), 1_000);
            let summary_status = normalize_summary_status(
                update.status,
                summary_text.as_deref(),
                summary_error.as_deref(),
            );
            let summary_updated_at =
                normalize_optional_timestamp_i64(update.updated_at).or_else(|| {
                    if summary_status == TranscriptionSummaryStatus::NotRequested {
                        None
                    } else {
                        Some(now)
                    }
                });

            let changed = conn.execute(
                r#"
                UPDATE transcription_records
                SET
                    summary_status = ?2,
                    summary_model_id = ?3,
                    summary_text = ?4,
                    summary_error = ?5,
                    summary_updated_at = ?6
                WHERE id = ?1
                "#,
                params![
                    record_id,
                    summary_status.as_db_value(),
                    summary_model_id,
                    summary_text,
                    summary_error,
                    summary_updated_at,
                ],
            )?;

            if changed == 0 {
                return Ok(None);
            }

            fetch_record_without_audio(&conn, &record_id)
        })
        .await
    }

    pub async fn update_processing_status(
        &self,
        record_id: String,
        status: TranscriptionProcessingStatus,
        error: Option<String>,
    ) -> anyhow::Result<Option<TranscriptionRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let processing_error = sanitize_optional_text(error.as_deref(), 1_000);
            let processing_status =
                normalize_processing_status(status, processing_error.as_deref());

            let changed = conn.execute(
                r#"
                UPDATE transcription_records
                SET
                    processing_status = ?2,
                    processing_error = ?3
                WHERE id = ?1
                "#,
                params![record_id, processing_status.as_db_value(), processing_error,],
            )?;

            if changed == 0 {
                return Ok(None);
            }

            fetch_record_without_audio(&conn, &record_id)
        })
        .await
    }

    pub async fn complete_record(
        &self,
        record_id: String,
        record: CompleteTranscriptionRecord,
    ) -> anyhow::Result<Option<TranscriptionRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;

            let model_id = sanitize_optional_text(record.model_id.as_deref(), 160);
            let aligner_model_id = sanitize_optional_text(record.aligner_model_id.as_deref(), 160);
            let language = sanitize_optional_text(record.language.as_deref(), 80);
            let duration_secs = record.duration_secs.filter(|v| v.is_finite() && *v >= 0.0);
            let processing_time_ms = if record.processing_time_ms.is_finite() {
                record.processing_time_ms.max(0.0)
            } else {
                0.0
            };
            let rtf = record.rtf.filter(|v| v.is_finite() && *v >= 0.0);
            let transcription = sanitize_required_text(record.transcription.as_str(), 100_000);
            let segments = sanitize_segments(record.segments);
            let words = sanitize_words(record.words);
            let summary_model_id = sanitize_optional_text(record.summary_model_id.as_deref(), 160);
            let summary_text = sanitize_optional_text(record.summary_text.as_deref(), 20_000);
            let summary_error = sanitize_optional_text(record.summary_error.as_deref(), 1_000);
            let summary_status = normalize_summary_status(
                record.summary_status,
                summary_text.as_deref(),
                summary_error.as_deref(),
            );
            let summary_updated_at = normalize_optional_timestamp_i64(record.summary_updated_at);
            let segments_json =
                serde_json::to_string(&segments).context("Failed serializing segments")?;
            let words_json = serde_json::to_string(&words).context("Failed serializing words")?;

            let changed = conn.execute(
                r#"
                UPDATE transcription_records
                SET
                    model_id = ?2,
                    aligner_model_id = ?3,
                    language = ?4,
                    processing_status = ?5,
                    processing_error = NULL,
                    duration_secs = ?6,
                    processing_time_ms = ?7,
                    rtf = ?8,
                    transcription = ?9,
                    segments_json = ?10,
                    words_json = ?11,
                    summary_status = ?12,
                    summary_model_id = ?13,
                    summary_text = ?14,
                    summary_error = ?15,
                    summary_updated_at = ?16
                WHERE id = ?1
                "#,
                params![
                    record_id,
                    model_id,
                    aligner_model_id,
                    language,
                    TranscriptionProcessingStatus::Ready.as_db_value(),
                    duration_secs,
                    processing_time_ms,
                    rtf,
                    transcription,
                    segments_json,
                    words_json,
                    summary_status.as_db_value(),
                    summary_model_id,
                    summary_text,
                    summary_error,
                    summary_updated_at,
                ],
            )?;

            if changed == 0 {
                return Ok(None);
            }

            fetch_record_without_audio(&conn, &record_id)
        })
        .await
    }

    pub async fn delete_record(&self, record_id: String) -> anyhow::Result<bool> {
        let media_root = self.media_root.clone();
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let audio_storage_path = conn
                .query_row(
                    "SELECT audio_storage_path FROM transcription_records WHERE id = ?1",
                    params![&record_id],
                    |row| row.get::<_, Option<String>>(0),
                )
                .optional()?
                .flatten();

            let changed = conn.execute(
                "DELETE FROM transcription_records WHERE id = ?1",
                params![record_id],
            )?;

            if changed > 0 {
                storage_layout::delete_media_file(&media_root, audio_storage_path.as_deref())?;
            }

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
            .map_err(|err| anyhow!("Transcription storage worker failed: {err}"))?
    }
}

fn fetch_record_without_audio(
    conn: &Connection,
    record_id: &str,
) -> anyhow::Result<Option<TranscriptionRecord>> {
    let record = conn
        .query_row(
            r#"
            SELECT
                id,
                created_at,
                model_id,
                aligner_model_id,
                language,
                processing_status,
                processing_error,
                duration_secs,
                processing_time_ms,
                rtf,
                audio_mime_type,
                audio_filename,
                transcription,
                segments_json,
                words_json,
                summary_status,
                summary_model_id,
                summary_text,
                summary_error,
                summary_updated_at
            FROM transcription_records
            WHERE id = ?1
            "#,
            params![record_id],
            map_transcription_record,
        )
        .optional()?;
    Ok(record)
}

fn map_transcription_record(row: &Row<'_>) -> rusqlite::Result<TranscriptionRecord> {
    Ok(TranscriptionRecord {
        id: row.get(0)?,
        created_at: i64_to_u64(row.get(1)?),
        model_id: row.get(2)?,
        aligner_model_id: row.get(3)?,
        language: row.get(4)?,
        processing_status: parse_processing_status(row.get(5)?),
        processing_error: row.get(6)?,
        duration_secs: row.get(7)?,
        processing_time_ms: row.get(8)?,
        rtf: row.get(9)?,
        audio_mime_type: row.get(10)?,
        audio_filename: row.get(11)?,
        transcription: row.get(12)?,
        segments: parse_json_vec(row.get(13)?),
        words: parse_json_vec(row.get(14)?),
        summary_status: parse_summary_status(row.get(15)?),
        summary_model_id: row.get(16)?,
        summary_text: row.get(17)?,
        summary_error: row.get(18)?,
        summary_updated_at: row.get::<_, Option<i64>>(19)?.map(i64_to_u64),
    })
}

fn parse_processing_status(raw: Option<String>) -> TranscriptionProcessingStatus {
    match raw
        .unwrap_or_else(|| "ready".to_string())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "pending" => TranscriptionProcessingStatus::Pending,
        "processing" => TranscriptionProcessingStatus::Processing,
        "failed" => TranscriptionProcessingStatus::Failed,
        _ => TranscriptionProcessingStatus::Ready,
    }
}

fn parse_summary_status(raw: Option<String>) -> TranscriptionSummaryStatus {
    match raw
        .unwrap_or_else(|| "not_requested".to_string())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "pending" => TranscriptionSummaryStatus::Pending,
        "ready" => TranscriptionSummaryStatus::Ready,
        "failed" => TranscriptionSummaryStatus::Failed,
        _ => TranscriptionSummaryStatus::NotRequested,
    }
}

fn normalize_processing_status(
    requested: TranscriptionProcessingStatus,
    processing_error: Option<&str>,
) -> TranscriptionProcessingStatus {
    if processing_error.is_some() {
        return TranscriptionProcessingStatus::Failed;
    }

    match requested {
        TranscriptionProcessingStatus::Failed => TranscriptionProcessingStatus::Pending,
        other => other,
    }
}

fn normalize_summary_status(
    requested: TranscriptionSummaryStatus,
    summary_text: Option<&str>,
    summary_error: Option<&str>,
) -> TranscriptionSummaryStatus {
    if summary_text.is_some() {
        return TranscriptionSummaryStatus::Ready;
    }
    if summary_error.is_some() {
        return TranscriptionSummaryStatus::Failed;
    }
    match requested {
        TranscriptionSummaryStatus::Ready => TranscriptionSummaryStatus::NotRequested,
        TranscriptionSummaryStatus::Failed => TranscriptionSummaryStatus::NotRequested,
        other => other,
    }
}

fn normalize_optional_timestamp_i64(value: Option<u64>) -> Option<i64> {
    value.and_then(|raw| i64::try_from(raw).ok())
}

fn transcription_preview(
    processing_status: TranscriptionProcessingStatus,
    processing_error: Option<&str>,
    content: &str,
) -> String {
    if processing_status == TranscriptionProcessingStatus::Pending {
        return "Queued for transcription".to_string();
    }
    if processing_status == TranscriptionProcessingStatus::Processing {
        return "Transcription in progress".to_string();
    }
    if processing_status == TranscriptionProcessingStatus::Failed {
        if let Some(error) = processing_error {
            let normalized = error.split_whitespace().collect::<Vec<_>>().join(" ");
            if !normalized.is_empty() {
                return truncate_string(&normalized, 160);
            }
        }
        return "Transcription failed".to_string();
    }

    let normalized = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return "No transcript".to_string();
    }
    truncate_string(&normalized, 160)
}

fn summary_preview(content: Option<&str>) -> Option<String> {
    let value = content.unwrap_or("").trim();
    if value.is_empty() {
        return None;
    }

    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        None
    } else {
        Some(truncate_string(&normalized, 200))
    }
}

fn sanitize_required_text(raw: &str, max_chars: usize) -> String {
    let normalized = raw.trim();
    if normalized.is_empty() {
        String::new()
    } else {
        truncate_string(normalized, max_chars)
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

fn sanitize_segments(segments: Vec<TranscriptionSegmentRecord>) -> Vec<TranscriptionSegmentRecord> {
    segments
        .into_iter()
        .filter_map(|segment| {
            let start = segment.start;
            let end = segment.end;
            let text = sanitize_required_text(segment.text.as_str(), 20_000);
            if !start.is_finite() || !end.is_finite() || end <= start || text.is_empty() {
                return None;
            }

            Some(TranscriptionSegmentRecord {
                start: start.max(0.0),
                end: end.max(start.max(0.0)),
                text,
                word_start: segment.word_start,
                word_end: segment.word_end,
            })
        })
        .collect()
}

fn sanitize_words(words: Vec<TranscriptionWordRecord>) -> Vec<TranscriptionWordRecord> {
    words
        .into_iter()
        .filter_map(|word| {
            let start = word.start;
            let end = word.end;
            let token = sanitize_required_text(word.word.as_str(), 160);
            if !start.is_finite() || !end.is_finite() || end <= start || token.is_empty() {
                return None;
            }

            Some(TranscriptionWordRecord {
                word: token,
                start: start.max(0.0),
                end: end.max(start.max(0.0)),
            })
        })
        .collect()
}

fn sanitize_audio_mime_type(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        "audio/wav".to_string()
    } else {
        truncate_string(trimmed, 80)
    }
}

fn ensure_transcription_records_aligner_model_id_column(conn: &Connection) -> anyhow::Result<()> {
    if transcription_records_has_column(conn, "aligner_model_id")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE transcription_records ADD COLUMN aligner_model_id TEXT NULL",
        [],
    )
    .context("Failed adding transcription_records.aligner_model_id column")?;
    Ok(())
}

fn ensure_transcription_records_processing_status_column(conn: &Connection) -> anyhow::Result<()> {
    if transcription_records_has_column(conn, "processing_status")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE transcription_records ADD COLUMN processing_status TEXT NOT NULL DEFAULT 'ready'",
        [],
    )
    .context("Failed adding transcription_records.processing_status column")?;
    Ok(())
}

fn ensure_transcription_records_processing_error_column(conn: &Connection) -> anyhow::Result<()> {
    if transcription_records_has_column(conn, "processing_error")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE transcription_records ADD COLUMN processing_error TEXT NULL",
        [],
    )
    .context("Failed adding transcription_records.processing_error column")?;
    Ok(())
}

fn ensure_transcription_records_segments_json_column(conn: &Connection) -> anyhow::Result<()> {
    if transcription_records_has_column(conn, "segments_json")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE transcription_records ADD COLUMN segments_json TEXT NOT NULL DEFAULT '[]'",
        [],
    )
    .context("Failed adding transcription_records.segments_json column")?;
    Ok(())
}

fn ensure_transcription_records_words_json_column(conn: &Connection) -> anyhow::Result<()> {
    if transcription_records_has_column(conn, "words_json")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE transcription_records ADD COLUMN words_json TEXT NOT NULL DEFAULT '[]'",
        [],
    )
    .context("Failed adding transcription_records.words_json column")?;
    Ok(())
}

fn ensure_transcription_records_summary_status_column(conn: &Connection) -> anyhow::Result<()> {
    if transcription_records_has_column(conn, "summary_status")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE transcription_records ADD COLUMN summary_status TEXT NOT NULL DEFAULT 'not_requested'",
        [],
    )
    .context("Failed adding transcription_records.summary_status column")?;
    Ok(())
}

fn ensure_transcription_records_summary_model_id_column(conn: &Connection) -> anyhow::Result<()> {
    if transcription_records_has_column(conn, "summary_model_id")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE transcription_records ADD COLUMN summary_model_id TEXT NULL",
        [],
    )
    .context("Failed adding transcription_records.summary_model_id column")?;
    Ok(())
}

fn ensure_transcription_records_summary_text_column(conn: &Connection) -> anyhow::Result<()> {
    if transcription_records_has_column(conn, "summary_text")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE transcription_records ADD COLUMN summary_text TEXT NULL",
        [],
    )
    .context("Failed adding transcription_records.summary_text column")?;
    Ok(())
}

fn ensure_transcription_records_summary_error_column(conn: &Connection) -> anyhow::Result<()> {
    if transcription_records_has_column(conn, "summary_error")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE transcription_records ADD COLUMN summary_error TEXT NULL",
        [],
    )
    .context("Failed adding transcription_records.summary_error column")?;
    Ok(())
}

fn ensure_transcription_records_summary_updated_at_column(conn: &Connection) -> anyhow::Result<()> {
    if transcription_records_has_column(conn, "summary_updated_at")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE transcription_records ADD COLUMN summary_updated_at INTEGER NULL",
        [],
    )
    .context("Failed adding transcription_records.summary_updated_at column")?;
    Ok(())
}

fn transcription_records_has_column(conn: &Connection, target: &str) -> anyhow::Result<bool> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(transcription_records)")
        .context("Failed to inspect transcription_records schema")?;
    let mut rows = stmt
        .query([])
        .context("Failed to query transcription_records schema info")?;

    while let Some(row) = rows
        .next()
        .context("Failed reading transcription_records schema row")?
    {
        let name: String = row
            .get(1)
            .context("Failed reading transcription_records column name")?;
        if name == target {
            return Ok(true);
        }
    }

    Ok(false)
}

fn parse_json_vec<T>(raw: Option<String>) -> Vec<T>
where
    T: for<'de> Deserialize<'de>,
{
    raw.and_then(|value| serde_json::from_str(value.as_str()).ok())
        .unwrap_or_default()
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

#[allow(dead_code)]
pub const fn default_list_limit() -> usize {
    DEFAULT_LIST_LIMIT
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_store() -> (TranscriptionStore, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "izwi-transcription-store-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let store =
            TranscriptionStore::initialize_at(root.join("izwi.sqlite3"), root.join("media"))
                .expect("test store should initialize");
        (store, root)
    }

    fn sample_record() -> NewTranscriptionRecord {
        NewTranscriptionRecord {
            model_id: Some("Parakeet-TDT-0.6B-v3".to_string()),
            aligner_model_id: Some("Qwen3-ForcedAligner-0.6B".to_string()),
            language: Some("English".to_string()),
            processing_status: TranscriptionProcessingStatus::Ready,
            processing_error: None,
            duration_secs: Some(6.0),
            processing_time_ms: 120.0,
            rtf: Some(0.5),
            audio_mime_type: "audio/wav".to_string(),
            audio_filename: Some("meeting.wav".to_string()),
            audio_bytes: vec![0_u8, 1_u8, 2_u8, 3_u8],
            transcription: "Hello there. General Kenobi.".to_string(),
            segments: vec![
                TranscriptionSegmentRecord {
                    start: 0.0,
                    end: 1.2,
                    text: "Hello there.".to_string(),
                    word_start: 0,
                    word_end: 1,
                },
                TranscriptionSegmentRecord {
                    start: 1.4,
                    end: 2.4,
                    text: "General Kenobi.".to_string(),
                    word_start: 2,
                    word_end: 3,
                },
            ],
            words: vec![
                TranscriptionWordRecord {
                    word: "Hello".to_string(),
                    start: 0.0,
                    end: 0.4,
                },
                TranscriptionWordRecord {
                    word: "there".to_string(),
                    start: 0.45,
                    end: 0.9,
                },
                TranscriptionWordRecord {
                    word: "General".to_string(),
                    start: 1.4,
                    end: 1.9,
                },
                TranscriptionWordRecord {
                    word: "Kenobi".to_string(),
                    start: 1.95,
                    end: 2.4,
                },
            ],
            summary_status: TranscriptionSummaryStatus::Ready,
            summary_model_id: Some("Qwen3.5-4B".to_string()),
            summary_text: Some("A short summary of the transcript.".to_string()),
            summary_error: None,
            summary_updated_at: Some(1),
        }
    }

    #[tokio::test]
    async fn persists_alignment_metadata() {
        let (store, root) = build_test_store();

        let created = store
            .create_record(sample_record())
            .await
            .expect("record should be created");

        assert_eq!(
            created.aligner_model_id.as_deref(),
            Some("Qwen3-ForcedAligner-0.6B")
        );
        assert_eq!(
            created.processing_status,
            TranscriptionProcessingStatus::Ready
        );
        assert_eq!(created.words.len(), 4);
        assert_eq!(created.segments.len(), 2);
        assert_eq!(created.summary_status, TranscriptionSummaryStatus::Ready);
        assert_eq!(created.summary_model_id.as_deref(), Some("Qwen3.5-4B"));
        assert_eq!(
            created.summary_text.as_deref(),
            Some("A short summary of the transcript.")
        );

        let fetched = store
            .get_record(created.id.clone())
            .await
            .expect("fetch should succeed")
            .expect("record should exist");

        assert_eq!(fetched.words.len(), 4);
        assert_eq!(fetched.segments[0].text, "Hello there.");

        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }

    #[tokio::test]
    async fn keeps_plain_transcriptions_without_timestamps() {
        let (store, root) = build_test_store();
        let mut record = sample_record();
        record.aligner_model_id = None;
        record.words = Vec::new();
        record.segments = Vec::new();
        record.processing_status = TranscriptionProcessingStatus::Ready;
        record.summary_status = TranscriptionSummaryStatus::NotRequested;
        record.summary_model_id = None;
        record.summary_text = None;
        record.summary_error = None;
        record.summary_updated_at = None;

        let created = store
            .create_record(record)
            .await
            .expect("record should be created");

        assert!(created.aligner_model_id.is_none());
        assert!(created.words.is_empty());
        assert!(created.segments.is_empty());
        assert_eq!(
            created.summary_status,
            TranscriptionSummaryStatus::NotRequested
        );
        assert!(created.summary_text.is_none());

        let summaries = store.list_records(10).await.expect("list should succeed");
        assert_eq!(summaries.len(), 1);
        assert!(summaries[0].transcription_preview.contains("Hello there."));
        assert_eq!(
            summaries[0].summary_status,
            TranscriptionSummaryStatus::NotRequested
        );

        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }

    #[tokio::test]
    async fn updates_pending_records_through_processing_and_completion() {
        let (store, root) = build_test_store();
        let mut record = sample_record();
        record.processing_status = TranscriptionProcessingStatus::Pending;
        record.duration_secs = None;
        record.processing_time_ms = 0.0;
        record.rtf = None;
        record.transcription = String::new();
        record.segments = Vec::new();
        record.words = Vec::new();
        record.summary_status = TranscriptionSummaryStatus::NotRequested;
        record.summary_model_id = None;
        record.summary_text = None;
        record.summary_error = None;
        record.summary_updated_at = None;

        let created = store
            .create_record(record)
            .await
            .expect("pending record should be created");
        assert_eq!(
            created.processing_status,
            TranscriptionProcessingStatus::Pending
        );
        assert_eq!(created.transcription, "");

        let processing = store
            .update_processing_status(
                created.id.clone(),
                TranscriptionProcessingStatus::Processing,
                None,
            )
            .await
            .expect("status update should succeed")
            .expect("record should exist");
        assert_eq!(
            processing.processing_status,
            TranscriptionProcessingStatus::Processing
        );

        let completed = store
            .complete_record(
                created.id.clone(),
                CompleteTranscriptionRecord {
                    model_id: Some("Parakeet-TDT-0.6B-v3".to_string()),
                    aligner_model_id: Some("Qwen3-ForcedAligner-0.6B".to_string()),
                    language: Some("English".to_string()),
                    duration_secs: Some(6.0),
                    processing_time_ms: 120.0,
                    rtf: Some(0.5),
                    transcription: "Hello there. General Kenobi.".to_string(),
                    segments: vec![TranscriptionSegmentRecord {
                        start: 0.0,
                        end: 1.2,
                        text: "Hello there.".to_string(),
                        word_start: 0,
                        word_end: 1,
                    }],
                    words: vec![TranscriptionWordRecord {
                        word: "Hello".to_string(),
                        start: 0.0,
                        end: 0.4,
                    }],
                    summary_status: TranscriptionSummaryStatus::Pending,
                    summary_model_id: Some("Qwen3.5-4B".to_string()),
                    summary_text: None,
                    summary_error: None,
                    summary_updated_at: None,
                },
            )
            .await
            .expect("completion should succeed")
            .expect("record should exist");

        assert_eq!(
            completed.processing_status,
            TranscriptionProcessingStatus::Ready
        );
        assert!(completed.processing_error.is_none());
        assert_eq!(
            completed.summary_status,
            TranscriptionSummaryStatus::Pending
        );
        assert_eq!(completed.transcription, "Hello there. General Kenobi.");

        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }

    #[tokio::test]
    async fn stores_failed_processing_errors_on_records() {
        let (store, root) = build_test_store();
        let mut record = sample_record();
        record.processing_status = TranscriptionProcessingStatus::Pending;
        record.transcription = String::new();
        record.segments = Vec::new();
        record.words = Vec::new();
        record.summary_status = TranscriptionSummaryStatus::NotRequested;
        record.summary_model_id = None;
        record.summary_text = None;
        record.summary_error = None;
        record.summary_updated_at = None;

        let created = store
            .create_record(record)
            .await
            .expect("pending record should be created");

        let failed = store
            .update_processing_status(
                created.id.clone(),
                TranscriptionProcessingStatus::Failed,
                Some("Runtime unavailable".to_string()),
            )
            .await
            .expect("failed status update should succeed")
            .expect("record should exist");

        assert_eq!(
            failed.processing_status,
            TranscriptionProcessingStatus::Failed
        );
        assert_eq!(
            failed.processing_error.as_deref(),
            Some("Runtime unavailable")
        );

        let summaries = store.list_records(10).await.expect("list should succeed");
        assert_eq!(
            summaries[0].processing_status,
            TranscriptionProcessingStatus::Failed
        );
        assert_eq!(summaries[0].transcription_preview, "Runtime unavailable");

        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }
}
