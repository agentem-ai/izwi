//! Persistent transcription history storage backed by SQLite.

use anyhow::{anyhow, Context};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::task;

const DEFAULT_LIST_LIMIT: usize = 200;

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionRecordSummary {
    pub id: String,
    pub created_at: u64,
    pub model_id: Option<String>,
    pub language: Option<String>,
    pub duration_secs: Option<f64>,
    pub processing_time_ms: f64,
    pub rtf: Option<f64>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub transcription_preview: String,
    pub transcription_chars: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionRecord {
    pub id: String,
    pub created_at: u64,
    pub model_id: Option<String>,
    pub language: Option<String>,
    pub duration_secs: Option<f64>,
    pub processing_time_ms: f64,
    pub rtf: Option<f64>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub transcription: String,
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
    pub language: Option<String>,
    pub duration_secs: Option<f64>,
    pub processing_time_ms: f64,
    pub rtf: Option<f64>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub audio_bytes: Vec<u8>,
    pub transcription: String,
}

#[derive(Clone)]
pub struct TranscriptionStore {
    db_path: PathBuf,
}

impl TranscriptionStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = resolve_db_path();

        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create transcription database directory: {}",
                    parent.display()
                )
            })?;
        }

        let conn = open_connection(&db_path).with_context(|| {
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
                language TEXT NULL,
                duration_secs REAL NULL,
                processing_time_ms REAL NOT NULL,
                rtf REAL NULL,
                audio_mime_type TEXT NOT NULL,
                audio_filename TEXT NULL,
                audio_bytes BLOB NOT NULL,
                transcription TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_transcription_records_created_at
                ON transcription_records(created_at DESC);
            "#,
        )
        .context("Failed to initialize transcription database schema")?;

        Ok(Self { db_path })
    }

    pub async fn list_records(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<TranscriptionRecordSummary>> {
        self.run_blocking(move |db_path| {
            let conn = open_connection(&db_path)?;
            let list_limit = i64::try_from(limit.clamp(1, 500).max(1)).unwrap_or(200);

            let mut stmt = conn.prepare(
                r#"
                SELECT
                    id,
                    created_at,
                    model_id,
                    language,
                    duration_secs,
                    processing_time_ms,
                    rtf,
                    audio_mime_type,
                    audio_filename,
                    transcription
                FROM transcription_records
                ORDER BY created_at DESC, id DESC
                LIMIT ?1
                "#,
            )?;

            let rows = stmt.query_map(params![list_limit], |row| {
                let transcription: String = row.get(9)?;
                Ok(TranscriptionRecordSummary {
                    id: row.get(0)?,
                    created_at: i64_to_u64(row.get(1)?),
                    model_id: row.get(2)?,
                    language: row.get(3)?,
                    duration_secs: row.get(4)?,
                    processing_time_ms: row.get(5)?,
                    rtf: row.get(6)?,
                    audio_mime_type: row.get(7)?,
                    audio_filename: row.get(8)?,
                    transcription_preview: transcription_preview(&transcription),
                    transcription_chars: transcription.chars().count(),
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
            let conn = open_connection(&db_path)?;
            let record = fetch_record_without_audio(&conn, &record_id)?;
            Ok(record)
        })
        .await
    }

    pub async fn get_audio(
        &self,
        record_id: String,
    ) -> anyhow::Result<Option<StoredTranscriptionAudio>> {
        self.run_blocking(move |db_path| {
            let conn = open_connection(&db_path)?;
            let audio = conn
                .query_row(
                    r#"
                    SELECT audio_bytes, audio_mime_type, audio_filename
                    FROM transcription_records
                    WHERE id = ?1
                    "#,
                    params![record_id],
                    |row| {
                        Ok(StoredTranscriptionAudio {
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
        record: NewTranscriptionRecord,
    ) -> anyhow::Result<TranscriptionRecord> {
        self.run_blocking(move |db_path| {
            let conn = open_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let record_id = format!("txr_{}", uuid::Uuid::new_v4().simple());

            let model_id = sanitize_optional_text(record.model_id.as_deref(), 160);
            let language = sanitize_optional_text(record.language.as_deref(), 80);
            let duration_secs = record.duration_secs.filter(|v| v.is_finite() && *v >= 0.0);
            let processing_time_ms = if record.processing_time_ms.is_finite() {
                record.processing_time_ms.max(0.0)
            } else {
                0.0
            };
            let rtf = record.rtf.filter(|v| v.is_finite() && *v >= 0.0);
            let audio_mime_type = sanitize_audio_mime_type(record.audio_mime_type.as_str());
            let audio_filename = sanitize_optional_text(record.audio_filename.as_deref(), 260);
            let transcription = record.transcription.trim().to_string();

            if record.audio_bytes.is_empty() {
                return Err(anyhow!("Audio payload cannot be empty"));
            }

            conn.execute(
                r#"
                INSERT INTO transcription_records (
                    id,
                    created_at,
                    model_id,
                    language,
                    duration_secs,
                    processing_time_ms,
                    rtf,
                    audio_mime_type,
                    audio_filename,
                    audio_bytes,
                    transcription
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
                "#,
                params![
                    &record_id,
                    now,
                    model_id,
                    language,
                    duration_secs,
                    processing_time_ms,
                    rtf,
                    audio_mime_type,
                    audio_filename,
                    record.audio_bytes,
                    transcription
                ],
            )?;

            let created = fetch_record_without_audio(&conn, &record_id)?
                .ok_or_else(|| anyhow!("Failed to fetch created transcription record"))?;
            Ok(created)
        })
        .await
    }

    pub async fn delete_record(&self, record_id: String) -> anyhow::Result<bool> {
        self.run_blocking(move |db_path| {
            let conn = open_connection(&db_path)?;
            let changed = conn.execute(
                "DELETE FROM transcription_records WHERE id = ?1",
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
                language,
                duration_secs,
                processing_time_ms,
                rtf,
                audio_mime_type,
                audio_filename,
                transcription
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
        language: row.get(3)?,
        duration_secs: row.get(4)?,
        processing_time_ms: row.get(5)?,
        rtf: row.get(6)?,
        audio_mime_type: row.get(7)?,
        audio_filename: row.get(8)?,
        transcription: row.get(9)?,
    })
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

fn transcription_preview(content: &str) -> String {
    let normalized = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return "No transcript".to_string();
    }
    truncate_string(&normalized, 160)
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

#[allow(dead_code)]
pub const fn default_list_limit() -> usize {
    DEFAULT_LIST_LIMIT
}
