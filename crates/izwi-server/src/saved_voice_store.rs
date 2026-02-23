//! Persistent saved voice storage backed by SQLite.

use anyhow::{anyhow, Context};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::task;

use crate::storage_layout::{self, MediaGroup};

const DEFAULT_LIST_LIMIT: usize = 500;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SavedVoiceSourceRouteKind {
    VoiceDesign,
    VoiceCloning,
}

impl SavedVoiceSourceRouteKind {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::VoiceDesign => "voice_design",
            Self::VoiceCloning => "voice_cloning",
        }
    }

    fn from_db_value(raw: &str) -> Option<Self> {
        match raw {
            "voice_design" => Some(Self::VoiceDesign),
            "voice_cloning" => Some(Self::VoiceCloning),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SavedVoiceSummary {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub name: String,
    pub reference_text_preview: String,
    pub reference_text_chars: usize,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub source_route_kind: Option<SavedVoiceSourceRouteKind>,
    pub source_record_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SavedVoice {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub name: String,
    pub reference_text: String,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub source_route_kind: Option<SavedVoiceSourceRouteKind>,
    pub source_record_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StoredSavedVoiceAudio {
    pub audio_bytes: Vec<u8>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewSavedVoice {
    pub name: String,
    pub reference_text: String,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub audio_bytes: Vec<u8>,
    pub source_route_kind: Option<SavedVoiceSourceRouteKind>,
    pub source_record_id: Option<String>,
}

#[derive(Clone)]
pub struct SavedVoiceStore {
    db_path: PathBuf,
    media_root: PathBuf,
}

impl SavedVoiceStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();

        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare saved voice storage layout")?;

        let conn = storage_layout::open_sqlite_connection(&db_path).with_context(|| {
            format!("Failed to open saved voice database: {}", db_path.display())
        })?;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS saved_voices (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                name TEXT NOT NULL COLLATE NOCASE,
                reference_text TEXT NOT NULL,
                audio_mime_type TEXT NOT NULL,
                audio_filename TEXT NULL,
                audio_storage_path TEXT NOT NULL,
                source_route_kind TEXT NULL,
                source_record_id TEXT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_saved_voices_name_nocase
                ON saved_voices(name COLLATE NOCASE);

            CREATE INDEX IF NOT EXISTS idx_saved_voices_updated_at
                ON saved_voices(updated_at DESC, created_at DESC);
            "#,
        )
        .context("Failed to initialize saved voice database schema")?;

        Ok(Self {
            db_path,
            media_root,
        })
    }

    pub async fn list_voices(&self, limit: usize) -> anyhow::Result<Vec<SavedVoiceSummary>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let list_limit = i64::try_from(limit.clamp(1, 2000).max(1)).unwrap_or(500);

            let mut stmt = conn.prepare(
                r#"
                SELECT
                    id,
                    created_at,
                    updated_at,
                    name,
                    reference_text,
                    audio_mime_type,
                    audio_filename,
                    source_route_kind,
                    source_record_id
                FROM saved_voices
                ORDER BY updated_at DESC, created_at DESC, id DESC
                LIMIT ?1
                "#,
            )?;

            let rows = stmt.query_map(params![list_limit], |row| {
                let reference_text: String = row.get(4)?;
                let source_route_raw: Option<String> = row.get(7)?;

                Ok(SavedVoiceSummary {
                    id: row.get(0)?,
                    created_at: i64_to_u64(row.get(1)?),
                    updated_at: i64_to_u64(row.get(2)?),
                    name: row.get(3)?,
                    reference_text_preview: reference_text_preview(reference_text.as_str()),
                    reference_text_chars: reference_text.chars().count(),
                    audio_mime_type: row.get(5)?,
                    audio_filename: row.get(6)?,
                    source_route_kind: source_route_raw
                        .as_deref()
                        .and_then(SavedVoiceSourceRouteKind::from_db_value),
                    source_record_id: row.get(8)?,
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

    pub async fn get_voice(&self, voice_id: String) -> anyhow::Result<Option<SavedVoice>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            fetch_voice_without_audio(&conn, &voice_id)
        })
        .await
    }

    pub async fn get_audio(
        &self,
        voice_id: String,
    ) -> anyhow::Result<Option<StoredSavedVoiceAudio>> {
        let media_root = self.media_root.clone();
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let audio = conn
                .query_row(
                    r#"
                    SELECT audio_storage_path, audio_mime_type, audio_filename
                    FROM saved_voices
                    WHERE id = ?1
                    "#,
                    params![voice_id],
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

            Ok(Some(StoredSavedVoiceAudio {
                audio_bytes,
                audio_mime_type,
                audio_filename,
            }))
        })
        .await
    }

    pub async fn create_voice(&self, voice: NewSavedVoice) -> anyhow::Result<SavedVoice> {
        let media_root = self.media_root.clone();
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;

            let now = now_unix_millis_i64();
            let voice_id = format!("sv_{}", uuid::Uuid::new_v4().simple());

            let name = sanitize_required_text(voice.name.as_str(), 120, "name")?;
            let reference_text =
                sanitize_required_text(voice.reference_text.as_str(), 4_000, "reference_text")?;
            let audio_mime_type = sanitize_audio_mime_type(voice.audio_mime_type.as_str());
            let audio_filename = sanitize_optional_text(voice.audio_filename.as_deref(), 260);
            let source_route_kind = voice
                .source_route_kind
                .map(SavedVoiceSourceRouteKind::as_db_value);
            let source_record_id = sanitize_optional_text(voice.source_record_id.as_deref(), 200);

            if voice.audio_bytes.is_empty() {
                return Err(anyhow!("Audio payload cannot be empty"));
            }

            let audio_storage_path = storage_layout::persist_audio_file(
                &media_root,
                MediaGroup::Generated,
                "saved-voices",
                &voice_id,
                audio_filename.as_deref(),
                audio_mime_type.as_str(),
                &voice.audio_bytes,
            )?;

            if let Err(err) = conn.execute(
                r#"
                INSERT INTO saved_voices (
                    id,
                    created_at,
                    updated_at,
                    name,
                    reference_text,
                    audio_mime_type,
                    audio_filename,
                    audio_storage_path,
                    source_route_kind,
                    source_record_id
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                "#,
                params![
                    &voice_id,
                    now,
                    now,
                    name,
                    reference_text,
                    audio_mime_type,
                    audio_filename,
                    audio_storage_path,
                    source_route_kind,
                    source_record_id,
                ],
            ) {
                let _ = storage_layout::delete_media_file(
                    &media_root,
                    Some(audio_storage_path.as_str()),
                );
                return Err(err).context("Failed to insert saved voice");
            }

            fetch_voice_without_audio(&conn, &voice_id)?
                .ok_or_else(|| anyhow!("Failed to fetch created saved voice"))
        })
        .await
    }

    pub async fn delete_voice(&self, voice_id: String) -> anyhow::Result<bool> {
        let media_root = self.media_root.clone();
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let audio_storage_path = conn
                .query_row(
                    "SELECT audio_storage_path FROM saved_voices WHERE id = ?1",
                    params![&voice_id],
                    |row| row.get::<_, Option<String>>(0),
                )
                .optional()?
                .flatten();

            let changed =
                conn.execute("DELETE FROM saved_voices WHERE id = ?1", params![voice_id])?;

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
            .map_err(|err| anyhow!("Saved voice storage worker failed: {err}"))?
    }
}

fn fetch_voice_without_audio(
    conn: &Connection,
    voice_id: &str,
) -> anyhow::Result<Option<SavedVoice>> {
    let voice = conn
        .query_row(
            r#"
            SELECT
                id,
                created_at,
                updated_at,
                name,
                reference_text,
                audio_mime_type,
                audio_filename,
                source_route_kind,
                source_record_id
            FROM saved_voices
            WHERE id = ?1
            "#,
            params![voice_id],
            map_saved_voice,
        )
        .optional()?;
    Ok(voice)
}

fn map_saved_voice(row: &Row<'_>) -> rusqlite::Result<SavedVoice> {
    let source_route_raw: Option<String> = row.get(7)?;
    Ok(SavedVoice {
        id: row.get(0)?,
        created_at: i64_to_u64(row.get(1)?),
        updated_at: i64_to_u64(row.get(2)?),
        name: row.get(3)?,
        reference_text: row.get(4)?,
        audio_mime_type: row.get(5)?,
        audio_filename: row.get(6)?,
        source_route_kind: source_route_raw
            .as_deref()
            .and_then(SavedVoiceSourceRouteKind::from_db_value),
        source_record_id: row.get(8)?,
    })
}

fn sanitize_required_text(raw: &str, max_chars: usize, field_name: &str) -> anyhow::Result<String> {
    let normalized = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return Err(anyhow!("{field_name} cannot be empty"));
    }
    Ok(truncate_string(&normalized, max_chars))
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

fn reference_text_preview(content: &str) -> String {
    let normalized = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return "No reference text".to_string();
    }
    truncate_string(&normalized, 140)
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
