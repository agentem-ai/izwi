//! Persistent text-to-speech project storage backed by SQLite.

use anyhow::Context;
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::task;

use crate::storage_layout::{self};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StudioProjectVoiceMode {
    BuiltIn,
    Saved,
}

impl StudioProjectVoiceMode {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::BuiltIn => "built_in",
            Self::Saved => "saved",
        }
    }

    fn from_db_value(raw: &str) -> Option<Self> {
        match raw {
            "built_in" => Some(Self::BuiltIn),
            "saved" => Some(Self::Saved),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StudioProjectExportFormat {
    Wav,
    Mp3,
    Flac,
}

impl StudioProjectExportFormat {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Flac => "flac",
        }
    }

    fn from_db_value(raw: &str) -> Option<Self> {
        match raw {
            "wav" => Some(Self::Wav),
            "mp3" => Some(Self::Mp3),
            "flac" => Some(Self::Flac),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StudioProjectRenderJobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl StudioProjectRenderJobStatus {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    fn from_db_value(raw: &str) -> Option<Self> {
        match raw {
            "queued" => Some(Self::Queued),
            "running" => Some(Self::Running),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            "cancelled" => Some(Self::Cancelled),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StudioProjectFolderRecord {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub name: String,
    pub parent_id: Option<String>,
    pub sort_order: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudioProjectMetaRecord {
    pub project_id: String,
    pub folder_id: Option<String>,
    pub tags: Vec<String>,
    pub default_export_format: StudioProjectExportFormat,
    pub last_render_job_id: Option<String>,
    pub last_rendered_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudioProjectPronunciationRecord {
    pub id: String,
    pub project_id: String,
    pub source_text: String,
    pub replacement_text: String,
    pub locale: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudioProjectSnapshotRecord {
    pub id: String,
    pub project_id: String,
    pub created_at: u64,
    pub label: Option<String>,
    pub project_name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudioProjectRenderJobRecord {
    pub id: String,
    pub project_id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub status: StudioProjectRenderJobStatus,
    pub error_message: Option<String>,
    pub queued_segment_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StudioProjectSummary {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub name: String,
    pub source_filename: Option<String>,
    pub model_id: Option<String>,
    pub voice_mode: StudioProjectVoiceMode,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
    pub speed: Option<f64>,
    pub segment_count: usize,
    pub rendered_segment_count: usize,
    pub total_chars: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudioProjectSegmentRecord {
    pub id: String,
    pub project_id: String,
    pub position: usize,
    pub text: String,
    pub model_id: Option<String>,
    pub voice_mode: Option<StudioProjectVoiceMode>,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
    pub input_chars: usize,
    pub speech_record_id: Option<String>,
    pub updated_at: u64,
    pub generation_time_ms: Option<f64>,
    pub audio_duration_secs: Option<f64>,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudioProjectRecord {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub name: String,
    pub source_filename: Option<String>,
    pub source_text: String,
    pub model_id: Option<String>,
    pub voice_mode: StudioProjectVoiceMode,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
    pub speed: Option<f64>,
    pub segments: Vec<StudioProjectSegmentRecord>,
}

#[derive(Debug, Clone)]
pub struct NewStudioProjectSegment {
    pub position: usize,
    pub text: String,
    pub model_id: Option<String>,
    pub voice_mode: Option<StudioProjectVoiceMode>,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewStudioProjectRecord {
    pub name: String,
    pub source_filename: Option<String>,
    pub source_text: String,
    pub model_id: Option<String>,
    pub voice_mode: StudioProjectVoiceMode,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
    pub speed: Option<f64>,
    pub segments: Vec<NewStudioProjectSegment>,
}

#[derive(Debug, Clone, Default)]
pub struct UpdateStudioProjectRecord {
    pub name: Option<String>,
    pub model_id: Option<String>,
    pub voice_mode: Option<StudioProjectVoiceMode>,
    pub speaker: Option<Option<String>>,
    pub saved_voice_id: Option<Option<String>>,
    pub speed: Option<Option<f64>>,
}

#[derive(Debug, Clone)]
pub struct NewStudioProjectFolderRecord {
    pub name: String,
    pub parent_id: Option<String>,
    pub sort_order: Option<i64>,
}

#[derive(Debug, Clone, Default)]
pub struct UpsertStudioProjectMetaRecord {
    pub folder_id: Option<Option<String>>,
    pub tags: Option<Vec<String>>,
    pub default_export_format: Option<StudioProjectExportFormat>,
    pub last_render_job_id: Option<Option<String>>,
    pub last_rendered_at: Option<Option<u64>>,
}

#[derive(Debug, Clone)]
pub struct NewStudioProjectPronunciationRecord {
    pub source_text: String,
    pub replacement_text: String,
    pub locale: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewStudioProjectSnapshotRecord {
    pub label: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewStudioProjectRenderJobRecord {
    pub queued_segment_ids: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct UpdateStudioProjectRenderJobRecord {
    pub status: Option<StudioProjectRenderJobStatus>,
    pub error_message: Option<Option<String>>,
}

#[derive(Clone)]
pub struct StudioProjectStore {
    db_path: PathBuf,
}

impl StudioProjectStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();

        Self::initialize_at(db_path, media_root)
    }

    fn initialize_at(db_path: PathBuf, media_root: PathBuf) -> anyhow::Result<Self> {
        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare Studio project storage layout")?;

        let conn = storage_layout::open_sqlite_connection(&db_path).with_context(|| {
            format!("Failed to open Studio project database: {}", db_path.display())
        })?;
        migrate_legacy_tts_schema(&conn)?;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS studio_projects (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                name TEXT NOT NULL,
                source_filename TEXT NULL,
                source_text TEXT NOT NULL,
                model_id TEXT NULL,
                voice_mode TEXT NOT NULL,
                speaker TEXT NULL,
                saved_voice_id TEXT NULL,
                speed REAL NULL
            );

            CREATE TABLE IF NOT EXISTS studio_project_segments (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                text TEXT NOT NULL,
                model_id TEXT NULL,
                voice_mode TEXT NULL,
                speaker TEXT NULL,
                saved_voice_id TEXT NULL,
                speech_record_id TEXT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS studio_project_folders (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                name TEXT NOT NULL,
                parent_id TEXT NULL,
                sort_order INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS studio_project_meta (
                project_id TEXT PRIMARY KEY,
                folder_id TEXT NULL,
                tags_json TEXT NOT NULL DEFAULT '[]',
                default_export_format TEXT NOT NULL DEFAULT 'wav',
                last_render_job_id TEXT NULL,
                last_rendered_at INTEGER NULL,
                FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE,
                FOREIGN KEY(folder_id) REFERENCES studio_project_folders(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS studio_project_pronunciations (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                source_text TEXT NOT NULL,
                replacement_text TEXT NOT NULL,
                locale TEXT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS studio_project_snapshots (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                label TEXT NULL,
                project_json TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS studio_project_render_jobs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT NULL,
                queued_segment_ids_json TEXT NOT NULL DEFAULT '[]',
                FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_studio_projects_updated_at
                ON studio_projects(updated_at DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_studio_project_segments_project_position
                ON studio_project_segments(project_id, position ASC);
            CREATE INDEX IF NOT EXISTS idx_studio_project_folders_parent
                ON studio_project_folders(parent_id, sort_order ASC, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_studio_project_pronunciations_project
                ON studio_project_pronunciations(project_id, updated_at DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_studio_project_snapshots_project
                ON studio_project_snapshots(project_id, created_at DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_studio_project_render_jobs_project
                ON studio_project_render_jobs(project_id, created_at DESC, id DESC);
            "#,
        )
        .context("Failed to initialize Studio project database schema")?;

        ensure_studio_project_segment_settings_columns(&conn)?;

        Ok(Self { db_path })
    }

    pub async fn list_projects(&self, limit: usize) -> anyhow::Result<Vec<StudioProjectSummary>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let list_limit = i64::try_from(limit.clamp(1, 200).max(1)).unwrap_or(100);
            let mut stmt = conn.prepare(
                r#"
                SELECT
                    p.id,
                    p.created_at,
                    p.updated_at,
                    p.name,
                    p.source_filename,
                    p.model_id,
                    p.voice_mode,
                    p.speaker,
                    p.saved_voice_id,
                    p.speed,
                    COUNT(s.id) AS segment_count,
                    COALESCE(SUM(CASE WHEN s.speech_record_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS rendered_segment_count,
                    COALESCE(SUM(LENGTH(s.text)), 0) AS total_chars
                FROM studio_projects p
                LEFT JOIN studio_project_segments s ON s.project_id = p.id
                GROUP BY
                    p.id,
                    p.created_at,
                    p.updated_at,
                    p.name,
                    p.source_filename,
                    p.model_id,
                    p.voice_mode,
                    p.speaker,
                    p.saved_voice_id,
                    p.speed
                ORDER BY p.updated_at DESC, p.id DESC
                LIMIT ?1
                "#,
            )?;

            let rows = stmt.query_map(params![list_limit], map_project_summary_row)?;
            let mut projects = Vec::new();
            for row in rows {
                projects.push(row?);
            }
            Ok(projects)
        })
        .await
    }

    pub async fn get_project(
        &self,
        project_id: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn create_project(
        &self,
        record: NewStudioProjectRecord,
    ) -> anyhow::Result<StudioProjectRecord> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;

            let now = now_unix_millis_i64();
            let project_id = format!("ttsp_{}", uuid::Uuid::new_v4().simple());

            tx.execute(
                r#"
                INSERT INTO studio_projects (
                    id,
                    created_at,
                    updated_at,
                    name,
                    source_filename,
                    source_text,
                    model_id,
                    voice_mode,
                    speaker,
                    saved_voice_id,
                    speed
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
                "#,
                params![
                    project_id,
                    now,
                    now,
                    record.name,
                    record.source_filename,
                    record.source_text,
                    record.model_id,
                    record.voice_mode.as_db_value(),
                    record.speaker,
                    record.saved_voice_id,
                    record.speed,
                ],
            )?;

            for segment in record.segments {
                let segment_id = format!("ttss_{}", uuid::Uuid::new_v4().simple());
                tx.execute(
                    r#"
                    INSERT INTO studio_project_segments (
                        id,
                        project_id,
                        position,
                        text,
                        model_id,
                        voice_mode,
                        speaker,
                        saved_voice_id,
                        speech_record_id,
                        updated_at
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, ?9)
                    "#,
                    params![
                        segment_id,
                        project_id,
                        usize_to_i64(segment.position),
                        segment.text,
                        segment.model_id,
                        segment.voice_mode.map(|value| value.as_db_value()),
                        segment.speaker,
                        segment.saved_voice_id,
                        now,
                    ],
                )?;
            }

            tx.commit()?;
            fetch_project(&conn, &project_id)?
                .ok_or_else(|| anyhow::anyhow!("Created Studio project was not found"))
        })
        .await
    }

    pub async fn update_project(
        &self,
        project_id: String,
        update: UpdateStudioProjectRecord,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;

            let current = tx
                .query_row(
                    r#"
                    SELECT
                        name,
                        model_id,
                        voice_mode,
                        speaker,
                        saved_voice_id,
                        speed
                    FROM studio_projects
                    WHERE id = ?1
                    "#,
                    params![project_id.as_str()],
                    |row| {
                        let voice_mode_raw: String = row.get(2)?;
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, Option<String>>(1)?,
                            StudioProjectVoiceMode::from_db_value(voice_mode_raw.as_str())
                                .unwrap_or(StudioProjectVoiceMode::BuiltIn),
                            row.get::<_, Option<String>>(3)?,
                            row.get::<_, Option<String>>(4)?,
                            row.get::<_, Option<f64>>(5)?,
                        ))
                    },
                )
                .optional()?;

            let Some((name, model_id, voice_mode, speaker, saved_voice_id, speed)) = current else {
                return Ok(None);
            };

            let next_name = update.name.unwrap_or(name);
            let next_model_id = update.model_id.or(model_id);
            let next_voice_mode = update.voice_mode.unwrap_or(voice_mode);
            let next_speaker = update.speaker.unwrap_or(speaker);
            let next_saved_voice_id = update.saved_voice_id.unwrap_or(saved_voice_id);
            let next_speed = update.speed.unwrap_or(speed);
            let now = now_unix_millis_i64();

            tx.execute(
                r#"
                UPDATE studio_projects
                SET
                    updated_at = ?2,
                    name = ?3,
                    model_id = ?4,
                    voice_mode = ?5,
                    speaker = ?6,
                    saved_voice_id = ?7,
                    speed = ?8
                WHERE id = ?1
                "#,
                params![
                    project_id.as_str(),
                    now,
                    next_name,
                    next_model_id,
                    next_voice_mode.as_db_value(),
                    next_speaker,
                    next_saved_voice_id,
                    next_speed,
                ],
            )?;

            tx.commit()?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn update_segment_text(
        &self,
        project_id: String,
        segment_id: String,
        text: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let now = now_unix_millis_i64();

            let updated = tx.execute(
                r#"
                UPDATE studio_project_segments
                SET
                    text = ?3,
                    speech_record_id = NULL,
                    updated_at = ?4
                WHERE project_id = ?1 AND id = ?2
                "#,
                params![project_id.as_str(), segment_id.as_str(), text, now],
            )?;

            if updated == 0 {
                tx.rollback()?;
                return Ok(None);
            }

            sync_project_source_text(&tx, project_id.as_str())?;
            touch_project(&tx, project_id.as_str(), now)?;
            tx.commit()?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn update_segment_settings(
        &self,
        project_id: String,
        segment_id: String,
        model_id: String,
        voice_mode: StudioProjectVoiceMode,
        speaker: Option<String>,
        saved_voice_id: Option<String>,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let now = now_unix_millis_i64();

            let updated = tx.execute(
                r#"
                UPDATE studio_project_segments
                SET
                    model_id = ?3,
                    voice_mode = ?4,
                    speaker = ?5,
                    saved_voice_id = ?6,
                    speech_record_id = NULL,
                    updated_at = ?7
                WHERE project_id = ?1 AND id = ?2
                "#,
                params![
                    project_id.as_str(),
                    segment_id.as_str(),
                    model_id,
                    voice_mode.as_db_value(),
                    speaker,
                    saved_voice_id,
                    now,
                ],
            )?;

            if updated == 0 {
                tx.rollback()?;
                return Ok(None);
            }

            touch_project(&tx, project_id.as_str(), now)?;
            tx.commit()?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn update_segment_text_and_settings(
        &self,
        project_id: String,
        segment_id: String,
        text: String,
        model_id: String,
        voice_mode: StudioProjectVoiceMode,
        speaker: Option<String>,
        saved_voice_id: Option<String>,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let now = now_unix_millis_i64();

            let updated = tx.execute(
                r#"
                UPDATE studio_project_segments
                SET
                    text = ?3,
                    model_id = ?4,
                    voice_mode = ?5,
                    speaker = ?6,
                    saved_voice_id = ?7,
                    speech_record_id = NULL,
                    updated_at = ?8
                WHERE project_id = ?1 AND id = ?2
                "#,
                params![
                    project_id.as_str(),
                    segment_id.as_str(),
                    text,
                    model_id,
                    voice_mode.as_db_value(),
                    speaker,
                    saved_voice_id,
                    now,
                ],
            )?;

            if updated == 0 {
                tx.rollback()?;
                return Ok(None);
            }

            sync_project_source_text(&tx, project_id.as_str())?;
            touch_project(&tx, project_id.as_str(), now)?;
            tx.commit()?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn split_segment(
        &self,
        project_id: String,
        segment_id: String,
        before_text: String,
        after_text: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let now = now_unix_millis_i64();

            let segment_state = tx
                .query_row(
                    r#"
                    SELECT
                        position,
                        model_id,
                        voice_mode,
                        speaker,
                        saved_voice_id
                    FROM studio_project_segments
                    WHERE project_id = ?1 AND id = ?2
                    "#,
                    params![project_id.as_str(), segment_id.as_str()],
                    |row| {
                        let voice_mode_raw: Option<String> = row.get(2)?;
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, Option<String>>(1)?,
                            voice_mode_raw
                                .as_deref()
                                .and_then(StudioProjectVoiceMode::from_db_value),
                            row.get::<_, Option<String>>(3)?,
                            row.get::<_, Option<String>>(4)?,
                        ))
                    },
                )
                .optional()?;

            let Some((position, model_id, voice_mode, speaker, saved_voice_id)) = segment_state
            else {
                tx.rollback()?;
                return Ok(None);
            };

            tx.execute(
                r#"
                UPDATE studio_project_segments
                SET position = position + 1
                WHERE project_id = ?1 AND position > ?2
                "#,
                params![project_id.as_str(), position],
            )?;

            tx.execute(
                r#"
                UPDATE studio_project_segments
                SET
                    text = ?3,
                    speech_record_id = NULL,
                    updated_at = ?4
                WHERE project_id = ?1 AND id = ?2
                "#,
                params![project_id.as_str(), segment_id.as_str(), before_text, now],
            )?;

            let next_segment_id = format!("ttss_{}", uuid::Uuid::new_v4().simple());
            tx.execute(
                r#"
                INSERT INTO studio_project_segments (
                    id,
                    project_id,
                    position,
                    text,
                    model_id,
                    voice_mode,
                    speaker,
                    saved_voice_id,
                    speech_record_id,
                    updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, ?9)
                "#,
                params![
                    next_segment_id,
                    project_id.as_str(),
                    position + 1,
                    after_text,
                    model_id,
                    voice_mode.map(|value| value.as_db_value()),
                    speaker,
                    saved_voice_id,
                    now
                ],
            )?;

            sync_project_source_text(&tx, project_id.as_str())?;
            touch_project(&tx, project_id.as_str(), now)?;
            tx.commit()?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn delete_segment(
        &self,
        project_id: String,
        segment_id: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let now = now_unix_millis_i64();

            let position = tx
                .query_row(
                    r#"
                    SELECT position
                    FROM studio_project_segments
                    WHERE project_id = ?1 AND id = ?2
                    "#,
                    params![project_id.as_str(), segment_id.as_str()],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?;

            let Some(position) = position else {
                tx.rollback()?;
                return Ok(None);
            };

            let deleted = tx.execute(
                r#"
                DELETE FROM studio_project_segments
                WHERE project_id = ?1 AND id = ?2
                "#,
                params![project_id.as_str(), segment_id.as_str()],
            )?;

            if deleted == 0 {
                tx.rollback()?;
                return Ok(None);
            }

            tx.execute(
                r#"
                UPDATE studio_project_segments
                SET position = position - 1
                WHERE project_id = ?1 AND position > ?2
                "#,
                params![project_id.as_str(), position],
            )?;

            sync_project_source_text(&tx, project_id.as_str())?;
            touch_project(&tx, project_id.as_str(), now)?;
            tx.commit()?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn merge_segment_with_next(
        &self,
        project_id: String,
        segment_id: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let now = now_unix_millis_i64();

            let current = tx
                .query_row(
                    r#"
                    SELECT id, position, text
                    FROM studio_project_segments
                    WHERE project_id = ?1 AND id = ?2
                    "#,
                    params![project_id.as_str(), segment_id.as_str()],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, i64>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    },
                )
                .optional()?;
            let Some((current_id, current_position, current_text)) = current else {
                tx.rollback()?;
                return Ok(None);
            };

            let next = tx
                .query_row(
                    r#"
                    SELECT id, position, text
                    FROM studio_project_segments
                    WHERE project_id = ?1 AND position = ?2
                    "#,
                    params![project_id.as_str(), current_position + 1],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, i64>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    },
                )
                .optional()?;
            let Some((next_id, next_position, next_text)) = next else {
                tx.rollback()?;
                return Ok(None);
            };

            let merged_text = format!("{}\n\n{}", current_text.trim(), next_text.trim())
                .trim()
                .to_string();
            tx.execute(
                r#"
                UPDATE studio_project_segments
                SET
                    text = ?3,
                    speech_record_id = NULL,
                    updated_at = ?4
                WHERE project_id = ?1 AND id = ?2
                "#,
                params![project_id.as_str(), current_id.as_str(), merged_text, now],
            )?;

            tx.execute(
                r#"
                DELETE FROM studio_project_segments
                WHERE project_id = ?1 AND id = ?2
                "#,
                params![project_id.as_str(), next_id.as_str()],
            )?;

            tx.execute(
                r#"
                UPDATE studio_project_segments
                SET position = position - 1
                WHERE project_id = ?1 AND position > ?2
                "#,
                params![project_id.as_str(), next_position],
            )?;

            sync_project_source_text(&tx, project_id.as_str())?;
            touch_project(&tx, project_id.as_str(), now)?;
            tx.commit()?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn reorder_segments(
        &self,
        project_id: String,
        ordered_segment_ids: Vec<String>,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;

            let existing_ids = {
                let mut stmt = tx.prepare(
                    r#"
                    SELECT id
                    FROM studio_project_segments
                    WHERE project_id = ?1
                    ORDER BY position ASC, id ASC
                    "#,
                )?;
                let rows =
                    stmt.query_map(params![project_id.as_str()], |row| row.get::<_, String>(0))?;
                let mut ids = Vec::new();
                for row in rows {
                    ids.push(row?);
                }
                ids
            };
            if existing_ids.is_empty() {
                tx.rollback()?;
                return Ok(None);
            }
            if ordered_segment_ids.len() != existing_ids.len() {
                anyhow::bail!("Reorder request must include every project segment exactly once.");
            }

            let existing_set = existing_ids.iter().collect::<std::collections::HashSet<_>>();
            let requested_set = ordered_segment_ids
                .iter()
                .collect::<std::collections::HashSet<_>>();
            if existing_set != requested_set {
                anyhow::bail!("Reorder request contains unknown or missing segment ids.");
            }

            let now = now_unix_millis_i64();
            for (position, segment_id) in ordered_segment_ids.iter().enumerate() {
                tx.execute(
                    r#"
                    UPDATE studio_project_segments
                    SET position = ?3, updated_at = ?4, speech_record_id = NULL
                    WHERE project_id = ?1 AND id = ?2
                    "#,
                    params![
                        project_id.as_str(),
                        segment_id.as_str(),
                        usize_to_i64(position),
                        now
                    ],
                )?;
            }

            sync_project_source_text(&tx, project_id.as_str())?;
            touch_project(&tx, project_id.as_str(), now)?;
            tx.commit()?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn delete_segments(
        &self,
        project_id: String,
        segment_ids: Vec<String>,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let existing = {
                let mut stmt = tx.prepare(
                    r#"
                    SELECT id, position
                    FROM studio_project_segments
                    WHERE project_id = ?1
                    ORDER BY position ASC, id ASC
                    "#,
                )?;
                let rows = stmt.query_map(params![project_id.as_str()], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                })?;
                let mut records = Vec::new();
                for row in rows {
                    records.push(row?);
                }
                records
            };
            if existing.is_empty() {
                tx.rollback()?;
                return Ok(None);
            }

            let remove_set = segment_ids
                .into_iter()
                .collect::<std::collections::HashSet<_>>();
            if remove_set.is_empty() {
                tx.rollback()?;
                return fetch_project(&conn, &project_id);
            }

            let remaining_count = existing
                .iter()
                .filter(|(id, _)| !remove_set.contains(id))
                .count();
            if remaining_count == 0 {
                anyhow::bail!("A project must keep at least one segment.");
            }

            for (segment_id, _) in &existing {
                if remove_set.contains(segment_id) {
                    tx.execute(
                        r#"
                        DELETE FROM studio_project_segments
                        WHERE project_id = ?1 AND id = ?2
                        "#,
                        params![project_id.as_str(), segment_id.as_str()],
                    )?;
                }
            }

            let mut new_position = 0usize;
            for (segment_id, _) in existing {
                if remove_set.contains(segment_id.as_str()) {
                    continue;
                }
                tx.execute(
                    r#"
                    UPDATE studio_project_segments
                    SET position = ?3
                    WHERE project_id = ?1 AND id = ?2
                    "#,
                    params![project_id.as_str(), segment_id.as_str(), usize_to_i64(new_position)],
                )?;
                new_position += 1;
            }

            let now = now_unix_millis_i64();
            sync_project_source_text(&tx, project_id.as_str())?;
            touch_project(&tx, project_id.as_str(), now)?;
            tx.commit()?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn attach_segment_record(
        &self,
        project_id: String,
        segment_id: String,
        speech_record_id: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let now = now_unix_millis_i64();

            let updated = tx.execute(
                r#"
                UPDATE studio_project_segments
                SET
                    speech_record_id = ?3,
                    updated_at = ?4
                WHERE project_id = ?1 AND id = ?2
                "#,
                params![
                    project_id.as_str(),
                    segment_id.as_str(),
                    speech_record_id,
                    now
                ],
            )?;

            if updated == 0 {
                tx.rollback()?;
                return Ok(None);
            }

            touch_project(&tx, project_id.as_str(), now)?;
            tx.commit()?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn delete_project(&self, project_id: String) -> anyhow::Result<bool> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            tx.execute(
                "DELETE FROM studio_project_segments WHERE project_id = ?1",
                params![project_id.as_str()],
            )?;
            let deleted = tx.execute(
                "DELETE FROM studio_projects WHERE id = ?1",
                params![project_id.as_str()],
            )?;
            tx.commit()?;
            Ok(deleted > 0)
        })
        .await
    }

    pub async fn list_folders(&self) -> anyhow::Result<Vec<StudioProjectFolderRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let mut stmt = conn.prepare(
                r#"
                SELECT id, created_at, updated_at, name, parent_id, sort_order
                FROM studio_project_folders
                ORDER BY
                    CASE WHEN parent_id IS NULL THEN 0 ELSE 1 END ASC,
                    parent_id ASC,
                    sort_order ASC,
                    updated_at DESC,
                    id DESC
                "#,
            )?;
            let rows = stmt.query_map([], map_folder_row)?;
            let mut folders = Vec::new();
            for row in rows {
                folders.push(row?);
            }
            Ok(folders)
        })
        .await
    }

    pub async fn create_folder(
        &self,
        record: NewStudioProjectFolderRecord,
    ) -> anyhow::Result<StudioProjectFolderRecord> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let id = format!("ttpf_{}", uuid::Uuid::new_v4().simple());
            conn.execute(
                r#"
                INSERT INTO studio_project_folders (id, created_at, updated_at, name, parent_id, sort_order)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                "#,
                params![
                    id.as_str(),
                    now,
                    now,
                    record.name,
                    record.parent_id,
                    record.sort_order.unwrap_or(0),
                ],
            )?;
            conn.query_row(
                r#"
                SELECT id, created_at, updated_at, name, parent_id, sort_order
                FROM studio_project_folders
                WHERE id = ?1
                "#,
                params![id.as_str()],
                map_folder_row,
            )
            .map_err(Into::into)
        })
        .await
    }

    pub async fn get_project_meta(
        &self,
        project_id: String,
    ) -> anyhow::Result<Option<StudioProjectMetaRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            fetch_project_meta(&conn, project_id.as_str())
        })
        .await
    }

    pub async fn upsert_project_meta(
        &self,
        project_id: String,
        update: UpsertStudioProjectMetaRecord,
    ) -> anyhow::Result<Option<StudioProjectMetaRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let exists = tx
                .query_row(
                    "SELECT 1 FROM studio_projects WHERE id = ?1",
                    params![project_id.as_str()],
                    |_| Ok(()),
                )
                .optional()?
                .is_some();
            if !exists {
                tx.rollback()?;
                return Ok(None);
            }

            let current = tx
                .query_row(
                    r#"
                    SELECT folder_id, tags_json, default_export_format, last_render_job_id, last_rendered_at
                    FROM studio_project_meta
                    WHERE project_id = ?1
                    "#,
                    params![project_id.as_str()],
                    |row| {
                        Ok((
                            row.get::<_, Option<String>>(0)?,
                            row.get::<_, Option<String>>(1)?,
                            row.get::<_, Option<String>>(2)?,
                            row.get::<_, Option<String>>(3)?,
                            row.get::<_, Option<i64>>(4)?,
                        ))
                    },
                )
                .optional()?;

            let default_current_export = current
                .as_ref()
                .and_then(|(_, _, format, _, _)| format.as_deref())
                .and_then(StudioProjectExportFormat::from_db_value)
                .unwrap_or(StudioProjectExportFormat::Wav);
            let next_folder_id = update.folder_id.unwrap_or_else(|| {
                current
                    .as_ref()
                    .and_then(|(folder_id, _, _, _, _)| folder_id.clone())
            });
            let next_tags = update.tags.unwrap_or_else(|| {
                current
                    .as_ref()
                    .and_then(|(_, tags_json, _, _, _)| tags_json.clone())
                    .map(|raw| parse_json_string_vec(Some(raw)))
                    .unwrap_or_default()
            });
            let next_export = update.default_export_format.unwrap_or(default_current_export);
            let next_last_render_job_id = update.last_render_job_id.unwrap_or_else(|| {
                current
                    .as_ref()
                    .and_then(|(_, _, _, last_render_job_id, _)| last_render_job_id.clone())
            });
            let next_last_rendered_at = update.last_rendered_at.unwrap_or_else(|| {
                current
                    .as_ref()
                    .and_then(|(_, _, _, _, last_rendered_at)| *last_rendered_at)
                    .map(i64_to_u64)
            });
            let next_last_rendered_at_i64 = next_last_rendered_at
                .and_then(|value| i64::try_from(value).ok());

            tx.execute(
                r#"
                INSERT INTO studio_project_meta (
                    project_id,
                    folder_id,
                    tags_json,
                    default_export_format,
                    last_render_job_id,
                    last_rendered_at
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                ON CONFLICT(project_id) DO UPDATE SET
                    folder_id = excluded.folder_id,
                    tags_json = excluded.tags_json,
                    default_export_format = excluded.default_export_format,
                    last_render_job_id = excluded.last_render_job_id,
                    last_rendered_at = excluded.last_rendered_at
                "#,
                params![
                    project_id.as_str(),
                    next_folder_id,
                    encode_json_string_vec(next_tags.as_slice()),
                    next_export.as_db_value(),
                    next_last_render_job_id,
                    next_last_rendered_at_i64,
                ],
            )?;

            tx.commit()?;
            fetch_project_meta(&conn, project_id.as_str())
        })
        .await
    }

    pub async fn list_project_pronunciations(
        &self,
        project_id: String,
    ) -> anyhow::Result<Vec<StudioProjectPronunciationRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let mut stmt = conn.prepare(
                r#"
                SELECT id, project_id, source_text, replacement_text, locale, created_at, updated_at
                FROM studio_project_pronunciations
                WHERE project_id = ?1
                ORDER BY updated_at DESC, id DESC
                "#,
            )?;
            let rows = stmt.query_map(params![project_id.as_str()], map_pronunciation_row)?;
            let mut entries = Vec::new();
            for row in rows {
                entries.push(row?);
            }
            Ok(entries)
        })
        .await
    }

    pub async fn create_project_pronunciation(
        &self,
        project_id: String,
        record: NewStudioProjectPronunciationRecord,
    ) -> anyhow::Result<Option<StudioProjectPronunciationRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let exists = conn
                .query_row(
                    "SELECT 1 FROM studio_projects WHERE id = ?1",
                    params![project_id.as_str()],
                    |_| Ok(()),
                )
                .optional()?
                .is_some();
            if !exists {
                return Ok(None);
            }

            let now = now_unix_millis_i64();
            let id = format!("ttpp_{}", uuid::Uuid::new_v4().simple());
            conn.execute(
                r#"
                INSERT INTO studio_project_pronunciations (
                    id, project_id, source_text, replacement_text, locale, created_at, updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                "#,
                params![
                    id.as_str(),
                    project_id.as_str(),
                    record.source_text,
                    record.replacement_text,
                    record.locale,
                    now,
                    now,
                ],
            )?;

            conn.query_row(
                r#"
                SELECT id, project_id, source_text, replacement_text, locale, created_at, updated_at
                FROM studio_project_pronunciations
                WHERE id = ?1
                "#,
                params![id.as_str()],
                map_pronunciation_row,
            )
            .optional()
            .map_err(Into::into)
        })
        .await
    }

    pub async fn delete_project_pronunciation(
        &self,
        project_id: String,
        pronunciation_id: String,
    ) -> anyhow::Result<bool> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let deleted = conn.execute(
                r#"
                DELETE FROM studio_project_pronunciations
                WHERE project_id = ?1 AND id = ?2
                "#,
                params![project_id.as_str(), pronunciation_id.as_str()],
            )?;
            Ok(deleted > 0)
        })
        .await
    }

    pub async fn list_project_snapshots(
        &self,
        project_id: String,
    ) -> anyhow::Result<Vec<StudioProjectSnapshotRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let mut stmt = conn.prepare(
                r#"
                SELECT
                    s.id,
                    s.project_id,
                    s.created_at,
                    s.label,
                    COALESCE(json_extract(s.project_json, '$.name'), p.name) AS project_name
                FROM studio_project_snapshots s
                LEFT JOIN studio_projects p ON p.id = s.project_id
                WHERE s.project_id = ?1
                ORDER BY s.created_at DESC, s.id DESC
                "#,
            )?;
            let rows = stmt.query_map(params![project_id.as_str()], map_snapshot_row)?;
            let mut snapshots = Vec::new();
            for row in rows {
                snapshots.push(row?);
            }
            Ok(snapshots)
        })
        .await
    }

    pub async fn create_project_snapshot(
        &self,
        project_id: String,
        record: NewStudioProjectSnapshotRecord,
    ) -> anyhow::Result<Option<StudioProjectSnapshotRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let Some(project) = fetch_project(&tx, project_id.as_str())? else {
                tx.rollback()?;
                return Ok(None);
            };
            let payload = serde_json::to_string(&project)
                .context("Failed to serialize project snapshot payload")?;
            let now = now_unix_millis_i64();
            let snapshot_id = format!("ttps_{}", uuid::Uuid::new_v4().simple());
            tx.execute(
                r#"
                INSERT INTO studio_project_snapshots (id, project_id, created_at, label, project_json)
                VALUES (?1, ?2, ?3, ?4, ?5)
                "#,
                params![
                    snapshot_id.as_str(),
                    project_id.as_str(),
                    now,
                    record.label,
                    payload,
                ],
            )?;
            tx.commit()?;

            conn.query_row(
                r#"
                SELECT
                    s.id,
                    s.project_id,
                    s.created_at,
                    s.label,
                    COALESCE(json_extract(s.project_json, '$.name'), p.name) AS project_name
                FROM studio_project_snapshots s
                LEFT JOIN studio_projects p ON p.id = s.project_id
                WHERE s.id = ?1
                "#,
                params![snapshot_id.as_str()],
                map_snapshot_row,
            )
            .optional()
            .map_err(Into::into)
        })
        .await
    }

    pub async fn restore_project_snapshot(
        &self,
        project_id: String,
        snapshot_id: String,
    ) -> anyhow::Result<Option<StudioProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let snapshot_json = tx
                .query_row(
                    r#"
                    SELECT project_json
                    FROM studio_project_snapshots
                    WHERE id = ?1 AND project_id = ?2
                    "#,
                    params![snapshot_id.as_str(), project_id.as_str()],
                    |row| row.get::<_, String>(0),
                )
                .optional()?;
            let Some(snapshot_json) = snapshot_json else {
                tx.rollback()?;
                return Ok(None);
            };

            let snapshot: StudioProjectRecord = serde_json::from_str(snapshot_json.as_str())
                .context("Failed to decode project snapshot payload")?;
            let now = now_unix_millis_i64();
            tx.execute(
                r#"
                UPDATE studio_projects
                SET
                    updated_at = ?2,
                    name = ?3,
                    source_filename = ?4,
                    source_text = ?5,
                    model_id = ?6,
                    voice_mode = ?7,
                    speaker = ?8,
                    saved_voice_id = ?9,
                    speed = ?10
                WHERE id = ?1
                "#,
                params![
                    project_id.as_str(),
                    now,
                    snapshot.name,
                    snapshot.source_filename,
                    snapshot.source_text,
                    snapshot.model_id,
                    snapshot.voice_mode.as_db_value(),
                    snapshot.speaker,
                    snapshot.saved_voice_id,
                    snapshot.speed,
                ],
            )?;

            tx.execute(
                "DELETE FROM studio_project_segments WHERE project_id = ?1",
                params![project_id.as_str()],
            )?;
            for segment in snapshot.segments {
                tx.execute(
                    r#"
                    INSERT INTO studio_project_segments (
                        id,
                        project_id,
                        position,
                        text,
                        model_id,
                        voice_mode,
                        speaker,
                        saved_voice_id,
                        speech_record_id,
                        updated_at
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                    "#,
                    params![
                        segment.id,
                        project_id.as_str(),
                        usize_to_i64(segment.position),
                        segment.text,
                        segment.model_id,
                        segment.voice_mode.map(|value| value.as_db_value()),
                        segment.speaker,
                        segment.saved_voice_id,
                        segment.speech_record_id,
                        now,
                    ],
                )?;
            }
            touch_project(&tx, project_id.as_str(), now)?;
            tx.commit()?;
            fetch_project(&conn, project_id.as_str())
        })
        .await
    }

    pub async fn list_project_render_jobs(
        &self,
        project_id: String,
    ) -> anyhow::Result<Vec<StudioProjectRenderJobRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let mut stmt = conn.prepare(
                r#"
                SELECT
                    id,
                    project_id,
                    created_at,
                    updated_at,
                    status,
                    error_message,
                    queued_segment_ids_json
                FROM studio_project_render_jobs
                WHERE project_id = ?1
                ORDER BY created_at DESC, id DESC
                "#,
            )?;
            let rows = stmt.query_map(params![project_id.as_str()], map_render_job_row)?;
            let mut jobs = Vec::new();
            for row in rows {
                jobs.push(row?);
            }
            Ok(jobs)
        })
        .await
    }

    pub async fn create_project_render_job(
        &self,
        project_id: String,
        record: NewStudioProjectRenderJobRecord,
    ) -> anyhow::Result<Option<StudioProjectRenderJobRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let exists = tx
                .query_row(
                    "SELECT 1 FROM studio_projects WHERE id = ?1",
                    params![project_id.as_str()],
                    |_| Ok(()),
                )
                .optional()?
                .is_some();
            if !exists {
                tx.rollback()?;
                return Ok(None);
            }
            let now = now_unix_millis_i64();
            let job_id = format!("ttsj_{}", uuid::Uuid::new_v4().simple());
            tx.execute(
                r#"
                INSERT INTO studio_project_render_jobs (
                    id,
                    project_id,
                    created_at,
                    updated_at,
                    status,
                    error_message,
                    queued_segment_ids_json
                ) VALUES (?1, ?2, ?3, ?4, ?5, NULL, ?6)
                "#,
                params![
                    job_id.as_str(),
                    project_id.as_str(),
                    now,
                    now,
                    StudioProjectRenderJobStatus::Queued.as_db_value(),
                    encode_json_string_vec(record.queued_segment_ids.as_slice()),
                ],
            )?;

            tx.execute(
                r#"
                INSERT INTO studio_project_meta (
                    project_id,
                    folder_id,
                    tags_json,
                    default_export_format,
                    last_render_job_id,
                    last_rendered_at
                )
                VALUES (
                    ?1,
                    (SELECT folder_id FROM studio_project_meta WHERE project_id = ?1),
                    COALESCE((SELECT tags_json FROM studio_project_meta WHERE project_id = ?1), '[]'),
                    COALESCE((SELECT default_export_format FROM studio_project_meta WHERE project_id = ?1), 'wav'),
                    ?2,
                    (SELECT last_rendered_at FROM studio_project_meta WHERE project_id = ?1)
                )
                ON CONFLICT(project_id) DO UPDATE SET
                    last_render_job_id = excluded.last_render_job_id
                "#,
                params![project_id.as_str(), job_id.as_str()],
            )?;

            tx.commit()?;
            fetch_render_job(&conn, project_id.as_str(), job_id.as_str())
        })
        .await
    }

    pub async fn update_project_render_job(
        &self,
        project_id: String,
        job_id: String,
        update: UpdateStudioProjectRenderJobRecord,
    ) -> anyhow::Result<Option<StudioProjectRenderJobRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let current = tx
                .query_row(
                    r#"
                    SELECT status, error_message
                    FROM studio_project_render_jobs
                    WHERE project_id = ?1 AND id = ?2
                    "#,
                    params![project_id.as_str(), job_id.as_str()],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, Option<String>>(1)?,
                        ))
                    },
                )
                .optional()?;
            let Some((current_status, current_error)) = current else {
                tx.rollback()?;
                return Ok(None);
            };

            let next_status = update
                .status
                .or_else(|| StudioProjectRenderJobStatus::from_db_value(current_status.as_str()))
                .unwrap_or(StudioProjectRenderJobStatus::Queued);
            let next_error = update.error_message.unwrap_or(current_error);
            let now = now_unix_millis_i64();
            tx.execute(
                r#"
                UPDATE studio_project_render_jobs
                SET updated_at = ?3, status = ?4, error_message = ?5
                WHERE project_id = ?1 AND id = ?2
                "#,
                params![
                    project_id.as_str(),
                    job_id.as_str(),
                    now,
                    next_status.as_db_value(),
                    next_error,
                ],
            )?;

            if next_status == StudioProjectRenderJobStatus::Completed {
                tx.execute(
                    r#"
                    INSERT INTO studio_project_meta (
                        project_id,
                        folder_id,
                        tags_json,
                        default_export_format,
                        last_render_job_id,
                        last_rendered_at
                    )
                    VALUES (
                        ?1,
                        (SELECT folder_id FROM studio_project_meta WHERE project_id = ?1),
                        COALESCE((SELECT tags_json FROM studio_project_meta WHERE project_id = ?1), '[]'),
                        COALESCE((SELECT default_export_format FROM studio_project_meta WHERE project_id = ?1), 'wav'),
                        ?2,
                        ?3
                    )
                    ON CONFLICT(project_id) DO UPDATE SET
                        last_render_job_id = excluded.last_render_job_id,
                        last_rendered_at = excluded.last_rendered_at
                    "#,
                    params![project_id.as_str(), job_id.as_str(), now],
                )?;
            }

            tx.commit()?;
            fetch_render_job(&conn, project_id.as_str(), job_id.as_str())
        })
        .await
    }

    async fn run_blocking<T, F>(&self, work: F) -> anyhow::Result<T>
    where
        T: Send + 'static,
        F: FnOnce(PathBuf) -> anyhow::Result<T> + Send + 'static,
    {
        let db_path = self.db_path.clone();
        task::spawn_blocking(move || work(db_path))
            .await
            .context("Studio project storage task join error")?
    }
}

fn migrate_legacy_tts_schema(conn: &Connection) -> anyhow::Result<()> {
    let legacy_exists = sqlite_table_exists(conn, "tts_projects")?;
    let studio_exists = sqlite_table_exists(conn, "studio_projects")?;

    if !legacy_exists || studio_exists {
        return Ok(());
    }

    conn.execute_batch(
        r#"
        ALTER TABLE tts_projects RENAME TO studio_projects;
        ALTER TABLE tts_project_segments RENAME TO studio_project_segments;
        ALTER TABLE tts_project_folders RENAME TO studio_project_folders;
        ALTER TABLE tts_project_meta RENAME TO studio_project_meta;
        ALTER TABLE tts_project_pronunciations RENAME TO studio_project_pronunciations;
        ALTER TABLE tts_project_snapshots RENAME TO studio_project_snapshots;
        ALTER TABLE tts_project_render_jobs RENAME TO studio_project_render_jobs;

        DROP INDEX IF EXISTS idx_tts_projects_updated_at;
        DROP INDEX IF EXISTS idx_tts_project_segments_project_position;
        DROP INDEX IF EXISTS idx_tts_project_folders_parent;
        DROP INDEX IF EXISTS idx_tts_project_pronunciations_project;
        DROP INDEX IF EXISTS idx_tts_project_snapshots_project;
        DROP INDEX IF EXISTS idx_tts_project_render_jobs_project;
        "#,
    )
    .context("Failed to migrate legacy TTS project schema to Studio naming")?;

    Ok(())
}

fn sqlite_table_exists(conn: &Connection, table_name: &str) -> anyhow::Result<bool> {
    let marker: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1 LIMIT 1",
            params![table_name],
            |row| row.get(0),
        )
        .optional()?;
    Ok(marker.is_some())
}

fn fetch_project(conn: &Connection, project_id: &str) -> anyhow::Result<Option<StudioProjectRecord>> {
    let project = conn
        .query_row(
            r#"
            SELECT
                id,
                created_at,
                updated_at,
                name,
                source_filename,
                source_text,
                model_id,
                voice_mode,
                speaker,
                saved_voice_id,
                speed
            FROM studio_projects
            WHERE id = ?1
            "#,
            params![project_id],
            map_project_row,
        )
        .optional()?;

    let Some(mut project) = project else {
        return Ok(None);
    };

    let mut stmt = conn.prepare(
        r#"
        SELECT
            s.id,
            s.project_id,
            s.position,
            s.text,
            s.model_id,
            s.voice_mode,
            s.speaker,
            s.saved_voice_id,
            s.speech_record_id,
            s.updated_at,
            h.generation_time_ms,
            h.audio_duration_secs,
            h.audio_filename
        FROM studio_project_segments s
        LEFT JOIN speech_history_records h
            ON h.id = s.speech_record_id
           AND h.route_kind = 'text_to_speech'
        WHERE s.project_id = ?1
        ORDER BY s.position ASC, s.id ASC
        "#,
    )?;

    let rows = stmt.query_map(params![project_id], map_segment_row)?;
    let mut segments = Vec::new();
    for row in rows {
        segments.push(row?);
    }
    project.segments = segments;

    Ok(Some(project))
}

fn map_project_summary_row(row: &Row<'_>) -> rusqlite::Result<StudioProjectSummary> {
    let voice_mode_raw: String = row.get(6)?;
    let segment_count = row.get::<_, i64>(10)?;
    let rendered_segment_count = row.get::<_, i64>(11)?;
    let total_chars = row.get::<_, i64>(12)?;

    Ok(StudioProjectSummary {
        id: row.get(0)?,
        created_at: i64_to_u64(row.get(1)?),
        updated_at: i64_to_u64(row.get(2)?),
        name: row.get(3)?,
        source_filename: row.get(4)?,
        model_id: row.get(5)?,
        voice_mode: StudioProjectVoiceMode::from_db_value(voice_mode_raw.as_str())
            .unwrap_or(StudioProjectVoiceMode::BuiltIn),
        speaker: row.get(7)?,
        saved_voice_id: row.get(8)?,
        speed: row.get(9)?,
        segment_count: i64_to_usize(segment_count).unwrap_or_default(),
        rendered_segment_count: i64_to_usize(rendered_segment_count).unwrap_or_default(),
        total_chars: i64_to_usize(total_chars).unwrap_or_default(),
    })
}

fn map_project_row(row: &Row<'_>) -> rusqlite::Result<StudioProjectRecord> {
    let voice_mode_raw: String = row.get(7)?;
    Ok(StudioProjectRecord {
        id: row.get(0)?,
        created_at: i64_to_u64(row.get(1)?),
        updated_at: i64_to_u64(row.get(2)?),
        name: row.get(3)?,
        source_filename: row.get(4)?,
        source_text: row.get(5)?,
        model_id: row.get(6)?,
        voice_mode: StudioProjectVoiceMode::from_db_value(voice_mode_raw.as_str())
            .unwrap_or(StudioProjectVoiceMode::BuiltIn),
        speaker: row.get(8)?,
        saved_voice_id: row.get(9)?,
        speed: row.get(10)?,
        segments: Vec::new(),
    })
}

fn map_segment_row(row: &Row<'_>) -> rusqlite::Result<StudioProjectSegmentRecord> {
    let text: String = row.get(3)?;
    let voice_mode_raw: Option<String> = row.get(5)?;
    Ok(StudioProjectSegmentRecord {
        id: row.get(0)?,
        project_id: row.get(1)?,
        position: i64_to_usize(row.get(2)?).unwrap_or_default(),
        input_chars: text.chars().count(),
        text,
        model_id: row.get(4)?,
        voice_mode: voice_mode_raw
            .as_deref()
            .and_then(StudioProjectVoiceMode::from_db_value),
        speaker: row.get(6)?,
        saved_voice_id: row.get(7)?,
        speech_record_id: row.get(8)?,
        updated_at: i64_to_u64(row.get(9)?),
        generation_time_ms: row.get(10)?,
        audio_duration_secs: row.get(11)?,
        audio_filename: row.get(12)?,
    })
}

fn sqlite_table_has_column(
    conn: &Connection,
    table_name: &str,
    column_name: &str,
) -> anyhow::Result<bool> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table_name})"))?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let name: String = row.get(1)?;
        if name == column_name {
            return Ok(true);
        }
    }
    Ok(false)
}

fn ensure_sqlite_table_column(
    conn: &Connection,
    table_name: &str,
    column_name: &str,
    column_definition: &str,
) -> anyhow::Result<()> {
    if sqlite_table_has_column(conn, table_name, column_name)? {
        return Ok(());
    }
    conn.execute(
        &format!(
            "ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        ),
        [],
    )?;
    Ok(())
}

fn ensure_studio_project_segment_settings_columns(conn: &Connection) -> anyhow::Result<()> {
    ensure_sqlite_table_column(conn, "studio_project_segments", "model_id", "TEXT NULL")?;
    ensure_sqlite_table_column(conn, "studio_project_segments", "voice_mode", "TEXT NULL")?;
    ensure_sqlite_table_column(conn, "studio_project_segments", "speaker", "TEXT NULL")?;
    ensure_sqlite_table_column(conn, "studio_project_segments", "saved_voice_id", "TEXT NULL")?;
    Ok(())
}

fn fetch_project_meta(
    conn: &Connection,
    project_id: &str,
) -> anyhow::Result<Option<StudioProjectMetaRecord>> {
    conn.query_row(
        r#"
        SELECT
            project_id,
            folder_id,
            tags_json,
            default_export_format,
            last_render_job_id,
            last_rendered_at
        FROM studio_project_meta
        WHERE project_id = ?1
        "#,
        params![project_id],
        map_meta_row,
    )
    .optional()
    .map_err(Into::into)
}

fn fetch_render_job(
    conn: &Connection,
    project_id: &str,
    job_id: &str,
) -> anyhow::Result<Option<StudioProjectRenderJobRecord>> {
    conn.query_row(
        r#"
        SELECT
            id,
            project_id,
            created_at,
            updated_at,
            status,
            error_message,
            queued_segment_ids_json
        FROM studio_project_render_jobs
        WHERE project_id = ?1 AND id = ?2
        "#,
        params![project_id, job_id],
        map_render_job_row,
    )
    .optional()
    .map_err(Into::into)
}

fn map_folder_row(row: &Row<'_>) -> rusqlite::Result<StudioProjectFolderRecord> {
    Ok(StudioProjectFolderRecord {
        id: row.get(0)?,
        created_at: i64_to_u64(row.get(1)?),
        updated_at: i64_to_u64(row.get(2)?),
        name: row.get(3)?,
        parent_id: row.get(4)?,
        sort_order: row.get(5)?,
    })
}

fn map_meta_row(row: &Row<'_>) -> rusqlite::Result<StudioProjectMetaRecord> {
    let tags_json = row.get::<_, Option<String>>(2)?;
    let export_format_raw = row.get::<_, Option<String>>(3)?;
    let last_rendered_at = row.get::<_, Option<i64>>(5)?;
    Ok(StudioProjectMetaRecord {
        project_id: row.get(0)?,
        folder_id: row.get(1)?,
        tags: parse_json_string_vec(tags_json),
        default_export_format: export_format_raw
            .as_deref()
            .and_then(StudioProjectExportFormat::from_db_value)
            .unwrap_or(StudioProjectExportFormat::Wav),
        last_render_job_id: row.get(4)?,
        last_rendered_at: last_rendered_at.map(i64_to_u64),
    })
}

fn map_pronunciation_row(row: &Row<'_>) -> rusqlite::Result<StudioProjectPronunciationRecord> {
    Ok(StudioProjectPronunciationRecord {
        id: row.get(0)?,
        project_id: row.get(1)?,
        source_text: row.get(2)?,
        replacement_text: row.get(3)?,
        locale: row.get(4)?,
        created_at: i64_to_u64(row.get(5)?),
        updated_at: i64_to_u64(row.get(6)?),
    })
}

fn map_snapshot_row(row: &Row<'_>) -> rusqlite::Result<StudioProjectSnapshotRecord> {
    Ok(StudioProjectSnapshotRecord {
        id: row.get(0)?,
        project_id: row.get(1)?,
        created_at: i64_to_u64(row.get(2)?),
        label: row.get(3)?,
        project_name: row.get(4)?,
    })
}

fn map_render_job_row(row: &Row<'_>) -> rusqlite::Result<StudioProjectRenderJobRecord> {
    let status_raw: String = row.get(4)?;
    let queued_segment_ids_json = row.get::<_, Option<String>>(6)?;
    Ok(StudioProjectRenderJobRecord {
        id: row.get(0)?,
        project_id: row.get(1)?,
        created_at: i64_to_u64(row.get(2)?),
        updated_at: i64_to_u64(row.get(3)?),
        status: StudioProjectRenderJobStatus::from_db_value(status_raw.as_str())
            .unwrap_or(StudioProjectRenderJobStatus::Queued),
        error_message: row.get(5)?,
        queued_segment_ids: parse_json_string_vec(queued_segment_ids_json),
    })
}

fn parse_json_string_vec(raw: Option<String>) -> Vec<String> {
    raw.and_then(|value| serde_json::from_str::<Vec<String>>(value.as_str()).ok())
        .unwrap_or_default()
}

fn encode_json_string_vec(values: &[String]) -> String {
    serde_json::to_string(values).unwrap_or_else(|_| "[]".to_string())
}

fn touch_project(conn: &Connection, project_id: &str, updated_at: i64) -> anyhow::Result<()> {
    conn.execute(
        "UPDATE studio_projects SET updated_at = ?2 WHERE id = ?1",
        params![project_id, updated_at],
    )?;
    Ok(())
}

fn sync_project_source_text(conn: &Connection, project_id: &str) -> anyhow::Result<()> {
    let mut stmt = conn.prepare(
        r#"
        SELECT text
        FROM studio_project_segments
        WHERE project_id = ?1
        ORDER BY position ASC, id ASC
        "#,
    )?;
    let rows = stmt.query_map(params![project_id], |row| row.get::<_, String>(0))?;

    let mut segment_texts = Vec::new();
    for row in rows {
        segment_texts.push(row?);
    }

    conn.execute(
        "UPDATE studio_projects SET source_text = ?2 WHERE id = ?1",
        params![project_id, segment_texts.join("\n\n")],
    )?;
    Ok(())
}

fn now_unix_millis_i64() -> i64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    i64::try_from(millis).unwrap_or(i64::MAX)
}

fn i64_to_u64(value: i64) -> u64 {
    u64::try_from(value).unwrap_or_default()
}

fn i64_to_usize(value: i64) -> Option<usize> {
    usize::try_from(value).ok()
}

fn usize_to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{
        NewStudioProjectRecord, NewStudioProjectSegment, StudioProjectStore, StudioProjectVoiceMode,
        UpdateStudioProjectRecord,
    };
    use crate::storage_layout;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_env_root(prefix: &str) -> PathBuf {
        let mut root = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        root.push(format!("izwi-{prefix}-{nonce}"));
        root
    }

    #[tokio::test]
    async fn create_and_update_project_round_trips_segments() {
        let root = test_env_root("studio-project-store");
        let db_path = root.join("test.sqlite3");
        let media_dir = root.join("media");
        std::fs::create_dir_all(&root).expect("root should be created");

        let store =
            StudioProjectStore::initialize_at(db_path, media_dir).expect("store should initialize");
        let conn = storage_layout::open_sqlite_connection(&store.db_path)
            .expect("speech schema connection should open");
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS speech_history_records (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                route_kind TEXT NOT NULL,
                model_id TEXT NULL,
                speaker TEXT NULL,
                language TEXT NULL,
                saved_voice_id TEXT NULL,
                speed REAL NULL,
                input_text TEXT NOT NULL,
                voice_description TEXT NULL,
                reference_text TEXT NULL,
                generation_time_ms REAL NULL,
                audio_duration_secs REAL NULL,
                rtf REAL NULL,
                tokens_generated INTEGER NULL,
                audio_mime_type TEXT NOT NULL,
                audio_filename TEXT NULL,
                audio_storage_path TEXT NOT NULL
            );
            "#,
        )
        .expect("speech schema should initialize");
        let created = store
            .create_project(NewStudioProjectRecord {
                name: "Narration project".to_string(),
                source_filename: Some("script.txt".to_string()),
                source_text: "Hello world. Another sentence.".to_string(),
                model_id: Some("Qwen3-TTS".to_string()),
                voice_mode: StudioProjectVoiceMode::BuiltIn,
                speaker: Some("Vivian".to_string()),
                saved_voice_id: None,
                speed: Some(1.1),
                segments: vec![
                    NewStudioProjectSegment {
                        position: 0,
                        text: "Hello world.".to_string(),
                        model_id: Some("Qwen3-TTS".to_string()),
                        voice_mode: Some(StudioProjectVoiceMode::BuiltIn),
                        speaker: Some("Vivian".to_string()),
                        saved_voice_id: None,
                    },
                    NewStudioProjectSegment {
                        position: 1,
                        text: "Another sentence.".to_string(),
                        model_id: Some("Qwen3-TTS".to_string()),
                        voice_mode: Some(StudioProjectVoiceMode::BuiltIn),
                        speaker: Some("Vivian".to_string()),
                        saved_voice_id: None,
                    },
                ],
            })
            .await
            .expect("project should be created");

        assert_eq!(created.segments.len(), 2);
        assert_eq!(created.speaker.as_deref(), Some("Vivian"));

        let updated = store
            .update_project(
                created.id.clone(),
                UpdateStudioProjectRecord {
                    voice_mode: Some(StudioProjectVoiceMode::Saved),
                    speaker: Some(None),
                    saved_voice_id: Some(Some("voice-1".to_string())),
                    ..UpdateStudioProjectRecord::default()
                },
            )
            .await
            .expect("update should succeed")
            .expect("project should exist");

        assert_eq!(updated.voice_mode, StudioProjectVoiceMode::Saved);
        assert_eq!(updated.saved_voice_id.as_deref(), Some("voice-1"));
        assert_eq!(updated.speaker, None);

        let segment = updated.segments.first().expect("segment should exist");
        let refreshed = store
            .update_segment_text(
                updated.id.clone(),
                segment.id.clone(),
                "Updated line.".to_string(),
            )
            .await
            .expect("segment update should succeed")
            .expect("project should exist");

        assert_eq!(refreshed.segments[0].text, "Updated line.");
        assert_eq!(refreshed.segments[0].speech_record_id, None);

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn split_and_delete_segment_reorders_project_blocks() {
        let root = test_env_root("studio-project-segment-ops");
        let db_path = root.join("test.sqlite3");
        let media_dir = root.join("media");
        std::fs::create_dir_all(&root).expect("root should be created");

        let store =
            StudioProjectStore::initialize_at(db_path, media_dir).expect("store should initialize");
        let conn = storage_layout::open_sqlite_connection(&store.db_path)
            .expect("speech schema connection should open");
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS speech_history_records (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                route_kind TEXT NOT NULL,
                model_id TEXT NULL,
                speaker TEXT NULL,
                language TEXT NULL,
                saved_voice_id TEXT NULL,
                speed REAL NULL,
                input_text TEXT NOT NULL,
                voice_description TEXT NULL,
                reference_text TEXT NULL,
                generation_time_ms REAL NULL,
                audio_duration_secs REAL NULL,
                rtf REAL NULL,
                tokens_generated INTEGER NULL,
                audio_mime_type TEXT NOT NULL,
                audio_filename TEXT NULL,
                audio_storage_path TEXT NOT NULL
            );
            "#,
        )
        .expect("speech schema should initialize");

        let created = store
            .create_project(NewStudioProjectRecord {
                name: "Narration project".to_string(),
                source_filename: Some("script.txt".to_string()),
                source_text: "Hello world. Another sentence.".to_string(),
                model_id: Some("Qwen3-TTS".to_string()),
                voice_mode: StudioProjectVoiceMode::BuiltIn,
                speaker: Some("Vivian".to_string()),
                saved_voice_id: None,
                speed: Some(1.0),
                segments: vec![
                    NewStudioProjectSegment {
                        position: 0,
                        text: "Hello world. Another sentence.".to_string(),
                        model_id: Some("Qwen3-TTS".to_string()),
                        voice_mode: Some(StudioProjectVoiceMode::BuiltIn),
                        speaker: Some("Vivian".to_string()),
                        saved_voice_id: None,
                    },
                    NewStudioProjectSegment {
                        position: 1,
                        text: "Closing line.".to_string(),
                        model_id: Some("Qwen3-TTS".to_string()),
                        voice_mode: Some(StudioProjectVoiceMode::BuiltIn),
                        speaker: Some("Vivian".to_string()),
                        saved_voice_id: None,
                    },
                ],
            })
            .await
            .expect("project should be created");

        let first_segment = created.segments.first().expect("segment should exist");
        let split = store
            .split_segment(
                created.id.clone(),
                first_segment.id.clone(),
                "Hello world.".to_string(),
                "Another sentence.".to_string(),
            )
            .await
            .expect("split should succeed")
            .expect("project should exist");

        assert_eq!(split.segments.len(), 3);
        assert_eq!(split.segments[0].text, "Hello world.");
        assert_eq!(split.segments[1].text, "Another sentence.");
        assert_eq!(split.segments[2].text, "Closing line.");
        assert_eq!(
            split.source_text,
            "Hello world.\n\nAnother sentence.\n\nClosing line."
        );

        let deleted = store
            .delete_segment(split.id.clone(), split.segments[1].id.clone())
            .await
            .expect("delete should succeed")
            .expect("project should exist");

        assert_eq!(deleted.segments.len(), 2);
        assert_eq!(deleted.segments[0].position, 0);
        assert_eq!(deleted.segments[1].position, 1);
        assert_eq!(deleted.segments[0].text, "Hello world.");
        assert_eq!(deleted.segments[1].text, "Closing line.");
        assert_eq!(deleted.source_text, "Hello world.\n\nClosing line.");

        let _ = std::fs::remove_dir_all(root);
    }
}
