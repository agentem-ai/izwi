use axum::extract::Query;
use base64::Engine;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::error::ApiError;

const DEFAULT_MAX_LIMIT: usize = 2000;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct CursorPaginationQuery {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub cursor: Option<String>,
}

impl CursorPaginationQuery {
    pub fn resolved_limit(&self, default_limit: usize, max_limit: usize) -> usize {
        let max_limit = max_limit.max(1).min(DEFAULT_MAX_LIMIT.max(1));
        self.limit
            .unwrap_or(default_limit.max(1))
            .clamp(1, max_limit)
    }

    pub fn decode_cursor<T: DeserializeOwned>(&self) -> Result<Option<T>, ApiError> {
        match self.cursor.as_deref() {
            Some(cursor) if !cursor.trim().is_empty() => decode_cursor(cursor).map(Some),
            _ => Ok(None),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CursorPagination {
    pub next_cursor: Option<String>,
    pub has_more: bool,
    pub limit: usize,
}

pub fn encode_cursor<T: Serialize>(value: &T) -> String {
    let json = serde_json::to_vec(value).unwrap_or_default();
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json)
}

pub fn decode_cursor<T: DeserializeOwned>(cursor: &str) -> Result<T, ApiError> {
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(cursor.as_bytes())
        .map_err(|_| ApiError::bad_request("Invalid pagination cursor"))?;
    serde_json::from_slice::<T>(bytes.as_slice())
        .map_err(|_| ApiError::bad_request("Invalid pagination cursor"))
}

pub fn resolve_cursor_query(Query(query): Query<CursorPaginationQuery>) -> CursorPaginationQuery {
    query
}
