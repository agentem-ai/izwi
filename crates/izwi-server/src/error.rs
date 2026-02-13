//! API error handling

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// API error type
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: msg.into(),
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(json!({
            "error": {
                "message": self.message,
                "type": match self.status {
                    StatusCode::BAD_REQUEST => "invalid_request_error",
                    StatusCode::NOT_FOUND => "not_found_error",
                    _ => "server_error",
                },
                "param": null,
                "code": self.status.as_str()
            }
        }));
        (self.status, body).into_response()
    }
}

impl From<izwi_core::Error> for ApiError {
    fn from(err: izwi_core::Error) -> Self {
        match &err {
            izwi_core::Error::ModelNotFound(_) => ApiError::not_found(err.to_string()),
            izwi_core::Error::ConfigError(_) => ApiError::bad_request(err.to_string()),
            _ => ApiError::internal(err.to_string()),
        }
    }
}
