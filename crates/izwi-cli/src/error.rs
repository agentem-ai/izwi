use console::style;
use std::fmt;
use std::io;

pub type Result<T> = std::result::Result<T, CliError>;

#[derive(Debug)]
pub enum CliError {
    /// API request failed
    ApiError { 
        status: u16, 
        message: String 
    },
    /// Server connection failed
    ConnectionError(String),
    /// Model not found
    ModelNotFound(String),
    /// Invalid input
    InvalidInput(String),
    /// Configuration error
    ConfigError(String),
    /// I/O error
    Io(io::Error),
    /// Serialization error
    Serialization(serde_json::Error),
    /// Other errors
    Other(String),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::ApiError { status, message } => {
                write!(f, "API error ({}): {}", style(status).red().bold(), message)
            }
            CliError::ConnectionError(msg) => {
                write!(f, "Connection error: {}", msg)
            }
            CliError::ModelNotFound(model) => {
                write!(f, "Model not found: {}", style(model).yellow())
            }
            CliError::InvalidInput(msg) => {
                write!(f, "Invalid input: {}", msg)
            }
            CliError::ConfigError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
            CliError::Io(e) => {
                write!(f, "I/O error: {}", e)
            }
            CliError::Serialization(e) => {
                write!(f, "Serialization error: {}", e)
            }
            CliError::Other(msg) => {
                write!(f, "{}", msg)
            }
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CliError::Io(e) => Some(e),
            CliError::Serialization(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for CliError {
    fn from(e: io::Error) -> Self {
        CliError::Io(e)
    }
}

impl From<serde_json::Error> for CliError {
    fn from(e: serde_json::Error) -> Self {
        CliError::Serialization(e)
    }
}

impl From<reqwest::Error> for CliError {
    fn from(e: reqwest::Error) -> Self {
        if e.is_connect() {
            CliError::ConnectionError(e.to_string())
        } else {
            CliError::Other(e.to_string())
        }
    }
}

impl From<anyhow::Error> for CliError {
    fn from(e: anyhow::Error) -> Self {
        CliError::Other(e.to_string())
    }
}
