use crate::error::{CliError, Result};
use std::time::Duration;

pub fn client(timeout: Option<Duration>) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .no_proxy()
        .user_agent(format!("izwi-cli/{}", env!("CARGO_PKG_VERSION")));

    if let Some(timeout) = timeout {
        builder = builder.timeout(timeout);
    }

    builder
        .build()
        .map_err(|e| CliError::Other(format!("Failed to initialize HTTP client: {}", e)))
}
