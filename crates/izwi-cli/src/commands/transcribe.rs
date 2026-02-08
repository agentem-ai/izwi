use crate::error::{CliError, Result};
use crate::TranscriptFormat;
use base64;
use std::path::PathBuf;

pub struct TranscribeArgs {
    pub file: PathBuf,
    pub model: String,
    pub language: Option<String>,
    pub format: TranscriptFormat,
    pub output: Option<PathBuf>,
    pub word_timestamps: bool,
}

pub async fn execute(args: TranscribeArgs, server: &str) -> Result<()> {
    // Verify file exists
    if !args.file.exists() {
        return Err(CliError::InvalidInput(format!(
            "File not found: {}",
            args.file.display()
        )));
    }

    // Read audio file
    let audio_data = tokio::fs::read(&args.file)
        .await
        .map_err(|e| CliError::Io(e))?;
    let audio_base64 = base64::encode(&audio_data);

    let format_str = match args.format {
        TranscriptFormat::Text => "text",
        TranscriptFormat::Json => "json",
        TranscriptFormat::Srt => "srt",
        TranscriptFormat::Vtt => "vtt",
    };

    let request_body = serde_json::json!({
        "model": args.model,
        "file": format!("data:audio/wav;base64,{}", audio_base64),
        "response_format": format_str,
    });

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/v1/audio/transcriptions", server))
        .json(&request_body)
        .send()
        .await
        .map_err(|e| CliError::ConnectionError(e.to_string()))?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        return Err(CliError::ApiError {
            status,
            message: text,
        });
    }

    let result = response
        .text()
        .await
        .map_err(|e| CliError::Other(e.to_string()))?;

    // Output result
    if let Some(output_path) = args.output {
        tokio::fs::write(&output_path, result)
            .await
            .map_err(|e| CliError::Io(e))?;
        println!("Transcription saved to: {}", output_path.display());
    } else {
        println!("{}", result);
    }

    Ok(())
}
