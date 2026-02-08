use crate::error::{CliError, Result};
use crate::http;
use crate::style::Theme;
use crate::AudioFormat;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::Read;
use std::path::PathBuf;

pub struct TtsArgs {
    pub text: String,
    pub model: String,
    pub speaker: String,
    pub output: Option<PathBuf>,
    pub format: AudioFormat,
    pub speed: f32,
    pub temperature: f32,
    pub stream: bool,
    pub play: bool,
}

pub async fn execute(args: TtsArgs, server: &str, theme: &Theme) -> Result<()> {
    let TtsArgs {
        text,
        model,
        speaker,
        output,
        format,
        speed,
        temperature,
        stream,
        play,
    } = args;

    // Read text from stdin if "-"
    let text = if text == "-" {
        let mut buffer = String::new();
        std::io::stdin()
            .read_to_string(&mut buffer)
            .map_err(|e| CliError::Io(e))?;
        buffer
    } else {
        text
    };

    if text.trim().is_empty() {
        return Err(CliError::InvalidInput("Text cannot be empty".to_string()));
    }

    theme.step(1, 2, &format!("Generating speech with '{}'...", model));

    let format_str = match format {
        AudioFormat::Wav => "wav",
        AudioFormat::Mp3 => "mp3",
        AudioFormat::Ogg => "ogg",
        AudioFormat::Flac => "flac",
        AudioFormat::Aac => "aac",
    };

    let request_body = serde_json::json!({
        "model": model,
        "input": text,
        "voice": speaker,
        "speed": speed,
        "temperature": temperature,
        "response_format": format_str,
        "stream": stream,
    });

    let client = http::client(Some(std::time::Duration::from_secs(300)))?;

    let start_time = std::time::Instant::now();

    if stream {
        // Streaming mode
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        pb.set_message("Generating audio chunks...");

        let response = client
            .post(format!("{}/v1/audio/speech", server))
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

        pb.finish_with_message("Generation complete");

        let audio_data = response
            .bytes()
            .await
            .map_err(|e| CliError::Other(e.to_string()))?;

        handle_output(audio_data, output.clone(), format.clone(), play, theme).await?;
    } else {
        // Non-streaming mode
        let pb = ProgressBar::new(100);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap(),
        );

        pb.set_message("Sending request...");

        let response = client
            .post(format!("{}/v1/audio/speech", server))
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

        pb.set_message("Receiving audio...");

        let audio_data = response
            .bytes()
            .await
            .map_err(|e| CliError::Other(e.to_string()))?;

        pb.finish_with_message("Complete");

        handle_output(audio_data, output, format, play, theme).await?;
    }

    let duration = start_time.elapsed();
    theme.step(2, 2, &format!("Done in {:.2}s", duration.as_secs_f64()));

    Ok(())
}

async fn handle_output(
    audio_data: bytes::Bytes,
    output: Option<PathBuf>,
    format: AudioFormat,
    _play: bool,
    theme: &Theme,
) -> Result<()> {
    let output_path = match output {
        Some(path) => path,
        None => {
            // Generate default filename
            let timestamp = chrono::Utc::now().timestamp();
            let extension = match format {
                AudioFormat::Wav => "wav",
                AudioFormat::Mp3 => "mp3",
                AudioFormat::Ogg => "ogg",
                AudioFormat::Flac => "flac",
                AudioFormat::Aac => "aac",
            };
            PathBuf::from(format!("izwi_output_{}.{}", timestamp, extension))
        }
    };

    let mut file = tokio::fs::File::create(&output_path)
        .await
        .map_err(|e| CliError::Io(e))?;

    tokio::io::AsyncWriteExt::write_all(&mut file, &audio_data)
        .await
        .map_err(|e| CliError::Io(e))?;

    theme.success(&format!("Audio saved to: {}", output_path.display()));

    if _play {
        theme.info("Playing audio... (not implemented in this version)");
        // Would use rodio or external player here
    }

    Ok(())
}
