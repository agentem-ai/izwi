use crate::error::{CliError, Result};
use crate::style::Theme;
use crate::AudioFormat;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{Read, Write};
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
    // Read text from stdin if "-"
    let text = if args.text == "-" {
        let mut buffer = String::new();
        std::io::stdin()
            .read_to_string(&mut buffer)
            .map_err(|e| CliError::Io(e))?;
        buffer
    } else {
        args.text
    };

    if text.trim().is_empty() {
        return Err(CliError::InvalidInput("Text cannot be empty".to_string()));
    }

    theme.step(1, 2, &format!("Generating speech with '{}'...", args.model));

    let format_str = match args.format {
        AudioFormat::Wav => "wav",
        AudioFormat::Mp3 => "mp3",
        AudioFormat::Ogg => "ogg",
        AudioFormat::Flac => "flac",
        AudioFormat::Aac => "aac",
    };

    let request_body = serde_json::json!({
        "model": args.model,
        "input": text,
        "voice": args.speaker,
        "speed": args.speed,
        "temperature": args.temperature,
        "response_format": format_str,
    });

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| CliError::Other(e.to_string()))?;

    let start_time = std::time::Instant::now();

    if args.stream {
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

        handle_output(audio_data, args.output, args.play, theme).await?;
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

        handle_output(audio_data, args.output, args.play, theme).await?;
    }

    let duration = start_time.elapsed();
    theme.step(2, 2, &format!("Done in {:.2}s", duration.as_secs_f64()));

    Ok(())
}

async fn handle_output(
    audio_data: bytes::Bytes,
    output: Option<PathBuf>,
    _play: bool,
    theme: &Theme,
) -> Result<()> {
    let output_path = match output {
        Some(path) => path,
        None => {
            // Generate default filename
            let timestamp = chrono::Utc::now().timestamp();
            PathBuf::from(format!("izwi_output_{}.wav", timestamp))
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
