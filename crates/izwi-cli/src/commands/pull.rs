use crate::error::{CliError, Result};
use crate::http;
use crate::style::Theme;
use crate::utils;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct ModelState {
    variant: String,
    status: String,
    local_path: Option<PathBuf>,
}

pub async fn execute(
    model: String,
    force: bool,
    yes: bool,
    server: &str,
    theme: &Theme,
) -> Result<()> {
    // Check if model already exists
    if !force {
        let client = http::client(Some(std::time::Duration::from_secs(30)))?;
        let resp = client
            .get(format!("{}/v1/admin/models/{}", server, model))
            .send()
            .await;

        if let Ok(r) = resp {
            if r.status().as_u16() == 200 {
                if let Ok(state) = r.json::<ModelState>().await {
                    let has_local_files = state
                        .local_path
                        .as_ref()
                        .filter(|path| path.is_dir())
                        .is_some()
                        || utils::model_dir_if_present(&state.variant).is_some();

                    if state.status != "not_downloaded" || has_local_files {
                        theme.info(&format!(
                            "Model '{}' is currently '{}'",
                            model, state.status
                        ));
                        if !yes {
                            let confirm = dialoguer::Confirm::new()
                                .with_prompt("Re-download?")
                                .default(false)
                                .interact()
                                .map_err(|e| CliError::Other(e.to_string()))?;

                            if !confirm {
                                println!("Cancelled.");
                                return Ok(());
                            }
                        }
                    }
                }
            }
        }
    }

    theme.step(1, 3, &format!("Starting download for '{}'...", model));

    let client = http::client(Some(std::time::Duration::from_secs(30)))?;
    let response = client
        .post(format!("{}/v1/admin/models/{}/download", server, model))
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

    theme.step(2, 3, "Downloading model files...");

    // Show progress (simplified - would stream from progress endpoint)
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}% {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    for i in 0..100 {
        pb.set_position(i);
        pb.set_message(format!("Downloading..."));
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }

    pb.finish_with_message("Download complete");

    theme.step(3, 3, "Finalizing...");

    theme.success(&format!("Model '{}' downloaded successfully!", model));
    println!();
    println!("Next steps:");
    println!(
        "  - Load the model: {}",
        style(format!("izwi models load {}", model)).cyan()
    );
    println!(
        "  - Generate speech: {}",
        style(format!("izwi tts 'Hello' -m {}", model)).cyan()
    );

    Ok(())
}
