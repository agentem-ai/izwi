use crate::error::{CliError, Result};
use crate::http;
use crate::style::Theme;
use crate::{ModelCommands, OutputFormat};
use comfy_table::Table;
use console::style;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
struct AdminModelsResponse {
    models: Vec<AdminModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdminModelInfo {
    variant: String,
    status: String,
    local_path: Option<PathBuf>,
    size_bytes: Option<u64>,
    download_progress: Option<f32>,
    error_message: Option<String>,
}

pub async fn execute(
    command: ModelCommands,
    server: &str,
    format: OutputFormat,
    quiet: bool,
) -> Result<()> {
    match command {
        ModelCommands::List { local, detailed } => {
            list_models(server, local, detailed, format, quiet).await
        }
        ModelCommands::Info { model, json } => show_model_info(server, &model, json, format).await,
        ModelCommands::Load { model, wait } => load_model(server, &model, wait).await,
        ModelCommands::Unload { model, yes } => unload_model(server, &model, yes).await,
        ModelCommands::Progress { model } => show_download_progress(server, model.as_deref()).await,
    }
}

async fn list_models(
    server: &str,
    local: bool,
    detailed: bool,
    format: OutputFormat,
    quiet: bool,
) -> Result<()> {
    if !quiet {
        println!("{}", style("Fetching models...").dim());
    }

    let client = http::client(Some(std::time::Duration::from_secs(30)))?;
    let response = client
        .get(format!("{}/v1/admin/models", server))
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

    let payload: AdminModelsResponse = response.json().await?;
    let mut models = payload.models;
    if local {
        models.retain(|m| m.status != "not_downloaded");
    }

    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&models)?);
        }
        OutputFormat::Yaml => {
            println!("{}", serde_json::to_string_pretty(&models)?);
        }
        OutputFormat::Plain => {
            for model in &models {
                println!("{}", model.variant);
            }
        }
        OutputFormat::Table => {
            print_models_table(&models, detailed);
        }
    }

    Ok(())
}

fn print_models_table(models: &[AdminModelInfo], detailed: bool) {
    let mut table = Table::new();
    if detailed {
        table.set_header(vec!["Variant", "Status", "Size", "Progress", "Local Path"]);
    } else {
        table.set_header(vec!["Variant", "Status", "Size"]);
    }

    for model in models {
        let size = model
            .size_bytes
            .map(|s| humansize::format_size(s, humansize::BINARY))
            .unwrap_or_else(|| "-".to_string());

        if detailed {
            let progress = model
                .download_progress
                .map(|p| format!("{:.1}%", p))
                .unwrap_or_else(|| "-".to_string());
            let local_path = model
                .local_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "-".to_string());
            table.add_row(vec![
                style(&model.variant).cyan().to_string(),
                status_color(&model.status),
                size,
                progress,
                local_path,
            ]);
        } else {
            table.add_row(vec![
                style(&model.variant).cyan().to_string(),
                status_color(&model.status),
                size,
            ]);
        }
    }

    println!("{}", table);
}

fn status_color(status: &str) -> String {
    match status {
        "ready" => style(status).green().to_string(),
        "downloaded" => style(status).green().to_string(),
        "downloading" => style(status).yellow().to_string(),
        "loading" => style(status).blue().to_string(),
        "not_downloaded" => style(status).dim().to_string(),
        "error" => style(status).red().to_string(),
        _ => style(status).dim().to_string(),
    }
}

async fn show_model_info(
    server: &str,
    model: &str,
    json: bool,
    format: OutputFormat,
) -> Result<()> {
    let client = http::client(Some(std::time::Duration::from_secs(30)))?;
    let response = client
        .get(format!("{}/v1/admin/models/{}", server, model))
        .send()
        .await
        .map_err(|e| CliError::ConnectionError(e.to_string()))?;

    if response.status().as_u16() == 404 {
        return Err(CliError::ModelNotFound(model.to_string()));
    }

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        return Err(CliError::ApiError {
            status,
            message: text,
        });
    }

    let info: AdminModelInfo = response.json().await?;

    if json || matches!(format, OutputFormat::Json) {
        println!("{}", serde_json::to_string_pretty(&info)?);
    } else {
        println!("{}: {}", style("Variant").bold(), info.variant);
        println!("{}: {}", style("Status").bold(), status_color(&info.status));
        println!(
            "{}: {}",
            style("Size").bold(),
            info.size_bytes
                .map(|s| humansize::format_size(s, humansize::BINARY))
                .unwrap_or_else(|| "-".to_string())
        );
        println!(
            "{}: {}",
            style("Progress").bold(),
            info.download_progress
                .map(|p| format!("{:.1}%", p))
                .unwrap_or_else(|| "-".to_string())
        );
        println!(
            "{}: {}",
            style("Local Path").bold(),
            info.local_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "-".to_string())
        );
        if let Some(err) = info.error_message {
            println!("{}: {}", style("Error").bold(), style(err).red());
        }
    }

    Ok(())
}

async fn load_model(server: &str, model: &str, wait: bool) -> Result<()> {
    let theme = Theme::default();
    theme.step(1, 3, "Loading model...");

    let client = http::client(Some(std::time::Duration::from_secs(60)))?;
    let response = client
        .post(format!("{}/v1/admin/models/{}/load", server, model))
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

    theme.success(&format!("Load requested for '{}'", model));

    if wait {
        theme.step(2, 3, "Waiting for model readiness...");
        for _ in 0..60 {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

            let resp = client
                .get(format!("{}/v1/admin/models/{}", server, model))
                .send()
                .await
                .map_err(|e| CliError::ConnectionError(e.to_string()))?;

            if !resp.status().is_success() {
                continue;
            }

            let info: AdminModelInfo = resp.json().await?;
            if info.status == "ready" {
                theme.step(3, 3, "Model is ready");
                return Ok(());
            }
            if info.status == "error" {
                return Err(CliError::Other(
                    info.error_message
                        .unwrap_or_else(|| "Model entered error state".to_string()),
                ));
            }
        }

        return Err(CliError::Other(
            "Timed out waiting for model to become ready".to_string(),
        ));
    }

    theme.step(3, 3, "Done");
    Ok(())
}

async fn unload_model(server: &str, model: &str, yes: bool) -> Result<()> {
    let theme = Theme::default();

    if !yes {
        println!(
            "This will unload model '{}' from memory.",
            style(model).cyan()
        );
        let confirm = dialoguer::Confirm::new()
            .with_prompt("Are you sure?")
            .default(false)
            .interact()
            .map_err(|e| CliError::Other(e.to_string()))?;

        if !confirm {
            println!("Cancelled.");
            return Ok(());
        }
    }

    let client = http::client(Some(std::time::Duration::from_secs(30)))?;
    let response = client
        .post(format!("{}/v1/admin/models/{}/unload", server, model))
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

    theme.success(&format!("Model '{}' unloaded successfully", model));
    Ok(())
}

async fn show_download_progress(server: &str, model: Option<&str>) -> Result<()> {
    let Some(model) = model else {
        println!("Please provide a model variant: izwi models progress <model>");
        return Ok(());
    };

    let client = http::client(Some(std::time::Duration::from_secs(30)))?;
    let response = client
        .get(format!("{}/v1/admin/models/{}", server, model))
        .send()
        .await
        .map_err(|e| CliError::ConnectionError(e.to_string()))?;

    if response.status().as_u16() == 404 {
        return Err(CliError::ModelNotFound(model.to_string()));
    }
    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        return Err(CliError::ApiError {
            status,
            message: text,
        });
    }

    let info: AdminModelInfo = response.json().await?;
    println!("{}: {}", style("Variant").bold(), info.variant);
    println!("{}: {}", style("Status").bold(), status_color(&info.status));
    println!(
        "{}: {}",
        style("Progress").bold(),
        info.download_progress
            .map(|p| format!("{:.1}%", p))
            .unwrap_or_else(|| "-".to_string())
    );

    Ok(())
}
