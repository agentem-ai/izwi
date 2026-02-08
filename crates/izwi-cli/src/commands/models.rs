use crate::error::{CliError, Result};
use crate::style::Theme;
use crate::{ModelCommands, OutputFormat};
use comfy_table::{Table, Column};
use console::style;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct ModelInfo {
    id: String,
    size: String,
    status: String,
    description: String,
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
        ModelCommands::Info { model, json } => {
            show_model_info(server, &model, json, format).await
        }
        ModelCommands::Load { model, wait } => {
            load_model(server, &model, wait).await
        }
        ModelCommands::Unload { model, yes } => {
            unload_model(server, &model, yes).await
        }
        ModelCommands::Progress { model } => {
            show_download_progress(server, model.as_deref()).await
        }
    }
}

async fn list_models(
    server: &str,
    _local: bool,
    detailed: bool,
    format: OutputFormat,
    quiet: bool,
) -> Result<()> {
    if !quiet {
        println!("{}", style("Fetching models...").dim());
    }

    let client = reqwest::Client::new();
    let response = client
        .get(format!("{}/v1/admin/models", server))
        .send()
        .await
        .map_err(|e| CliError::ConnectionError(e.to_string()))?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        return Err(CliError::ApiError { status, message: text });
    }

    let models: serde_json::Value = response.json().await?;

    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&models)?);
        }
        OutputFormat::Yaml => {
            // Would need yaml dependency
            println!("{}", serde_json::to_string_pretty(&models)?);
        }
        _ => {
            print_models_table(&models, detailed);
        }
    }

    Ok(())
}

fn print_models_table(models: &serde_json::Value, detailed: bool) {
    let mut table = Table::new();
    table.set_header(vec!["Model", "Status", "Size"]);

    if let Some(data) = models.get("data").and_then(|d| d.as_array()) {
        for model in data {
            let id = model.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
            let status = model.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");
            let size = model.get("size").and_then(|v| v.as_u64())
                .map(|s| humansize::format_size(s, humansize::BINARY))
                .unwrap_or_else(|| "-".to_string());

            table.add_row(vec![
                style(id).cyan().to_string(),
                status_color(status),
                size,
            ]);
        }
    }

    println!("{}", table);

    if detailed {
        println!("\n{}: Use '{}' to download a model", 
            style("Tip").yellow().bold(),
            style("izwi pull <model>").cyan()
        );
    }
}

fn status_color(status: &str) -> String {
    match status {
        "ready" => style(status).green().to_string(),
        "downloading" => style(status).yellow().to_string(),
        "loading" => style(status).blue().to_string(),
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
    let client = reqwest::Client::new();
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
        return Err(CliError::ApiError { status, message: text });
    }

    let info: serde_json::Value = response.json().await?;

    if json || matches!(format, OutputFormat::Json) {
        println!("{}", serde_json::to_string_pretty(&info)?);
    } else {
        println!("{}: {}", style("Model").bold(), model);
        if let Some(desc) = info.get("description").and_then(|v| v.as_str()) {
            println!("{}: {}", style("Description").bold(), desc);
        }
        if let Some(status) = info.get("status").and_then(|v| v.as_str()) {
            println!("{}: {}", style("Status").bold(), status_color(status));
        }
    }

    Ok(())
}

async fn load_model(server: &str, model: &str, wait: bool) -> Result<()> {
    let theme = Theme::default();
    theme.step(1, 3, "Loading model...");

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/v1/admin/models/{}/load", server, model))
        .send()
        .await
        .map_err(|e| CliError::ConnectionError(e.to_string()))?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        return Err(CliError::ApiError { status, message: text });
    }

    theme.success(&format!("Model '{}' loaded successfully", model));

    if wait {
        println!("Waiting for model to be ready...");
        // Poll until ready
        for _ in 0..60 {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            
            let resp = client
                .get(format!("{}/v1/admin/models/{}", server, model))
                .send()
                .await?;
            
            if let Ok(info) = resp.json::<serde_json::Value>().await {
                if let Some(status) = info.get("status").and_then(|v| v.as_str()) {
                    if status == "ready" {
                        theme.success("Model is ready!");
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}

async fn unload_model(server: &str, model: &str, yes: bool) -> Result<()> {
    let theme = Theme::default();

    if !yes {
        println!("This will unload model '{}' from memory.", style(model).cyan());
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

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/v1/admin/models/{}/unload", server, model))
        .send()
        .await
        .map_err(|e| CliError::ConnectionError(e.to_string()))?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        return Err(CliError::ApiError { status, message: text });
    }

    theme.success(&format!("Model '{}' unloaded successfully", model));
    Ok(())
}

async fn show_download_progress(server: &str, model: Option<&str>) -> Result<()> {
    let theme = Theme::default();
    
    if let Some(m) = model {
        theme.info(&format!("Watching download progress for '{}'...", m));
        // Would stream progress updates here
    } else {
        theme.info("Showing all download progress...");
    }
    
    Ok(())
}
