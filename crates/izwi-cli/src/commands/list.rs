use crate::error::{CliError, Result};
use crate::OutputFormat;
use comfy_table::Table;
use console::style;

pub async fn execute(
    local: bool,
    detailed: bool,
    server: &str,
    format: OutputFormat,
) -> Result<()> {
    let client = reqwest::Client::new();
    let response = client
        .get(format!("{}/v1/models", server))
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
        OutputFormat::Plain => {
            if let Some(data) = models.get("data").and_then(|d| d.as_array()) {
                for model in data {
                    if let Some(id) = model.get("id").and_then(|v| v.as_str()) {
                        println!("{}", id);
                    }
                }
            }
        }
        _ => {
            print_models_table(&models, local, detailed);
        }
    }

    Ok(())
}

fn print_models_table(models: &serde_json::Value, _local: bool, detailed: bool) {
    let mut table = Table::new();
    
    if detailed {
        table.set_header(vec!["Model ID", "Type", "Size", "Status"]);
    } else {
        table.set_header(vec!["Model ID", "Status"]);
    }

    if let Some(data) = models.get("data").and_then(|d| d.as_array()) {
        for model in data {
            let id = model.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
            let status = model.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");
            
            let status_colored = match status {
                "ready" => style(status).green().to_string(),
                "downloading" => style(status).yellow().to_string(),
                "loading" => style(status).blue().to_string(),
                "error" => style(status).red().to_string(),
                _ => style(status).dim().to_string(),
            };

            if detailed {
                let model_type = model.get("type").and_then(|v| v.as_str()).unwrap_or("-");
                let size = model.get("size").and_then(|v| v.as_u64())
                    .map(|s| humansize::format_size(s, humansize::BINARY))
                    .unwrap_or_else(|| "-".to_string());
                
                table.add_row(vec![
                    style(id).cyan().to_string(),
                    model_type.to_string(),
                    size,
                    status_colored,
                ]);
            } else {
                table.add_row(vec![
                    style(id).cyan().to_string(),
                    status_colored,
                ]);
            }
        }
    }

    println!("{}", table);
    println!();
    println!("{}: Use {} to download more models", 
        style("Tip").yellow().bold(),
        style("izwi pull <model>").cyan()
    );
}
