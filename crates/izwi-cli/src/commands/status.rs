use crate::error::Result;
use crate::http;
use crate::style::Theme;
use comfy_table::Table;
use console::style;

pub async fn execute(
    detailed: bool,
    watch: Option<u64>,
    server: &str,
    theme: &Theme,
) -> Result<()> {
    if let Some(interval) = watch {
        // Watch mode
        theme.info(&format!(
            "Watching status every {} seconds (Ctrl+C to stop)...",
            interval
        ));

        loop {
            print!("\x1B[2J\x1B[1;1H"); // Clear screen
            show_status(server, detailed).await?;
            tokio::time::sleep(tokio::time::Duration::from_secs(interval)).await;
        }
    } else {
        show_status(server, detailed).await
    }
}

async fn show_status(server: &str, detailed: bool) -> Result<()> {
    let client = http::client(Some(std::time::Duration::from_secs(10)))?;

    // Health check
    let health_resp = client.get(format!("{}/v1/health", server)).send().await;

    match health_resp {
        Ok(resp) => {
            if resp.status().is_success() {
                println!(
                    "{}  Server: {}",
                    style("●").green(),
                    style("Healthy").green()
                );
            } else {
                println!(
                    "{}  Server: {} (Status: {})",
                    style("●").red(),
                    style("Unhealthy").red(),
                    resp.status()
                );
            }
        }
        Err(e) => {
            println!(
                "{}  Server: {} - {}",
                style("●").red(),
                style("Unreachable").red(),
                e
            );
            return Ok(());
        }
    }

    // Get models status
    let models_resp = client
        .get(format!("{}/v1/admin/models", server))
        .send()
        .await;

    if let Ok(resp) = models_resp {
        if let Ok(data) = resp.json::<serde_json::Value>().await {
            let models = data.get("models").and_then(|d| d.as_array());

            if let Some(models) = models {
                println!("\n{}", style("Models:").bold());

                let mut table = Table::new();
                table.set_header(vec!["Model", "Status", "Size"]);

                for model in models {
                    let id = model
                        .get("variant")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let status = model
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let size = model
                        .get("size_bytes")
                        .and_then(|v| v.as_u64())
                        .map(|s| humansize::format_size(s, humansize::BINARY))
                        .unwrap_or_else(|| "-".to_string());

                    let status_color = match status {
                        "ready" => style(status).green().to_string(),
                        "downloading" => style(status).yellow().to_string(),
                        "loading" => style(status).blue().to_string(),
                        "error" => style(status).red().to_string(),
                        _ => style(status).dim().to_string(),
                    };

                    table.add_row(vec![style(id).cyan().to_string(), status_color, size]);
                }

                println!("{}", table);
            }
        }
    }

    if detailed {
        println!("\n{}", style("Server Info:").bold());
        println!("  Endpoint: {}", server);
        println!("  Version:  {}", env!("CARGO_PKG_VERSION"));
        println!(
            "  Platform: {}-{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        );
    }

    Ok(())
}
