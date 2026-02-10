use anyhow::{Context, Result};
use clap::Parser;
use tauri::{WebviewUrl, WebviewWindowBuilder};
use url::Url;

#[derive(Debug, Parser)]
#[command(
    name = "izwi-desktop",
    about = "Tauri desktop shell for Izwi local inference",
    version
)]
struct DesktopArgs {
    /// Base URL of the Izwi local server
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    server_url: String,

    /// Desktop window title
    #[arg(long, default_value = "Izwi")]
    window_title: String,

    /// Initial window width
    #[arg(long, default_value = "1360")]
    width: f64,

    /// Initial window height
    #[arg(long, default_value = "900")]
    height: f64,
}

fn main() -> Result<()> {
    let args = DesktopArgs::parse();
    let server_url = Url::parse(&args.server_url)
        .with_context(|| format!("invalid --server-url value: {}", args.server_url))?;
    let window_title = args.window_title.clone();
    let width = args.width;
    let height = args.height;

    tauri::Builder::default()
        .setup(move |app| {
            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(server_url.clone()))
                .title(window_title.as_str())
                .inner_size(width, height)
                .min_inner_size(960.0, 680.0)
                .resizable(true)
                .build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .map_err(|e| anyhow::anyhow!("failed to run desktop app: {}", e))?;

    Ok(())
}
