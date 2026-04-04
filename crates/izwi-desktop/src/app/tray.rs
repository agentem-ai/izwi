use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::process::Child;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::TrayIconBuilder,
    AppHandle, Manager, Runtime,
};
use url::Url;

use super::server::{maybe_start_local_server, shutdown_child};

const TRAY_ID: &str = "izwi-tray";
const TRAY_OPEN_ID: &str = "tray_open";
const TRAY_SETTINGS_ID: &str = "tray_settings";
const TRAY_MODELS_ID: &str = "tray_models";
const TRAY_CHECK_UPDATES_ID: &str = "tray_check_updates";
const TRAY_SERVER_STATUS_ID: &str = "tray_server_status";
const TRAY_MODELS_STATUS_ID: &str = "tray_models_status";
const TRAY_RESTART_SERVER_ID: &str = "tray_restart_server";
const TRAY_QUIT_ID: &str = "tray_quit";

const STATUS_POLL_INTERVAL: Duration = Duration::from_secs(8);
const STATUS_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone)]
pub struct TrayConfig {
    pub server_url: Url,
    pub local_server_mode: bool,
    pub managed_server: Arc<Mutex<Option<Child>>>,
}

#[derive(Debug, Deserialize)]
struct TrayModelsResponse {
    models: Vec<TrayModelInfo>,
}

#[derive(Debug, Deserialize)]
struct TrayModelInfo {
    status: String,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct ModelStatusCounts {
    total: usize,
    ready: usize,
    loading: usize,
    errors: usize,
}

struct TrayStatusSnapshot {
    server_label: String,
    models_label: String,
}

pub fn build_basic_tray<R: Runtime>(app: &AppHandle<R>, config: TrayConfig) -> Result<()> {
    let open_item = MenuItem::with_id(app, TRAY_OPEN_ID, "Open Izwi", true, None::<&str>)
        .context("failed to build tray open item")?;
    let settings_item = MenuItem::with_id(app, TRAY_SETTINGS_ID, "Settings", true, None::<&str>)
        .context("failed to build tray settings item")?;
    let models_item = MenuItem::with_id(app, TRAY_MODELS_ID, "Models", true, None::<&str>)
        .context("failed to build tray models item")?;
    let check_updates_item = MenuItem::with_id(
        app,
        TRAY_CHECK_UPDATES_ID,
        "Check for Updates",
        true,
        None::<&str>,
    )
    .context("failed to build tray check updates item")?;
    let section_separator =
        PredefinedMenuItem::separator(app).context("failed to build tray separator item")?;
    let server_status_item = MenuItem::with_id(
        app,
        TRAY_SERVER_STATUS_ID,
        "Server: checking...",
        false,
        None::<&str>,
    )
    .context("failed to build tray server status item")?;
    let models_status_item = MenuItem::with_id(
        app,
        TRAY_MODELS_STATUS_ID,
        "Models: checking...",
        false,
        None::<&str>,
    )
    .context("failed to build tray models status item")?;
    let restart_server_label = if config.local_server_mode {
        "Restart Server"
    } else {
        "Restart Server (local only)"
    };
    let restart_server_item = MenuItem::with_id(
        app,
        TRAY_RESTART_SERVER_ID,
        restart_server_label,
        config.local_server_mode,
        None::<&str>,
    )
    .context("failed to build tray restart server item")?;
    let quit_separator = PredefinedMenuItem::separator(app)
        .context("failed to build tray quit separator item")?;
    let quit_item = MenuItem::with_id(app, TRAY_QUIT_ID, "Quit Izwi", true, None::<&str>)
        .context("failed to build tray quit item")?;
    let menu = Menu::with_items(
        app,
        &[
            &open_item,
            &settings_item,
            &models_item,
            &check_updates_item,
            &section_separator,
            &server_status_item,
            &models_status_item,
            &restart_server_item,
            &quit_separator,
            &quit_item,
        ],
    )
    .context("failed to build tray menu")?;

    let event_config = config.clone();
    let mut tray_builder = TrayIconBuilder::with_id(TRAY_ID)
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(move |app_handle, event| match event.id().0.as_str() {
            TRAY_OPEN_ID => {
                let _ = show_main_window(app_handle);
            }
            TRAY_SETTINGS_ID => {
                let _ = navigate_main_window(app_handle, "/settings");
            }
            TRAY_MODELS_ID => {
                let _ = navigate_main_window(app_handle, "/models");
            }
            TRAY_CHECK_UPDATES_ID => {
                let _ = show_main_window(app_handle);
                let _ = emit_tray_event(app_handle, "izwi:tray-check-updates", None);
            }
            TRAY_RESTART_SERVER_ID => {
                let _ = restart_local_server(app_handle, &event_config);
            }
            TRAY_QUIT_ID => {
                app_handle.exit(0);
            }
            _ => {}
        });

    if let Some(icon) = app.default_window_icon() {
        tray_builder = tray_builder.icon(icon.clone());
    }

    tray_builder
        .build(app)
        .context("failed to create tray icon")?;
    spawn_status_poller(config.server_url, server_status_item, models_status_item);

    Ok(())
}

pub fn show_main_window<R: Runtime>(app: &AppHandle<R>) -> Result<()> {
    let window = app
        .get_webview_window("main")
        .context("main window is unavailable")?;
    let _ = window.show();
    let _ = window.unminimize();
    let _ = window.set_focus();
    Ok(())
}

fn navigate_main_window<R: Runtime>(app: &AppHandle<R>, route: &str) -> Result<()> {
    show_main_window(app)?;
    let detail = json!({ "path": route }).to_string();
    emit_tray_event(app, "izwi:tray-route", Some(detail.as_str()))
}

fn emit_tray_event<R: Runtime>(
    app: &AppHandle<R>,
    event_name: &str,
    detail_json: Option<&str>,
) -> Result<()> {
    let window = app
        .get_webview_window("main")
        .context("main window is unavailable")?;

    let js_event_name = js_string_literal(event_name);
    let script = if let Some(detail_json) = detail_json {
        format!(
            "window.dispatchEvent(new CustomEvent({}, {{ detail: {} }}));",
            js_event_name, detail_json
        )
    } else {
        format!("window.dispatchEvent(new Event({}));", js_event_name)
    };

    window
        .eval(script)
        .context("failed to dispatch tray event to frontend")
}

fn js_string_literal(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r");
    format!("\"{}\"", escaped)
}

fn restart_local_server<R: Runtime>(app: &AppHandle<R>, config: &TrayConfig) -> Result<()> {
    if !config.local_server_mode {
        return Ok(());
    }

    let mut child_slot = config
        .managed_server
        .lock()
        .map_err(|_| anyhow::anyhow!("failed to acquire managed server lock"))?;
    if let Some(mut child) = child_slot.take() {
        shutdown_child(&mut child);
    }

    if let Some(next_child) = maybe_start_local_server(app, &config.server_url)? {
        *child_slot = Some(next_child);
    }

    Ok(())
}

fn spawn_status_poller<R: Runtime>(
    server_url: Url,
    server_status_item: MenuItem<R>,
    models_status_item: MenuItem<R>,
) {
    tauri::async_runtime::spawn(async move {
        let client = match Client::builder().timeout(STATUS_REQUEST_TIMEOUT).build() {
            Ok(client) => client,
            Err(_) => {
                let _ = server_status_item.set_text("Server: unavailable");
                let _ = models_status_item.set_text("Models: unavailable");
                return;
            }
        };

        loop {
            let snapshot = fetch_status_snapshot(&client, &server_url).await;
            let _ = server_status_item.set_text(snapshot.server_label);
            let _ = models_status_item.set_text(snapshot.models_label);
            tokio::time::sleep(STATUS_POLL_INTERVAL).await;
        }
    });
}

async fn fetch_status_snapshot(client: &Client, server_url: &Url) -> TrayStatusSnapshot {
    let Some(health_url) = api_url(server_url, "v1/internal/health") else {
        return TrayStatusSnapshot {
            server_label: "Server: unavailable".to_string(),
            models_label: "Models: unavailable".to_string(),
        };
    };

    let server_online = match client.get(health_url).send().await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    };

    if !server_online {
        return TrayStatusSnapshot {
            server_label: "Server: offline".to_string(),
            models_label: "Models: unavailable".to_string(),
        };
    }

    let models_label = match api_url(server_url, "v1/admin/models") {
        Some(models_url) => match client.get(models_url).send().await {
            Ok(response) if response.status().is_success() => {
                match response.json::<TrayModelsResponse>().await {
                    Ok(models_response) => {
                        format_models_label(model_status_counts(&models_response.models))
                    }
                    Err(_) => "Models: unavailable".to_string(),
                }
            }
            _ => "Models: unavailable".to_string(),
        },
        None => "Models: unavailable".to_string(),
    };

    TrayStatusSnapshot {
        server_label: "Server: online".to_string(),
        models_label,
    }
}

fn api_url(server_url: &Url, path: &str) -> Option<Url> {
    let mut base_url = server_url.clone();
    base_url.set_query(None);
    base_url.set_fragment(None);
    base_url.set_path("/");
    base_url.join(path).ok()
}

fn model_status_counts(models: &[TrayModelInfo]) -> ModelStatusCounts {
    let mut counts = ModelStatusCounts {
        total: models.len(),
        ..ModelStatusCounts::default()
    };

    for model in models {
        match model.status.as_str() {
            "ready" => counts.ready += 1,
            "downloading" | "loading" => counts.loading += 1,
            "error" => counts.errors += 1,
            _ => {}
        }
    }

    counts
}

fn format_models_label(counts: ModelStatusCounts) -> String {
    if counts.total == 0 {
        return "Models: none installed".to_string();
    }

    let mut parts = Vec::new();
    if counts.ready > 0 {
        parts.push(format!("{} ready", counts.ready));
    }
    if counts.loading > 0 {
        parts.push(format!("{} loading", counts.loading));
    }
    if counts.errors > 0 {
        parts.push(format!("{} errors", counts.errors));
    }
    if parts.is_empty() {
        parts.push(format!("{} not ready", counts.total));
    }

    format!("Models: {}", parts.join(", "))
}
