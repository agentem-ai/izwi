use anyhow::{Context, Result};
use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::TrayIconBuilder,
    AppHandle, Manager, Runtime,
};

const TRAY_ID: &str = "izwi-tray";
const TRAY_OPEN_ID: &str = "tray_open";
const TRAY_QUIT_ID: &str = "tray_quit";

pub fn build_basic_tray<R: Runtime>(app: &AppHandle<R>) -> Result<()> {
    let open_item = MenuItem::with_id(app, TRAY_OPEN_ID, "Open Izwi", true, None::<&str>)
        .context("failed to build tray open item")?;
    let separator = PredefinedMenuItem::separator(app)
        .context("failed to build tray separator item")?;
    let quit_item = MenuItem::with_id(app, TRAY_QUIT_ID, "Quit Izwi", true, None::<&str>)
        .context("failed to build tray quit item")?;
    let menu = Menu::with_items(app, &[&open_item, &separator, &quit_item])
        .context("failed to build tray menu")?;

    let mut tray_builder = TrayIconBuilder::with_id(TRAY_ID)
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(|app_handle, event| match event.id().0.as_str() {
            TRAY_OPEN_ID => {
                let _ = show_main_window(app_handle);
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
