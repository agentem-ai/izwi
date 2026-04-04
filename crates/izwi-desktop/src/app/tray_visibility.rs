use std::sync::Mutex;

use tauri::{AppHandle, State};

pub struct TrayVisibilityState {
    visible: Mutex<bool>,
}

impl TrayVisibilityState {
    pub fn new(initial_visible: bool) -> Self {
        Self {
            visible: Mutex::new(initial_visible),
        }
    }

    pub fn is_visible(&self) -> bool {
        self.visible.lock().map(|guard| *guard).unwrap_or(true)
    }

    pub fn set_visible(&self, visible: bool) {
        if let Ok(mut guard) = self.visible.lock() {
            *guard = visible;
        }
    }
}

#[tauri::command]
pub fn tray_icon_visible(state: State<'_, TrayVisibilityState>) -> bool {
    state.is_visible()
}

#[tauri::command]
pub fn set_tray_icon_visible(
    app: AppHandle,
    state: State<'_, TrayVisibilityState>,
    visible: bool,
) -> std::result::Result<(), String> {
    let tray = app
        .tray_by_id(super::tray::TRAY_ID)
        .ok_or_else(|| "Tray icon is unavailable".to_string())?;

    tray.set_visible(visible)
        .map_err(|error| error.to_string())?;
    state.set_visible(visible);
    Ok(())
}
