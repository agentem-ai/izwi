use tauri::AppHandle;
use tauri_plugin_autostart::ManagerExt as _;

#[tauri::command]
pub fn launch_at_login_enabled(app: AppHandle) -> std::result::Result<bool, String> {
    app.autolaunch()
        .is_enabled()
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn set_launch_at_login_enabled(
    app: AppHandle,
    enabled: bool,
) -> std::result::Result<(), String> {
    if enabled {
        app.autolaunch().enable().map_err(|error| error.to_string())
    } else {
        app.autolaunch()
            .disable()
            .map_err(|error| error.to_string())
    }
}
