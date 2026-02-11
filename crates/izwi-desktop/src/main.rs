use anyhow::{Context, Result};
use clap::Parser;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};
use url::Url;

#[derive(Debug, Parser)]
#[command(
    name = "izwi-desktop",
    about = "Tauri desktop shell for Izwi local inference",
    version
)]
struct DesktopArgs {
    /// Base URL of the Izwi local server
    #[arg(long, default_value = "http://localhost:8080")]
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
            #[cfg(target_os = "macos")]
            if is_running_from_macos_app_bundle() {
                if let Err(err) = ensure_macos_cli_links(app.handle()) {
                    eprintln!(
                        "warning: could not configure terminal commands automatically: {err}"
                    );
                }
            }

            let mut window_builder =
                WebviewWindowBuilder::new(app, "main", WebviewUrl::External(server_url.clone()))
                    .title(window_title.as_str())
                    .inner_size(width, height)
                    .min_inner_size(960.0, 680.0)
                    .resizable(true);

            if let Some(icon) = app.default_window_icon() {
                window_builder = window_builder.icon(icon.clone())?;
            }

            window_builder.build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .map_err(|e| anyhow::anyhow!("failed to run desktop app: {}", e))?;

    Ok(())
}

#[cfg(target_os = "macos")]
fn is_running_from_macos_app_bundle() -> bool {
    let exe = match std::env::current_exe() {
        Ok(path) => path,
        Err(_) => return false,
    };

    exe.components()
        .any(|component| component.as_os_str() == "Contents")
}

#[cfg(target_os = "macos")]
fn ensure_macos_cli_links<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> Result<()> {
    use std::process::Command;

    let resource_dir = match app.path().resource_dir() {
        Ok(path) => path,
        Err(err) => {
            // Raw dev/release binaries (outside an app bundle) don't have a resource directory.
            if err.to_string().to_lowercase().contains("unknown path") {
                return Ok(());
            }
            return Err(err.into());
        }
    };
    let cli_target = resource_dir.join("bin").join("izwi");
    let server_target = resource_dir.join("bin").join("izwi-server");

    // In dev mode these files may not exist yet; skip link setup in that case.
    if !cli_target.exists() {
        return Ok(());
    }

    let link_dir = preferred_path_bin_dir();
    let cli_link = link_dir.join("izwi");
    let server_link = link_dir.join("izwi-server");

    if has_non_symlink_collision(&cli_link)? {
        eprintln!(
            "warning: {} exists and is not a symlink; not overwriting",
            cli_link.display()
        );
        return Ok(());
    }
    if server_target.exists() && has_non_symlink_collision(&server_link)? {
        eprintln!(
            "warning: {} exists and is not a symlink; not overwriting",
            server_link.display()
        );
        return Ok(());
    }

    let mut needs_privileged_install = false;

    if let Err(err) = std::fs::create_dir_all(&link_dir) {
        needs_privileged_install = true;
        eprintln!("warning: {}", err);
    }

    if let Err(err) = ensure_symlink(&cli_target, &cli_link) {
        needs_privileged_install = true;
        eprintln!("warning: {}", err);
    }

    if server_target.exists() {
        if let Err(err) = ensure_symlink(&server_target, &server_link) {
            needs_privileged_install = true;
            eprintln!("warning: {}", err);
        }
    }

    if needs_privileged_install {
        let mut shell_cmd = vec![
            format!("mkdir -p '{}'", escape_single_quotes(&link_dir)),
            format!(
                "ln -sf '{}' '{}'",
                escape_single_quotes(&cli_target),
                escape_single_quotes(&cli_link)
            ),
        ];

        if server_target.exists() {
            shell_cmd.push(format!(
                "ln -sf '{}' '{}'",
                escape_single_quotes(&server_target),
                escape_single_quotes(&server_link)
            ));
        }

        let shell_cmd = shell_cmd.join(" && ");
        let apple_script = format!(
            "do shell script \"{}\" with administrator privileges",
            escape_applescript(&shell_cmd)
        );

        match Command::new("osascript")
            .arg("-e")
            .arg(apple_script)
            .status()
        {
            Ok(status) if status.success() => return Ok(()),
            Ok(_) | Err(_) => {
                eprintln!("warning: automatic privileged setup was not completed");
                eprintln!(
                    "run manually: {}",
                    manual_link_command(&cli_target, &cli_link)
                );
                if server_target.exists() {
                    eprintln!(
                        "run manually: {}",
                        manual_link_command(&server_target, &server_link)
                    );
                }
            }
        }
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn ensure_symlink(target: &std::path::Path, link: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::symlink;

    let existing = std::fs::symlink_metadata(link).ok();
    if let Some(metadata) = existing {
        if metadata.file_type().is_symlink() {
            let current_target = std::fs::read_link(link)
                .with_context(|| format!("failed reading existing link {}", link.display()))?;
            if current_target == target {
                return Ok(());
            }

            std::fs::remove_file(link)
                .with_context(|| format!("failed removing stale link {}", link.display()))?;
        } else {
            anyhow::bail!("{} exists and is not a symlink", link.display());
        }
    }

    symlink(target, link).with_context(|| {
        format!(
            "failed to create symlink {} -> {}",
            link.display(),
            target.display()
        )
    })?;

    Ok(())
}

#[cfg(target_os = "macos")]
fn has_non_symlink_collision(path: &std::path::Path) -> Result<bool> {
    let existing = std::fs::symlink_metadata(path).ok();
    Ok(matches!(existing, Some(metadata) if !metadata.file_type().is_symlink()))
}

#[cfg(target_os = "macos")]
fn preferred_path_bin_dir() -> std::path::PathBuf {
    use std::path::PathBuf;

    let preferred = [
        PathBuf::from("/opt/homebrew/bin"),
        PathBuf::from("/usr/local/bin"),
    ];

    if let Some(path) = std::env::var_os("PATH") {
        for entry in std::env::split_paths(&path) {
            if preferred.iter().any(|candidate| candidate == &entry) {
                return entry;
            }
        }
    }

    if cfg!(target_arch = "aarch64") {
        PathBuf::from("/opt/homebrew/bin")
    } else {
        PathBuf::from("/usr/local/bin")
    }
}

#[cfg(target_os = "macos")]
fn manual_link_command(target: &std::path::Path, link: &std::path::Path) -> String {
    format!(
        "sudo ln -sf '{}' '{}'",
        escape_single_quotes(target),
        escape_single_quotes(link)
    )
}

#[cfg(target_os = "macos")]
fn escape_single_quotes(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\'', r"'\''")
}

#[cfg(target_os = "macos")]
fn escape_applescript(input: &str) -> String {
    input.replace('\\', "\\\\").replace('\"', "\\\"")
}
