use anyhow::{Context, Result};
use clap::Parser;
use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
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
    let (server_host, server_port) = server_host_port(&server_url)?;
    let server_origin = format!("{}://{}:{}", server_url.scheme(), server_host, server_port);
    let window_title = args.window_title.clone();
    let width = args.width;
    let height = args.height;
    let managed_server = Arc::new(Mutex::new(None::<Child>));
    let setup_server_handle = Arc::clone(&managed_server);

    let app = tauri::Builder::default()
        .setup(move |app| {
            if let Some(server_child) = maybe_start_local_server(app.handle(), &server_url)? {
                let mut child_slot = setup_server_handle
                    .lock()
                    .map_err(|_| anyhow::anyhow!("failed to acquire server startup lock"))?;
                *child_slot = Some(server_child);
            }

            #[cfg(target_os = "macos")]
            if is_running_from_macos_app_bundle() {
                if let Err(err) = ensure_macos_cli_links(app.handle()) {
                    eprintln!(
                        "warning: could not configure terminal commands automatically: {err}"
                    );
                }
            }

            let init_script = format!(
                "window.__IZWI_SERVER_URL__ = {};",
                js_string_literal(&server_origin)
            );
            let mut window_builder =
                WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                    .initialization_script(init_script)
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
        .build(tauri::generate_context!())
        .map_err(|e| anyhow::anyhow!("failed to build desktop app: {}", e))?;

    let exit_code = app.run_return(|_, _| {});

    if let Ok(mut child_slot) = managed_server.lock() {
        if let Some(mut child) = child_slot.take() {
            shutdown_child(&mut child);
        }
    }

    if exit_code != 0 {
        return Err(anyhow::anyhow!(
            "desktop app exited with code {}",
            exit_code
        ));
    }

    Ok(())
}

fn js_string_literal(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r");
    format!("\"{}\"", escaped)
}

fn server_host_port(server_url: &Url) -> Result<(String, u16)> {
    let host = server_url
        .host_str()
        .context("--server-url must include a host")?
        .to_string();
    let port = server_url
        .port_or_known_default()
        .context("--server-url must include a port or use a known scheme")?;
    Ok((host, port))
}

fn maybe_start_local_server<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    server_url: &Url,
) -> Result<Option<Child>> {
    const START_TIMEOUT: Duration = Duration::from_secs(15);
    const POLL_INTERVAL: Duration = Duration::from_millis(200);
    const CONNECT_TIMEOUT: Duration = Duration::from_millis(250);

    let (host, port) = server_host_port(server_url)?;
    if !is_local_server_host(&host) {
        return Ok(None);
    }

    if is_server_reachable(&host, port, CONNECT_TIMEOUT) {
        return Ok(None);
    }

    let mut cmd = match resolve_server_binary(app) {
        Some(path) => Command::new(path),
        None => Command::new(platform_binary_name("izwi-server")),
    };

    let bind_host = if host == "localhost" {
        "127.0.0.1"
    } else {
        host.as_str()
    };

    cmd.env("IZWI_HOST", bind_host)
        .env("IZWI_PORT", port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .stdin(Stdio::null());

    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to start izwi-server for {}:{}", host, port))?;

    let started = Instant::now();
    while started.elapsed() < START_TIMEOUT {
        if is_server_reachable(&host, port, CONNECT_TIMEOUT) {
            return Ok(Some(child));
        }

        if let Some(status) = child
            .try_wait()
            .context("failed while checking izwi-server status")?
        {
            anyhow::bail!(
                "izwi-server exited before becoming ready on {}:{} (status: {})",
                host,
                port,
                status
            );
        }

        thread::sleep(POLL_INTERVAL);
    }

    shutdown_child(&mut child);
    anyhow::bail!("timed out waiting for izwi-server on {}:{}", host, port)
}

fn is_local_server_host(host: &str) -> bool {
    matches!(host, "localhost" | "127.0.0.1" | "::1" | "0.0.0.0" | "::")
}

fn is_server_reachable(host: &str, port: u16, timeout: Duration) -> bool {
    let addrs = match (host, port).to_socket_addrs() {
        Ok(addrs) => addrs.collect::<Vec<_>>(),
        Err(_) => return false,
    };

    addrs
        .iter()
        .any(|addr| TcpStream::connect_timeout(addr, timeout).is_ok())
}

fn resolve_server_binary<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> Option<PathBuf> {
    let binary_name = platform_binary_name("izwi-server");
    let mut candidates = Vec::new();

    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join("bin").join(&binary_name));
        candidates.push(resource_dir.join(&binary_name));
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join(&binary_name));
        }
    }

    candidates.into_iter().find(|candidate| candidate.exists())
}

fn platform_binary_name(name: &str) -> String {
    if cfg!(windows) {
        format!("{}.exe", name)
    } else {
        name.to_string()
    }
}

fn shutdown_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
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
