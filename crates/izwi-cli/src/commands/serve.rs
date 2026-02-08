use crate::error::{CliError, Result};
use crate::style::Theme;
use console::style;
use std::path::PathBuf;
use std::process::{Command, Stdio};

pub struct ServeArgs {
    pub host: String,
    pub port: u16,
    pub models_dir: Option<PathBuf>,
    pub max_batch_size: usize,
    pub metal: bool,
    pub threads: Option<usize>,
    pub max_concurrent: usize,
    pub timeout: u64,
    pub log_level: String,
    pub dev: bool,
    pub cors: bool,
    pub no_ui: bool,
}

pub async fn execute(args: ServeArgs) -> Result<()> {
    let theme = Theme::default();

    // Print banner
    theme.print_banner();

    // Check if we're in a supported environment
    let platform = detect_platform();
    println!("   Platform: {}", style(&platform).cyan());

    // Show configuration
    println!("\n{}", style("Configuration:").bold().underlined());
    println!("  Host:           {}:{}", args.host, args.port);
    if let Some(ref dir) = args.models_dir {
        println!("  Models dir:     {}", dir.display());
    }
    println!("  Max batch:      {}", args.max_batch_size);
    println!("  Max concurrent: {}", args.max_concurrent);
    println!("  Timeout:        {}s", args.timeout);
    println!(
        "  Metal GPU:      {}",
        if args.metal || cfg!(target_os = "macos") {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!("  Log level:      {}", args.log_level);

    // Set environment variables
    std::env::set_var("RUST_LOG", &args.log_level);
    std::env::set_var("IZWI_HOST", &args.host);
    std::env::set_var("IZWI_PORT", args.port.to_string());
    std::env::set_var("IZWI_MAX_BATCH_SIZE", args.max_batch_size.to_string());
    std::env::set_var("IZWI_MAX_CONCURRENT", args.max_concurrent.to_string());
    std::env::set_var("IZWI_TIMEOUT", args.timeout.to_string());

    if args.metal || cfg!(target_os = "macos") {
        std::env::set_var("IZWI_USE_METAL", "1");
    }

    if let Some(threads) = args.threads {
        std::env::set_var("IZWI_NUM_THREADS", threads.to_string());
    }

    if let Some(ref dir) = args.models_dir {
        std::env::set_var("IZWI_MODELS_DIR", dir.to_string_lossy().to_string());
    }

    if args.cors {
        std::env::set_var("IZWI_CORS", "1");
    }

    println!("\n{}", style("Starting server...").bold());

    // Try to start the server
    // For now, we'll use cargo run as a fallback, but in production
    // this would use the compiled binary
    let server_binary = if args.dev {
        "cargo".to_string()
    } else {
        // Look for compiled binary
        let binary_path = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .map(|p| p.join("izwi-server"))
            .or_else(|| {
                std::env::current_dir()
                    .ok()
                    .map(|p| p.join("target/release/izwi-server"))
            })
            .unwrap_or_else(|| PathBuf::from("izwi-server"));

        if binary_path.exists() {
            binary_path.to_string_lossy().to_string()
        } else {
            // Fallback to cargo run
            println!("  {}", style("Using development mode (cargo run)").yellow());
            "cargo".to_string()
        }
    };

    let mut cmd = if server_binary == "cargo" {
        let mut c = Command::new("cargo");
        c.arg("run").arg("--bin").arg("izwi-server");
        if !args.dev {
            c.arg("--release");
        }
        c
    } else {
        Command::new(server_binary)
    };

    // Configure command
    cmd.env("RUST_LOG", &args.log_level);
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    println!("\n{}", style("Server is running!").green().bold());
    println!(
        "  API endpoint: {}",
        style(format!("http://{}:{}/v1", args.host, args.port)).cyan()
    );
    if !args.no_ui {
        println!(
            "  Web UI:       {}",
            style(format!("http://{}:{}", args.host, args.port)).cyan()
        );
    }
    println!("\nPress Ctrl+C to stop the server.\n");

    // Run the server and wait for it
    let mut child = cmd
        .spawn()
        .map_err(|e| CliError::Other(format!("Failed to start server: {}", e)))?;

    // Wait for the process
    let status = child
        .wait()
        .map_err(|e| CliError::Other(format!("Server error: {}", e)))?;

    if !status.success() {
        return Err(CliError::Other(format!(
            "Server exited with code: {:?}",
            status.code()
        )));
    }

    Ok(())
}

fn detect_platform() -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let mut features = vec![];

    if cfg!(target_os = "macos") {
        features.push("Metal");
    }

    if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
        features.push("CUDA");
    }

    let feature_str = if features.is_empty() {
        String::new()
    } else {
        format!(" [{}]", features.join(", "))
    };

    format!("{}-{}{}", os, arch, feature_str)
}
