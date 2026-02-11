use crate::error::{CliError, Result};
use crate::style::Theme;
use crate::ConfigCommands;
use console::style;
use std::path::PathBuf;

pub async fn execute(
    command: ConfigCommands,
    config_path_override: Option<&PathBuf>,
    theme: &Theme,
) -> Result<()> {
    let config_path = get_config_path(config_path_override)?;

    match command {
        ConfigCommands::Show => show_config(&config_path, theme).await,
        ConfigCommands::Set { key, value } => set_config(&config_path, &key, &value, theme).await,
        ConfigCommands::Get { key } => get_config(&config_path, &key, theme).await,
        ConfigCommands::Edit => edit_config(&config_path, theme).await,
        ConfigCommands::Reset { yes } => reset_config(&config_path, yes, theme).await,
        ConfigCommands::Path => {
            println!("{}", config_path.display());
            Ok(())
        }
    }
}

fn get_config_path(override_path: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return Ok(path.clone());
    }

    let config_dir = dirs::config_dir()
        .ok_or_else(|| CliError::ConfigError("Could not find config directory".to_string()))?;
    Ok(config_dir.join("izwi").join("config.toml"))
}

fn default_models_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("izwi")
        .join("models")
}

fn default_config_contents() -> Result<String> {
    let mut root = toml::Table::new();

    let mut server = toml::Table::new();
    server.insert(
        "host".to_string(),
        toml::Value::String("localhost".to_string()),
    );
    server.insert("port".to_string(), toml::Value::Integer(8080));
    root.insert("server".to_string(), toml::Value::Table(server));

    let mut models = toml::Table::new();
    models.insert(
        "dir".to_string(),
        toml::Value::String(default_models_dir().to_string_lossy().to_string()),
    );
    root.insert("models".to_string(), toml::Value::Table(models));

    let body = toml::to_string_pretty(&root).map_err(|e| CliError::ConfigError(e.to_string()))?;
    Ok(format!("# Izwi Configuration\n\n{body}"))
}

async fn show_config(path: &PathBuf, theme: &Theme) -> Result<()> {
    if !path.exists() {
        theme.info("No configuration file found. Using defaults.");
        println!("\nDefault configuration:");
        println!("  server.host = \"localhost\"");
        println!("  server.port = 8080");
        println!("  models.dir = \"{}\"", default_models_dir().display());
        return Ok(());
    }

    let content = tokio::fs::read_to_string(path)
        .await
        .map_err(|e| CliError::Io(e))?;

    println!("{}", style("Configuration:").bold());
    println!("  Path: {}", path.display());
    println!();
    println!("{}", content);
    Ok(())
}

async fn set_config(path: &PathBuf, key: &str, value: &str, theme: &Theme) -> Result<()> {
    // Ensure config directory exists
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| CliError::Io(e))?;
    }

    // Read existing or create new
    let mut config = if path.exists() {
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| CliError::Io(e))?;
        content
            .parse::<toml::Table>()
            .unwrap_or_else(|_| toml::Table::new())
    } else {
        toml::Table::new()
    };

    // Parse key path (e.g., "server.host")
    let parts: Vec<&str> = key.split('.').collect();
    if parts.is_empty() {
        return Err(CliError::InvalidInput("Invalid key format".to_string()));
    }

    // Set the value
    let last = parts.len() - 1;
    let mut current = &mut config;
    for (i, part) in parts.iter().enumerate() {
        if i == last {
            current.insert(part.to_string(), toml::Value::String(value.to_string()));
        } else {
            let entry = current
                .entry(part.to_string())
                .or_insert_with(|| toml::Value::Table(toml::Table::new()));
            if let toml::Value::Table(t) = entry {
                current = t;
            } else {
                return Err(CliError::ConfigError(format!(
                    "Key '{}' is not a table",
                    part
                )));
            }
        }
    }

    // Write back
    let content =
        toml::to_string_pretty(&config).map_err(|e| CliError::ConfigError(e.to_string()))?;
    tokio::fs::write(path, content)
        .await
        .map_err(|e| CliError::Io(e))?;

    theme.success(&format!("Set {} = {}", key, value));
    Ok(())
}

async fn get_config(path: &PathBuf, key: &str, _theme: &Theme) -> Result<()> {
    if !path.exists() {
        println!("{} not set (using default)", key);
        return Ok(());
    }

    let content = tokio::fs::read_to_string(path)
        .await
        .map_err(|e| CliError::Io(e))?;
    let config: toml::Table = content
        .parse::<toml::Table>()
        .map_err(|e: toml::de::Error| CliError::ConfigError(e.to_string()))?;

    // Navigate to key
    let parts: Vec<&str> = key.split('.').collect();
    let mut current = &toml::Value::Table(config);
    for part in &parts {
        if let toml::Value::Table(t) = current {
            if let Some(v) = t.get(*part) {
                current = v;
            } else {
                println!("{} not set", key);
                return Ok(());
            }
        } else {
            println!("{} not set", key);
            return Ok(());
        }
    }

    println!("{} = {}", key, current);
    Ok(())
}

async fn edit_config(path: &PathBuf, theme: &Theme) -> Result<()> {
    // Ensure file exists
    if !path.exists() {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| CliError::Io(e))?;
        }
        tokio::fs::write(path, default_config_contents()?)
            .await
            .map_err(|e| CliError::Io(e))?;
    }

    // Open in default editor
    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
    let status = tokio::process::Command::new(&editor)
        .arg(path)
        .status()
        .await
        .map_err(|e| CliError::Other(format!("Failed to launch editor: {}", e)))?;

    if !status.success() {
        return Err(CliError::Other("Editor exited with error".to_string()));
    }

    theme.success("Configuration updated");
    Ok(())
}

async fn reset_config(path: &PathBuf, yes: bool, theme: &Theme) -> Result<()> {
    if !yes {
        println!("This will delete your configuration file.");
        let confirm = dialoguer::Confirm::new()
            .with_prompt("Are you sure?")
            .default(false)
            .interact()
            .map_err(|e| CliError::Other(e.to_string()))?;

        if !confirm {
            println!("Cancelled.");
            return Ok(());
        }
    }

    if path.exists() {
        tokio::fs::remove_file(path)
            .await
            .map_err(|e| CliError::Io(e))?;
    }

    theme.success("Configuration reset to defaults");
    Ok(())
}
