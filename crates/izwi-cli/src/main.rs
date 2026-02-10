//! Izwi CLI - World-class command-line interface for audio inference
//!
//! Inspired by vLLM, SGlang, Ollama, and llama.cpp CLIs
#![allow(dead_code)]

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

mod commands;
mod config;
mod error;
mod http;
mod style;
mod utils;

use error::Result;
use style::Theme;

/// Izwi - High-performance audio inference engine CLI
///
/// A world-class CLI for text-to-speech and speech-to-text inference
/// optimized for Apple Silicon and CUDA devices.
///
/// Examples:
///   izwi serve                    # Start the server
///   izwi models list              # List available models
///   izwi pull qwen3-tts-0.6b      # Download a model
///   izwi tts "Hello world"        # Generate speech
///   izwi transcribe audio.wav     # Transcribe audio
#[derive(Parser)]
#[command(
    name = "izwi",
    about = "High-performance audio inference engine",
    long_about = "Izwi is a world-class audio inference engine for text-to-speech (TTS) and automatic speech recognition (ASR). Optimized for Apple Silicon and CUDA devices.",
    version = env!("CARGO_PKG_VERSION"),
    author = "Agentem <info@agentem.com>",
    help_template = style::HELP_TEMPLATE,
    arg_required_else_help = true,
    propagate_version = true,
    disable_colored_help = false,
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Configuration file path
    #[arg(long, global = true, value_name = "PATH")]
    pub config: Option<PathBuf>,

    /// Server URL for API commands
    #[arg(
        long,
        global = true,
        value_name = "URL",
        default_value = "http://localhost:8080"
    )]
    pub server: String,

    /// Output format
    #[arg(
        long = "output-format",
        global = true,
        value_enum,
        default_value = "table"
    )]
    pub output_format: OutputFormat,

    /// Suppress all output except results
    #[arg(long, global = true)]
    pub quiet: bool,

    /// Enable verbose output
    #[arg(long, global = true)]
    pub verbose: bool,

    /// Disable colored output
    #[arg(long, global = true)]
    pub no_color: bool,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the inference server
    ///
    /// Launches the HTTP API server with optional configuration.
    /// Supports graceful shutdown with Ctrl+C.
    #[command(name = "serve", alias = "server")]
    Serve {
        /// Startup mode
        ///
        /// - server: Start only the HTTP server
        /// - desktop: Start server and desktop app
        /// - web: Start server and open the web UI in your browser
        #[arg(long, value_enum, default_value = "server", env = "IZWI_SERVE_MODE")]
        mode: ServeMode,

        /// Host to bind to
        #[arg(short = 'H', long, default_value = "0.0.0.0", env = "IZWI_HOST")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value = "8080", env = "IZWI_PORT")]
        port: u16,

        /// Models directory
        #[arg(short, long, env = "IZWI_MODELS_DIR")]
        models_dir: Option<PathBuf>,

        /// Maximum batch size
        #[arg(long, default_value = "8", env = "IZWI_MAX_BATCH_SIZE")]
        max_batch_size: usize,

        /// Enable Metal GPU acceleration (macOS only)
        #[arg(long, env = "IZWI_USE_METAL")]
        metal: bool,

        /// Number of CPU threads
        #[arg(short, long, env = "IZWI_NUM_THREADS")]
        threads: Option<usize>,

        /// Maximum concurrent requests
        #[arg(long, default_value = "100", env = "IZWI_MAX_CONCURRENT")]
        max_concurrent: usize,

        /// Request timeout in seconds
        #[arg(long, default_value = "300", env = "IZWI_TIMEOUT")]
        timeout: u64,

        /// Log level
        #[arg(long, default_value = "info", env = "RUST_LOG")]
        log_level: String,

        /// Enable development mode with hot reload
        #[arg(long, hide = true)]
        dev: bool,

        /// Enable CORS for all origins
        #[arg(long)]
        cors: bool,

        /// Disable the web UI
        #[arg(long)]
        no_ui: bool,
    },

    /// Manage models
    #[command(name = "models", alias = "model")]
    Models {
        #[command(subcommand)]
        command: ModelCommands,
    },

    /// Download a model from HuggingFace
    ///
    /// Pulls a model from the HuggingFace Hub and caches it locally.
    /// Supports resume on interrupted downloads.
    #[command(name = "pull", alias = "download")]
    Pull {
        /// Model variant to download
        ///
        /// Examples: qwen3-tts-0.6b-base, qwen3-tts-1.7b-customvoice
        model: String,

        /// Force re-download even if model exists
        #[arg(short, long)]
        force: bool,

        /// Download without confirmation
        #[arg(short, long)]
        yes: bool,
    },

    /// Remove a downloaded model
    #[command(name = "rm", alias = "remove")]
    Rm {
        /// Model variant to remove
        model: String,

        /// Remove without confirmation
        #[arg(short, long)]
        yes: bool,
    },

    /// List available and downloaded models
    ///
    /// Shows both locally available models and models that can be downloaded.
    #[command(name = "list", alias = "ls")]
    List {
        /// Show only downloaded models
        #[arg(short, long)]
        local: bool,

        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Text-to-speech generation
    ///
    /// Generate speech from text using a TTS model.
    /// Supports streaming output and various audio formats.
    #[command(name = "tts", alias = "speak")]
    Tts {
        /// Text to synthesize (or "-" to read from stdin)
        text: String,

        /// Model to use
        #[arg(short, long, default_value = "qwen3-tts-0.6b-base")]
        model: String,

        /// Speaker voice (built-in or reference audio path)
        #[arg(short, long, default_value = "default")]
        speaker: String,

        /// Output file path
        #[arg(short, long, value_name = "PATH")]
        output: Option<PathBuf>,

        /// Audio format
        #[arg(short, long, value_enum, default_value = "wav")]
        format: AudioFormat,

        /// Speech speed multiplier
        #[arg(short = 'r', long, default_value = "1.0")]
        speed: f32,

        /// Temperature for sampling
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Stream output in real-time
        #[arg(long)]
        stream: bool,

        /// Play audio immediately after generation
        #[arg(short, long)]
        play: bool,
    },

    /// Speech-to-text transcription
    ///
    /// Transcribe audio to text using an ASR model.
    #[command(name = "transcribe", alias = "asr")]
    Transcribe {
        /// Audio file to transcribe
        file: PathBuf,

        /// Model to use
        #[arg(short, long, default_value = "qwen3-asr-0.6b")]
        model: String,

        /// Language hint (auto-detect if not specified)
        #[arg(short, long)]
        language: Option<String>,

        /// Output format
        #[arg(short, long, value_enum, default_value = "text")]
        format: TranscriptFormat,

        /// Output file (default: stdout)
        #[arg(short, long, value_name = "PATH")]
        output: Option<PathBuf>,

        /// Include word-level timestamps
        #[arg(long)]
        word_timestamps: bool,
    },

    /// Chat with a multimodal model
    ///
    /// Interactive chat with audio understanding capabilities.
    #[command(name = "chat")]
    Chat {
        /// Model to use
        #[arg(short, long, default_value = "qwen3-tts-1.7b-base")]
        model: String,

        /// Initial system prompt
        #[arg(short, long)]
        system: Option<String>,

        /// Voice to use for responses
        #[arg(short, long)]
        voice: Option<String>,
    },

    /// Run benchmarks
    ///
    /// Performance testing for models and inference engine.
    #[command(name = "bench", alias = "benchmark")]
    Bench {
        /// Benchmark type
        #[command(subcommand)]
        command: BenchCommands,
    },

    /// Show system status and health
    ///
    /// Display server health, loaded models, and resource usage.
    #[command(name = "status", alias = "info")]
    Status {
        /// Show detailed metrics
        #[arg(short, long)]
        detailed: bool,

        /// Watch mode (continuous updates)
        #[arg(short, long, value_name = "SECONDS")]
        watch: Option<u64>,
    },

    /// Show version information
    #[command(name = "version", alias = "v")]
    Version {
        /// Show detailed version info including dependencies
        #[arg(short, long)]
        full: bool,
    },

    /// Manage configuration
    #[command(name = "config")]
    Config {
        #[command(subcommand)]
        command: ConfigCommands,
    },

    /// Generate shell completions
    #[command(name = "completions")]
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[derive(Subcommand)]
pub enum ModelCommands {
    /// List available models
    List {
        /// Show only downloaded models
        #[arg(short, long)]
        local: bool,

        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Show model information
    Info {
        /// Model variant
        model: String,

        /// Show raw JSON
        #[arg(long)]
        json: bool,
    },

    /// Load a model into memory
    Load {
        /// Model variant to load
        model: String,

        /// Wait for model to be fully loaded
        #[arg(short, long)]
        wait: bool,
    },

    /// Unload a model from memory
    Unload {
        /// Model variant to unload (or "all")
        model: String,

        /// Unload without confirmation
        #[arg(short, long)]
        yes: bool,
    },

    /// Show download progress
    Progress {
        /// Model variant
        model: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum BenchCommands {
    /// Benchmark TTS inference
    Tts {
        /// Model to benchmark
        #[arg(short, long, default_value = "qwen3-tts-0.6b-base")]
        model: String,

        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: u32,

        /// Text to synthesize
        #[arg(
            short,
            long,
            default_value = "Hello, this is a benchmark test for text to speech synthesis."
        )]
        text: String,

        /// Enable warmup iteration
        #[arg(long)]
        warmup: bool,
    },

    /// Benchmark ASR inference
    Asr {
        /// Model to benchmark
        #[arg(short, long, default_value = "qwen3-asr-0.6b")]
        model: String,

        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: u32,

        /// Audio file to use
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Enable warmup iteration
        #[arg(long)]
        warmup: bool,
    },

    /// Benchmark system throughput
    Throughput {
        /// Duration in seconds
        #[arg(short, long, default_value = "30")]
        duration: u64,

        /// Concurrent requests
        #[arg(short, long, default_value = "1")]
        concurrent: u32,
    },
}

#[derive(Subcommand)]
pub enum ConfigCommands {
    /// Show current configuration
    Show,

    /// Set a configuration value
    Set {
        /// Configuration key (e.g., server.host, models.dir)
        key: String,
        /// Configuration value
        value: String,
    },

    /// Get a configuration value
    Get {
        /// Configuration key
        key: String,
    },

    /// Edit configuration in default editor
    Edit,

    /// Reset configuration to defaults
    Reset {
        /// Reset without confirmation
        #[arg(short, long)]
        yes: bool,
    },

    /// Show configuration file path
    Path,
}

#[derive(Clone, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable table format
    Table,
    /// JSON output
    Json,
    /// Plain text
    Plain,
    /// YAML format
    Yaml,
}

#[derive(Clone, ValueEnum)]
pub enum AudioFormat {
    /// WAV format (PCM)
    Wav,
    /// MP3 format
    Mp3,
    /// OGG Vorbis
    Ogg,
    /// FLAC format
    Flac,
    /// AAC format
    Aac,
}

#[derive(Clone, ValueEnum)]
pub enum TranscriptFormat {
    /// Plain text output
    Text,
    /// JSON format with metadata
    Json,
    /// Verbose JSON format with timing metadata
    VerboseJson,
}

#[derive(Clone, ValueEnum)]
pub enum Shell {
    Bash,
    Zsh,
    Fish,
    PowerShell,
    Elvish,
}

#[derive(Clone, ValueEnum)]
pub enum ServeMode {
    /// Start the API server only
    Server,
    /// Start API server and desktop application
    Desktop,
    /// Start API server and open the web UI in a browser tab
    Web,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let no_color = cli.no_color || std::env::var_os("NO_COLOR").is_some();

    // Initialize theme based on color preference
    let theme = if no_color {
        Theme::no_color()
    } else {
        Theme::default()
    };

    // Initialize logging
    if cli.verbose {
        tracing_subscriber::fmt::init();
    }

    match cli.command {
        Commands::Serve {
            mode,
            host,
            port,
            models_dir,
            max_batch_size,
            metal,
            threads,
            max_concurrent,
            timeout,
            log_level,
            dev,
            cors,
            no_ui,
        } => {
            commands::serve::execute(commands::serve::ServeArgs {
                mode,
                host,
                port,
                models_dir,
                max_batch_size,
                metal,
                threads,
                max_concurrent,
                timeout,
                log_level,
                dev,
                cors,
                no_ui,
            })
            .await?;
        }

        Commands::Models { command } => {
            commands::models::execute(command, &cli.server, cli.output_format, cli.quiet).await?;
        }

        Commands::Pull { model, force, yes } => {
            commands::pull::execute(model, force, yes, &cli.server, &theme).await?;
        }

        Commands::Rm { model, yes } => {
            commands::rm::execute(model, yes, &cli.server, &theme).await?;
        }

        Commands::List { local, detailed } => {
            commands::list::execute(local, detailed, &cli.server, cli.output_format).await?;
        }

        Commands::Tts {
            text,
            model,
            speaker,
            output,
            format,
            speed,
            temperature,
            stream,
            play,
        } => {
            commands::tts::execute(
                commands::tts::TtsArgs {
                    text,
                    model,
                    speaker,
                    output,
                    format,
                    speed,
                    temperature,
                    stream,
                    play,
                },
                &cli.server,
                &theme,
            )
            .await?;
        }

        Commands::Transcribe {
            file,
            model,
            language,
            format,
            output,
            word_timestamps,
        } => {
            commands::transcribe::execute(
                commands::transcribe::TranscribeArgs {
                    file,
                    model,
                    language,
                    format,
                    output,
                    word_timestamps,
                },
                &cli.server,
            )
            .await?;
        }

        Commands::Chat {
            model,
            system,
            voice,
        } => {
            commands::chat::execute(
                commands::chat::ChatArgs {
                    model,
                    system,
                    voice,
                },
                &cli.server,
                &theme,
            )
            .await?;
        }

        Commands::Bench { command } => {
            commands::bench::execute(command, &cli.server, &theme).await?;
        }

        Commands::Status { detailed, watch } => {
            commands::status::execute(detailed, watch, &cli.server, &theme).await?;
        }

        Commands::Version { full } => {
            commands::version::execute(full, &theme);
        }

        Commands::Config { command } => {
            commands::config::execute(command, cli.config.as_ref(), &theme).await?;
        }

        Commands::Completions { shell } => {
            commands::completions::execute(shell);
        }
    }

    Ok(())
}
