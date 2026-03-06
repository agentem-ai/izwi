pub mod cli;

use crate::commands;
use crate::error::Result;
use crate::style::Theme;

use self::cli::{Backend, Cli, Commands, ServeMode};

pub async fn run(cli: Cli, theme: Theme) -> Result<()> {
    let Cli {
        command,
        config,
        server,
        output_format,
        quiet,
        ..
    } = cli;

    match command {
        Commands::Serve {
            mode,
            host,
            port,
            models_dir,
            max_batch_size,
            backend,
            threads,
            max_concurrent,
            timeout,
            log_level,
            dev,
            cors,
            no_ui,
        } => {
            commands::serve::execute(serve_args(
                mode,
                host,
                port,
                models_dir,
                max_batch_size,
                backend,
                threads,
                max_concurrent,
                timeout,
                log_level,
                dev,
                cors,
                no_ui,
            ))
            .await?;
        }
        Commands::Models { command } => {
            commands::models::execute(command, &server, output_format, quiet).await?;
        }
        Commands::Pull { model, force, yes } => {
            commands::pull::execute(model, force, yes, &server, &theme).await?;
        }
        Commands::Rm { model, yes } => {
            commands::rm::execute(model, yes, &server, &theme).await?;
        }
        Commands::List { local, detailed } => {
            commands::list::execute(local, detailed, &server, output_format).await?;
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
                &server,
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
                &server,
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
                &server,
                &theme,
            )
            .await?;
        }
        Commands::Diarize {
            file,
            model,
            num_speakers,
            format,
            output,
            transcribe,
            asr_model,
        } => {
            commands::diarize::execute(
                commands::diarize::DiarizeArgs {
                    file,
                    model,
                    num_speakers,
                    format,
                    output,
                    transcribe,
                    asr_model,
                },
                &server,
            )
            .await?;
        }
        Commands::Align {
            file,
            text,
            model,
            format,
            output,
        } => {
            commands::align::execute(
                commands::align::AlignArgs {
                    file,
                    text,
                    model,
                    format,
                    output,
                },
                &server,
            )
            .await?;
        }
        Commands::Bench { command } => {
            commands::bench::execute(command, &server, &theme).await?;
        }
        Commands::Status { detailed, watch } => {
            commands::status::execute(detailed, watch, &server, &theme).await?;
        }
        Commands::Version { full } => {
            commands::version::execute(full, &theme);
        }
        Commands::Config { command } => {
            commands::config::execute(command, config.as_ref(), &theme).await?;
        }
        Commands::Completions { shell } => {
            commands::completions::execute(shell);
        }
    }

    Ok(())
}

fn serve_args(
    mode: ServeMode,
    host: String,
    port: u16,
    models_dir: Option<std::path::PathBuf>,
    max_batch_size: usize,
    backend: Backend,
    threads: Option<usize>,
    max_concurrent: usize,
    timeout: u64,
    log_level: String,
    dev: bool,
    cors: bool,
    no_ui: bool,
) -> commands::serve::ServeArgs {
    commands::serve::ServeArgs {
        mode,
        host,
        port,
        models_dir,
        max_batch_size,
        backend: backend.as_str().to_string(),
        threads,
        max_concurrent,
        timeout,
        log_level,
        dev,
        cors,
        no_ui,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_args_converts_backend_enum_to_string() {
        let args = serve_args(
            ServeMode::Server,
            "127.0.0.1".to_string(),
            8080,
            None,
            8,
            Backend::Cuda,
            Some(4),
            32,
            120,
            "info".to_string(),
            false,
            true,
            false,
        );

        assert_eq!(args.backend, "cuda");
        assert!(matches!(args.mode, ServeMode::Server));
        assert_eq!(args.host, "127.0.0.1");
    }
}
