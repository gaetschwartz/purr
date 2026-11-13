//! Whisper UI CLI - Audio transcription command-line interface
#![deny(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)]

mod build_info;
mod cli;
mod color;
mod download_prompt;
mod global;
mod json;
mod logging;
mod models;
mod output;
mod progress;
mod runtime;
mod scope;
mod stats;
mod sys_command;
mod sys_report;
mod theme;
mod trace;
mod verbosity;
mod vt;

use crate::download_prompt::prompt_for_model_download;
use crate::theme::Themed as _;
use cli::{ASCII_ART, Cli, Commands, OutputFormat, TranscriptionConfigWithCliExt};
use futures::StreamExt as _;
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use miette::IntoDiagnostic as _;
use purr_common::platform::ChunkKind;
use purr_core::{
    ModelManager, StreamEvent, StreamingTranscription, TranscriptionConfig, install_logging_hooks,
    is_file_path, is_valid_url, model::vad_or_none, transcribe_file_stream, transcribe_file_sync,
    transcribe_url_stream, transcribe_url_sync,
};
use std::borrow::Cow;
use std::path::Path;
use std::process::ExitCode;
use tracing::{debug, info};

const APP_NAME: &str = env!("CARGO_BIN_NAME");

#[tokio::main]
async fn main() -> ExitCode {
    match main_impl().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            logging::record_fatal(&error);
            errorln!("{}: {}", "Error".error(), error);
            ExitCode::FAILURE
        }
    }
}

async fn main_impl() -> miette::Result<()> {
    let cli = global::init(Cli::from_env());

    logging::init(cli)?;
    debug!("Command line arguments: {:?}", cli);
    install_logging_hooks();

    if let Some(command) = &cli.command {
        return handle_command(command).await;
    }

    let Some(audio_input) = cli.audio_input.clone() else {
        statusln!("{ASCII_ART}\n");
        return Err(miette::miette!(
            "No audio file or URL specified. Please provide an audio file path or URL to transcribe."
        ));
    };

    let is_url = is_valid_url(&audio_input);
    let is_path = is_file_path(&audio_input);

    if !is_url && !is_path {
        return Err(miette::miette!(
            "Invalid input: '{audio_input}'. Please provide a valid file path or URL."
        ));
    }

    if is_path {
        let path = Path::new(&audio_input);
        if !path.exists() {
            return Err(miette::miette!("Audio file not found: {}", path.display()));
        }
    }

    let kind = if is_url { "URL" } else { "file" };
    info!("Input: {audio_input} ({kind})");
    statusln!("{} {} ({kind})", "Transcribing:".info(), audio_input.path());

    let config = setup_config(cli).await?;

    if let Some(lang) = &config.language {
        info!("Language: {lang}");
        statusln!("{} {lang}", "Language:".info());
    }

    if cli.no_stream {
        debug!("Transcribing without streaming");

        let result = if is_url {
            transcribe_url_sync(&audio_input, Some(config))
                .await
                .into_diagnostic()?
        } else {
            transcribe_file_sync(&audio_input, Some(config))
                .await
                .into_diagnostic()?
        };

        handle_output(result, cli)?;
    } else {
        debug!("Streaming transcription");

        let stream = if is_url {
            transcribe_url_stream(&audio_input, config)
                .await
                .into_diagnostic()?
        } else {
            transcribe_file_stream(&audio_input, config)
                .await
                .into_diagnostic()?
        };

        handle_streaming_output(stream, cli).await?;
    }

    Ok(())
}

fn progress_line_bar() -> miette::Result<ProgressBar> {
    let style = ProgressStyle::with_template("{msg}").into_diagnostic()?;
    Ok(output::add_bar(
        ProgressBar::new_spinner()
            .with_style(style)
            .with_finish(ProgressFinish::AndClear),
    ))
}

/// Handle streaming transcription output
async fn handle_streaming_output(
    mut stream: StreamingTranscription,
    cli: &Cli,
) -> miette::Result<()> {
    use std::fs;

    let mut all_chunks = Vec::new();
    let mut output_buffer = String::new();

    let progress_bar = if cli.progress && output::is_interactive() {
        Some(progress_line_bar()?)
    } else {
        None
    };
    // A bar is drawn below the last line, so each chunk has to end its own line.
    let one_chunk_per_line = progress_bar.is_some();

    while let Some(event) = stream.next().await {
        let chunk = match event? {
            StreamEvent::Progress(progress) => {
                if let Some(bar) = &progress_bar {
                    bar.set_message(progress::line(&progress).muted().to_string());
                }
                continue;
            }
            StreamEvent::Beams(snapshot) => {
                traceln!("{}", trace::line(&snapshot));
                continue;
            }
            StreamEvent::Chunk(chunk) => chunk,
        };
        match chunk.kind {
            ChunkKind::Committed => {}
            ChunkKind::Tentative => continue,
        }
        all_chunks.push(chunk.clone());

        if !chunk.text.is_empty() || matches!(cli.output, OutputFormat::Json) {
            let chunk_text: Cow<'_, str> = match cli.output {
                OutputFormat::Text => {
                    if cli.timestamps {
                        format!("[{:.2}s -> {:.2}s] {}", chunk.start, chunk.end, chunk.text).into()
                    } else {
                        Cow::Borrowed(&chunk.text)
                    }
                }
                OutputFormat::Json => {
                    serde_json::to_string(&json::chunk(&chunk, cli.word_timestamps))
                        .into_diagnostic()?
                        .into()
                }
                OutputFormat::Srt => format!(
                    "{}\n{} --> {}\n{}\n",
                    chunk.chunk_index + 1,
                    format_srt_time(chunk.start),
                    format_srt_time(chunk.end),
                    chunk.text
                )
                .into(),
                OutputFormat::Txt => Cow::Borrowed(&chunk.text),
            };

            if cli.output_file.is_some() {
                output_buffer.push_str(&chunk_text);
                if !matches!(cli.output, OutputFormat::Txt) {
                    output_buffer.push('\n');
                }
            } else if matches!(cli.output, OutputFormat::Json) {
                resultln!("{chunk_text}")?;
            } else {
                result!("{chunk_text}")?;
                if !chunk.text.is_empty() && !chunk.text.ends_with('\n') {
                    if matches!(cli.output, OutputFormat::Srt) || one_chunk_per_line {
                        resultln!()?;
                    } else {
                        result!(" ")?;
                    }
                }
            }
        }
        if let Some(ref chunk_stats) = chunk.final_stats
            && cli.verbosity.is_verbose()
        {
            stats::show("Streaming Transcription Statistics:", chunk_stats, None);
        }
    }

    if let Some(bar) = &progress_bar {
        bar.finish_and_clear();
    }

    if let Some(output_file) = &cli.output_file {
        fs::write(output_file, &output_buffer).into_diagnostic()?;
        if cli.verbosity.is_verbose() {
            statusln!(
                "{} Streaming output written to: {}",
                "Success:".success(),
                output_file.display().path()
            );
        }
    } else {
        resultln!()?;
    }

    if cli.verbosity.is_verbose() {
        debug!("Processed {} chunks", all_chunks.len());
    }

    if all_chunks.is_empty() {
        return Err(miette::miette!(
            "No audio content was transcribed. The file may not be a valid audio file."
        ));
    }

    Ok(())
}

/// Handle subcommands
async fn handle_command(command: &Commands) -> miette::Result<()> {
    match command {
        Commands::Models { command } => models::handle_model_command(command).await,
        Commands::Sys { json } => sys_command::handle(*json),
        Commands::Runtime { command } => runtime::handle_runtime_command(command),
    }
}

/// Format time for SRT subtitles (HH:MM:SS,mmm)
fn format_srt_time(seconds: f64) -> String {
    let hours = (seconds / 3600.0) as u32;
    let minutes = ((seconds % 3600.0) / 60.0) as u32;
    let secs = (seconds % 60.0) as u32;
    let millis = ((seconds % 1.0) * 1000.0) as u32;

    format!("{hours:02}:{minutes:02}:{secs:02},{millis:03}")
}

async fn setup_config(cli: &Cli) -> miette::Result<TranscriptionConfig> {
    let mut config = TranscriptionConfig::new();

    config = setup_model_config(cli, config).await?;

    config = config.with_cli_options(cli);

    if config.vad_model_path.is_none() {
        let installed = ModelManager::new()?.vad_model_path().await;
        config = config.with_vad_model(vad_or_none(installed));
    }

    Ok(config)
}

async fn setup_model_config(
    cli: &Cli,
    mut config: TranscriptionConfig,
) -> Result<TranscriptionConfig, miette::Error> {
    let model_manager = ModelManager::new()?;
    if let Some(ref model_string) = cli.model {
        let model_path = Path::new(model_string);
        if model_path.is_absolute() {
            if !model_path.exists() {
                return Err(miette::miette!(
                    "Model file not found at: {}",
                    model_path.display()
                ));
            }
            config = config.with_model_path(model_path);
        } else {
            let model_path = std::env::current_dir().into_diagnostic()?.join(model_path);
            if model_path.exists() {
                config = config.with_model_path(model_path);
            } else {
                let model = model_manager.model(model_string)?;
                if model_manager.is_model_downloaded(model).await? {
                    config = config.with_model_path(model_manager.get_model_path(model));
                } else if let Some(model_path) =
                    prompt_for_model_download(&model_manager, Some(model)).await?
                {
                    config = config.with_model_path(model_path);
                } else {
                    return Err(miette::miette!(
                        "No model specified and no downloaded models found."
                    ));
                }
            }
        }
    } else if let Some(model) = model_manager.find_first_available_model().await? {
        config = config.with_model_path(model);
    } else if let Some(model_path) = prompt_for_model_download(&model_manager, None).await? {
        config = config.with_model_path(model_path);
    } else {
        return Err(miette::miette!(
            "No model specified and no downloaded models found."
        ));
    }
    Ok(config)
}

fn handle_output(result: purr_core::SyncTranscriptionResult, cli: &Cli) -> miette::Result<()> {
    let output_content = match cli.output {
        OutputFormat::Text => {
            if cli.timestamps {
                result
                    .segments
                    .iter()
                    .map(|segment| {
                        format!(
                            "[{:.2}s -> {:.2}s] {}",
                            segment.start, segment.end, segment.text
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                result.text.clone()
            }
        }
        OutputFormat::Json => {
            serde_json::to_string_pretty(&json::document(&result, cli.word_timestamps))
                .into_diagnostic()?
        }
        OutputFormat::Srt => result
            .segments
            .iter()
            .enumerate()
            .map(|(i, segment)| {
                format!(
                    "{}\n{} --> {}\n{}\n",
                    i + 1,
                    format_srt_time(segment.start),
                    format_srt_time(segment.end),
                    segment.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
        OutputFormat::Txt => result.text.clone(),
    };

    if let Some(output_file) = &cli.output_file {
        std::fs::write(output_file, &output_content).into_diagnostic()?;
        if cli.verbosity.is_verbose() {
            statusln!(
                "{} Output written to: {}",
                "Success:".success(),
                output_file.display().path()
            );
        }
    } else {
        result!("{output_content}")?;
    }

    if cli.verbosity.is_verbose() {
        stats::show(
            "Transcription Statistics:",
            &result.stats,
            Some(result.accel.selected.pretty_name()),
        );
    }

    Ok(())
}
