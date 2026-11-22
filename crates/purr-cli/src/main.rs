//! Whisper UI CLI - Audio transcription command-line interface
#![deny(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)]

mod build_info;
mod cli;
mod color;
mod download_prompt;
mod engine_status;
mod engines;
mod engines_command;
mod error;
mod fatal;
mod global;
mod inputs;
mod json;
mod language_arg;
mod logging;
mod models;
mod output;
mod output_plan;
mod profile_text;
mod progress;
mod runtime;
mod scope;
mod stats;
mod sys_command;
mod sys_report;
mod template;
mod theme;
mod verbosity;
mod vt;

use crate::download_prompt::prompt_for_model_download;
use crate::error::{CliError, CliResult};
use crate::language_arg::LanguageVerdict;
use crate::theme::Themed as _;
use cli::{ASCII_ART, Cli, Commands, OutputFormat};
use futures::StreamExt as _;
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use purr_common::devices::DeviceOrder;
use purr_common::platform::ChunkKind;
use purr_core::accel::engine_report::{EngineReport, reports};
use purr_core::accel::selector::EngineDevices;
use purr_core::engine::Engine;
use purr_core::{
    ModelManager,
    config::{ModelRef, RunConfig, Windowing},
    is_file_path, is_valid_url,
    model::vad_or_none,
    pipeline::{
        events::{StreamEvent, Transcription},
        result::{SyncTranscriptionResult, collect},
        run::{Input, transcribe},
    },
};
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use tracing::{debug, info};

const APP_NAME: &str = env!("CARGO_BIN_NAME");

#[tokio::main]
async fn main() -> ExitCode {
    match main_impl().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            logging::record_fatal(&error);
            errorln!("{}", fatal::render(error));
            ExitCode::FAILURE
        }
    }
}

async fn main_impl() -> CliResult<()> {
    let cli = global::init(Cli::from_env());

    logging::init(cli)?;
    debug!("Command line arguments: {:?}", cli);
    let engines = engines::all();
    for engine in &engines {
        engine.install_logging();
    }

    if let Some(kind) = cli.throw_error {
        return Err(kind.raise());
    }

    if let Some(command) = &cli.command {
        return handle_command(command).await;
    }

    let Some(audio_input) = cli.audio_input.clone() else {
        statusln!("{ASCII_ART}\n");
        return Err(CliError::NoAudioInput);
    };

    let is_url = is_valid_url(&audio_input);
    let is_path = is_file_path(&audio_input);

    if !is_url && !is_path {
        return Err(CliError::InvalidAudioInput { input: audio_input });
    }

    if is_path {
        let path = Path::new(&audio_input);
        if !path.exists() {
            return Err(CliError::AudioFileNotFound {
                path: path.to_path_buf(),
            });
        }
    }

    let kind = if is_url { "URL" } else { "file" };
    info!("Input: {audio_input} ({kind})");
    statusln!("{} {} ({kind})", "Transcribing:".info(), audio_input.path());

    let (config, model) = setup_config(cli, &engines).await?;

    if let Some(lang) = &config.language {
        info!("Language: {lang}");
        statusln!("{} {lang}", "Language:".info());
    }

    let input = if is_url {
        Input::url(&audio_input, cli.input_config())?
    } else {
        Input::File(PathBuf::from(&audio_input))
    };

    if cli.no_stream {
        debug!("Transcribing without streaming");
        let transcription = transcribe(
            input,
            config.with_windowing(Windowing::Whole),
            &model,
            &engines,
        )
        .await?;
        handle_output(collect(transcription).await?, cli)?;
    } else {
        debug!("Streaming transcription");
        let stream = transcribe(input, config, &model, &engines).await?;
        handle_streaming_output(stream, cli).await?;
    }

    Ok(())
}

fn progress_line_bar() -> CliResult<ProgressBar> {
    let style = ProgressStyle::with_template("{msg}")
        .map_err(|source| CliError::ProgressTemplate { source })?;
    Ok(output::add_bar(
        ProgressBar::new_spinner()
            .with_style(style)
            .with_finish(ProgressFinish::AndClear),
    ))
}

/// Handle streaming transcription output
async fn handle_streaming_output(mut stream: Transcription, cli: &Cli) -> CliResult<()> {
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
                        .map_err(|source| CliError::SerializeJson { source })?
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
        fs::write(output_file, &output_buffer).map_err(|source| CliError::WriteOutput {
            path: output_file.clone(),
            source,
        })?;
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
        return Err(CliError::NothingTranscribed);
    }

    Ok(())
}

/// Handle subcommands
async fn handle_command(command: &Commands) -> CliResult<()> {
    match command {
        Commands::Models { command } => models::handle_model_command(command).await,
        Commands::Sys { json } => sys_command::handle(*json),
        Commands::Engines { json } => engines_command::handle(*json),
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

/// `--device` resolved against the devices of the engine that runs `model`.
fn device_order_for(
    cli: &Cli,
    engines: &[Arc<dyn Engine>],
    model: &ModelRef,
) -> CliResult<DeviceOrder> {
    let found = reports(engines, &[]);
    let nothing = [];
    let own = found
        .iter()
        .find(|report| report.engine == model.engine)
        .map_or(
            EngineDevices {
                engine: model.engine,
                devices: &nothing,
            },
            EngineReport::listed,
        );
    let others: Vec<EngineDevices<'_>> = found
        .iter()
        .filter(|report| report.engine != model.engine)
        .map(EngineReport::listed)
        .collect();
    Ok(cli.device_order_among(own, &others)?)
}

async fn setup_config(cli: &Cli, engines: &[Arc<dyn Engine>]) -> CliResult<(RunConfig, ModelRef)> {
    let model = setup_model(cli).await?;

    let mut config = cli.run_config();
    if let Some(spec) = ModelManager::new()?.listed(&model.path) {
        match language_arg::check(cli.language.as_ref(), &spec.id.to_string(), &spec.profile)? {
            LanguageVerdict::Applies => {}
            LanguageVerdict::HintIgnored => {
                statusln!(
                    "{} {} finds the language itself; --language is ignored.",
                    "Warning:".warning(),
                    spec.family.label()
                );
                config = config.with_opt_language(None);
            }
        }
    }
    if cli.device.is_some() {
        config = config.with_devices(device_order_for(cli, engines, &model)?);
    }

    if config.vad_model.is_none() {
        let installed = ModelManager::new()?.vad_model_path().await;
        config = config.with_vad_model(vad_or_none(installed));
    }

    Ok((config, model))
}

async fn setup_model(cli: &Cli) -> CliResult<ModelRef> {
    let model_manager = ModelManager::new()?;
    let path: PathBuf;
    if let Some(ref model_string) = cli.model {
        let model_path = Path::new(model_string);
        if model_path.is_absolute() {
            if !model_path.exists() {
                return Err(CliError::ModelFileNotFound {
                    path: model_path.to_path_buf(),
                });
            }
            path = model_path.to_path_buf();
        } else {
            let model_path = std::env::current_dir()
                .map_err(|source| CliError::CurrentDir { source })?
                .join(model_path);
            if model_path.exists() {
                path = model_path;
            } else {
                let model = model_manager.model(model_string)?;
                if !engines::built().contains(&model.engine) {
                    return Err(CliError::EngineNotBuilt {
                        model: model.id.to_string(),
                        engine: model.engine.label().to_string(),
                    });
                }
                if model_manager.is_model_downloaded(model).await? {
                    path = model_manager.get_model_path(model);
                } else if let Some(model_path) =
                    prompt_for_model_download(&model_manager, Some(model)).await?
                {
                    path = model_path;
                } else {
                    return Err(CliError::NoModelAvailable);
                }
            }
        }
    } else if let Some(model) = model_manager
        .find_first_available_model(&engines::built())
        .await?
    {
        path = model;
    } else if let Some(model_path) = prompt_for_model_download(&model_manager, None).await? {
        path = model_path;
    } else {
        return Err(CliError::NoModelAvailable);
    }
    Ok(model_manager.reference(path).await?)
}

fn handle_output(result: SyncTranscriptionResult, cli: &Cli) -> CliResult<()> {
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
                .map_err(|source| CliError::SerializeJson { source })?
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
        std::fs::write(output_file, &output_content).map_err(|source| CliError::WriteOutput {
            path: output_file.clone(),
            source,
        })?;
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
