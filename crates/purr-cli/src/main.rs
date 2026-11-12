//! Whisper UI CLI - Audio transcription command-line interface
mod cli;
mod fmt;
mod json;
mod models;
mod progress;
mod runtime;
mod sys_report;
mod trace;

use crate::fmt::{MyFormatter, Verbosity, VerbosityLevel};
use clap::Parser as _;
use cli::{ASCII_ART, Cli, Commands, OutputFormat, TranscriptionConfigWithCliExt};
use indicatif::{HumanBytes, HumanDuration};
use miette::IntoDiagnostic as _;
use owo_colors::OwoColorize as _;
use purr_common::platform::ChunkKind;
use purr_core::model::registry::ModelSpec;
use purr_core::model::vad_or_none;
use purr_core::{
    ModelManager, StreamEvent, StreamingTranscription, TranscriptionConfig,
    accel::backend::AccelBackend, dev::FeatureStatus, install_logging_hooks, is_file_path,
    is_valid_url, transcribe_file_stream, transcribe_file_sync, transcribe_url_stream,
    transcribe_url_sync,
};
use purr_core::{SystemInfo, math::ByteSpeed};
use shadow_rs::shadow;
use std::path::{Path, PathBuf};
use std::process;
use std::{
    borrow::Cow,
    io::{self, Write},
};
use tracing::{Level, debug, error, info, warn};
use tracing_subscriber::{EnvFilter, fmt::time::Uptime};

const APP_NAME: &str = env!("CARGO_BIN_NAME");

shadow!(build);

#[tokio::main]
async fn main() -> miette::Result<()> {
    // Initialize tracing subscriber
    if let Err(e) = main_impl().await {
        println!("{}: {}", "Error".red().bold(), e);
        drop(e);
        process::exit(1);
    }
    Ok(())
}

async fn main_impl() -> miette::Result<()> {
    let cli = Cli::parse();

    // Setup logging
    setup_tracing(&cli)?;
    debug!("Command line arguments: {:?}", cli);
    install_logging_hooks();

    // Handle subcommands
    if let Some(command) = cli.command {
        return handle_command(command, cli.verbosity).await;
    }

    // Handle transcription (original behavior)
    let Some(audio_input) = cli.audio_input.clone() else {
        println!("{ASCII_ART}\n");
        error!(
            "No audio file or URL specified. Please provide an audio file path or URL to transcribe."
        );
        std::process::exit(1);
    };

    // Determine if input is a URL or file path
    let is_url = is_valid_url(&audio_input);
    let is_path = is_file_path(&audio_input);

    if !is_url && !is_path {
        error!(
            "Invalid input: '{}'. Please provide a valid file path or URL.",
            audio_input.bold()
        );
        process::exit(1);
    }

    // Validate file exists if it's a file path
    if is_path {
        let path = std::path::Path::new(&audio_input);
        if !path.exists() {
            error!("Audio file not found: {}", path.display().bold());
            process::exit(1);
        }
    }

    // Display input type
    if is_url {
        info!("Input: {} (URL)", audio_input.cyan());
    } else {
        info!("Input: {} (file)", audio_input.cyan());
    }

    let config = setup_config(&cli).await?;

    // Print startup info
    info!("{}", "Whisper UI - Audio Transcription".blue().bold());
    if let Some(lang) = &config.language {
        info!("Language: {lang}");
    }

    if cli.no_stream {
        info!("Transcribing audio...");

        let result = if is_url {
            transcribe_url_sync(&audio_input, Some(config))
                .await
                .into_diagnostic()?
        } else {
            transcribe_file_sync(&audio_input, Some(config))
                .await
                .into_diagnostic()?
        };

        handle_output(result, &cli)?;
    } else {
        info!("Streaming transcription...");

        // Handle streaming transcription
        let stream = if is_url {
            transcribe_url_stream(&audio_input, config)
                .await
                .into_diagnostic()?
        } else {
            transcribe_file_stream(&audio_input, config)
                .await
                .into_diagnostic()?
        };

        // Process streaming results
        handle_streaming_output(stream, &cli).await?;
    }

    Ok(())
}

fn setup_tracing(cli: &Cli) -> Result<(), miette::Error> {
    match *cli.verbosity.verbose {
        VerbosityLevel::DEBUG_VALUE.. => {
            tracing_subscriber::fmt()
                .with_env_filter(
                    EnvFilter::builder()
                        .with_default_directive(Level::TRACE.into())
                        .from_env()
                        .into_diagnostic()?,
                )
                .with_timer(Uptime::default())
                .with_writer(io::stderr)
                .init();
        }
        VerbosityLevel::VERBOSE_VALUE => {
            tracing_subscriber::fmt()
                .with_timer(Uptime::default())
                .event_format(MyFormatter::new(cli.verbosity))
                .with_writer(io::stderr)
                .init();
        }
        VerbosityLevel::NORMAL_VALUE => {
            tracing_subscriber::fmt()
                .event_format(MyFormatter::new(cli.verbosity))
                .with_writer(io::stderr)
                .init();
        }
    }

    Ok(())
}

/// Handle streaming transcription output
async fn handle_streaming_output(
    mut stream: StreamingTranscription,
    cli: &Cli,
) -> miette::Result<()> {
    use std::fs;

    let mut all_chunks = Vec::new();
    let mut output_buffer = String::new();
    let mut stdout = io::stdout();

    use futures::StreamExt;

    let mut progress_shown = false;
    while let Some(event) = stream.next().await {
        let chunk = match event? {
            StreamEvent::Progress(progress) => {
                if cli.progress {
                    eprint!("\r{}", progress::line(&progress));
                    progress_shown = true;
                }
                continue;
            }
            StreamEvent::Beams(snapshot) => {
                if progress_shown {
                    eprint!("\r");
                }
                eprintln!("{}", trace::line(&snapshot));
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
            // Format the chunk for real-time output
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

            // Print to stdout or accumulate for file output
            if cli.output_file.is_some() {
                output_buffer.push_str(&chunk_text);
                if !matches!(cli.output, OutputFormat::Txt) {
                    output_buffer.push('\n');
                }
            } else if matches!(cli.output, OutputFormat::Json) {
                writeln!(stdout, "{chunk_text}").into_diagnostic()?;
            } else {
                write!(stdout, "{chunk_text}").into_diagnostic()?;
                if !chunk.text.is_empty() && !chunk.text.ends_with('\n') {
                    if matches!(cli.output, OutputFormat::Srt) {
                        writeln!(stdout).into_diagnostic()?;
                    } else {
                        write!(stdout, " ").into_diagnostic()?;
                    }
                }
                stdout.flush().into_diagnostic()?;
            }
        }
        // Check for final statistics
        if let Some(ref stats) = chunk.final_stats {
            // Display statistics after processing is complete
            if cli.verbosity.is_verbose() {
                eprintln!();
                eprintln!("{}", "Streaming Transcription Statistics:".green().bold());
                eprintln!("Audio duration: {:.2}s", stats.audio_duration);
                eprintln!("Processing time: {:.2}s", stats.processing_time);
                eprintln!("Real-time factor: {:.2}x", stats.real_time_factor());
                eprintln!("Segments: {}", stats.segment_count);
                eprintln!("Words: {}", stats.word_count);
                eprintln!("Words per minute: {:.1}", stats.words_per_minute());
            }
        }
    }

    if progress_shown {
        eprintln!();
    }

    // Write to file if specified
    if let Some(output_file) = &cli.output_file {
        fs::write(output_file, &output_buffer).into_diagnostic()?;
        if cli.verbosity.is_verbose() {
            info!(
                "\n{} Streaming output written to: {}",
                "Success:".green().bold(),
                output_file.display()
            );
        }
    } else {
        println!(); // Final newline for stdout
    }

    if cli.verbosity.is_verbose() {
        debug!("Processed {} chunks", all_chunks.len());
    }

    // If no chunks were produced, consider this a failure
    if all_chunks.is_empty() {
        error!("No audio content was transcribed. The file may not be a valid audio file.");
        process::exit(1);
    }

    Ok(())
}

/// Prompt user to download base model when none is found
async fn prompt_for_model_download(
    model_manager: &ModelManager,
    model: Option<&ModelSpec>,
) -> miette::Result<Option<PathBuf>> {
    if let Some(model) = model {
        println!();
        println!(
            "{} Model {} is not downloaded!",
            "Notice:".yellow().bold(),
            model.id.green().bold()
        );
    } else {
        println!();
        println!("{} No Whisper model found!", "Notice:".yellow().bold());
        println!("To transcribe audio, you need to download a Whisper model first.");
        println!();
        println!(
            "The {} model is recommended for most users:",
            model_manager.registry().default_model().id.green().bold()
        );
    }
    let model = model.unwrap_or_else(|| model_manager.registry().default_model());
    println!("  • {}", model.description.green());
    println!("  • Size: {}", HumanBytes(model.file.size_bytes).yellow());
    println!(
        "  • Download time: {}",
        HumanDuration(model.estimated_download_time(64u32 * ByteSpeed::MIBPS)).yellow()
    );
    println!();

    print!(
        "Would you like to download the {} model now? [Y/n]: ",
        model.id
    );
    io::stdout().flush().into_diagnostic()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input).into_diagnostic()?;
    let input = input.trim().to_lowercase();

    // Default to 'yes' if user just presses enter
    let should_download = input.is_empty() || input == "y" || input == "yes";

    if should_download {
        println!();
        println!(
            "{} Downloading {} model...",
            "Info:".blue().bold(),
            model.id
        );

        match model_manager.download_model(model, None).await {
            Ok(model_path) => {
                models::ensure_vad(model_manager).await;
                println!(
                    "{} Model downloaded successfully to: {}",
                    "Success:".green().bold(),
                    model_path.display()
                );
                println!();
                Ok(Some(model_path))
            }
            Err(e) => {
                error!("Failed to download model: {}", e);
                Err(e.into())
            }
        }
    } else {
        println!();
        println!("Model download cancelled. You can download a model later with:");
        println!(
            "  {}{}",
            env!("CARGO_PKG_NAME").cyan(),
            " models download <model>".cyan()
        );
        Ok(None)
    }
}

/// Handle subcommands
async fn handle_command(command: Commands, verbosity: Verbosity) -> miette::Result<()> {
    match command {
        Commands::Models { command } => models::handle_model_command(command, verbosity).await,
        Commands::Sys { json } => handle_sys_command(verbosity, json).await,
        Commands::Runtime { command } => runtime::handle_runtime_command(command),
    }
}

/// Handle system subcommands
async fn handle_sys_command(_verbosity: Verbosity, json: bool) -> miette::Result<()> {
    let sys = SystemInfo::get();

    if json {
        println!("{}", sys_report::json(&sys).into_diagnostic()?);
        return Ok(());
    }

    fn feature_status(backend: AccelBackend, feature: &FeatureStatus) -> String {
        match feature {
            FeatureStatus::Disabled => "Not built".yellow().to_string(),
            FeatureStatus::RuntimeLibraryMissing { library, .. } => {
                let hint = backend
                    .runtime_library_hint()
                    .map(|hint| format!("; {hint}"))
                    .unwrap_or_default();
                format!("Built, but the runtime library {library} cannot be loaded{hint}")
                    .red()
                    .to_string()
            }
            FeatureStatus::EnabledButNotAvailable => {
                "Built, but no usable device".red().to_string()
            }
            FeatureStatus::Available(_) => "Available".green().to_string(),
        }
    }

    println!(
        "{}",
        "Acceleration backends (default priority order, best first):"
            .blue()
            .bold()
    );
    println!();
    let longest = AccelBackend::DEFAULT_PRIORITY
        .iter()
        .map(|b| b.pretty_name().len())
        .max()
        .unwrap_or(0);

    for backend in AccelBackend::DEFAULT_PRIORITY {
        let status = sys
            .backends
            .get(backend)
            .unwrap_or(&FeatureStatus::Disabled);
        println!(
            "  * {:longest$} - {}",
            backend.pretty_name(),
            feature_status(*backend, status)
        );
        if let FeatureStatus::RuntimeLibraryMissing { searched, .. } = status {
            println!("    {}", runtime::searched_summary(searched).dimmed());
        }
    }
    println!();

    println!("{}", "Devices:".blue().bold());
    println!();

    if sys.devices.is_empty() {
        warn!("{} No devices found.", "Info:".blue().bold());
        warn!("{} To enable GPU support, ensure:", "Info:".blue().bold());
        warn!("  • Vulkan drivers are installed");
        warn!("  • Compatible GPU hardware is available");
        warn!("  • Vulkan feature is enabled (use --features vulkan)");
    } else {
        for device in &sys.devices {
            let picked_for: Vec<_> = AccelBackend::DEFAULT_PRIORITY
                .iter()
                .copied()
                .filter(|backend| sys.picked_gpus.get(backend) == Some(&device.id))
                .collect();
            let mut lines = sys_report::device_lines(device, &picked_for).into_iter();
            if let Some(headline) = lines.next() {
                println!("{}", headline.green());
            }
            for line in lines {
                println!("{line}");
            }
            println!();
        }
    }

    println!("{}", "CPU:".blue().bold());
    println!();
    let mut lines = sys_report::cpu_lines(&sys.cpu).into_iter();
    if let Some(headline) = lines.next() {
        println!("{}", headline.green());
    }
    for line in lines {
        println!("{line}");
    }

    Ok(())
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
    // Build transcription config
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
            // If absolute path, use it directly
            if !model_path.exists() {
                return Err(miette::miette!(
                    "Model file not found at: {}",
                    model_path.display()
                ));
            }
            config = config.with_model_path(model_path);
        } else {
            // Otherwise, resolve relative to current directory
            let model_path = std::env::current_dir().into_diagnostic()?.join(model_path);
            if model_path.exists() {
                config = config.with_model_path(model_path);
            } else {
                // intelligently check if the model is downloaded
                let model = model_manager.model(model_string)?;
                if model_manager.is_model_downloaded(model).await? {
                    config = config.with_model_path(model_manager.get_model_path(model));
                } else {
                    // If not downloaded, prompt user to download
                    if let Some(model_path) =
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
        }
    } else {
        // No model specified, check if any downloaded models exist
        if let Some(model) = model_manager.find_first_available_model().await? {
            config = config.with_model_path(model);
        } else {
            // Prompt user to download the recommended model
            if let Some(model_path) = prompt_for_model_download(&model_manager, None).await? {
                config = config.with_model_path(model_path);
            } else {
                return Err(miette::miette!(
                    "No model specified and no downloaded models found."
                ));
            }
        }
    }
    Ok(config)
}

fn handle_output(result: purr_core::SyncTranscriptionResult, cli: &Cli) -> miette::Result<()> {
    // Prepare output content
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

    // Write output to file or stdout
    if let Some(output_file) = &cli.output_file {
        use std::fs;
        fs::write(output_file, &output_content).into_diagnostic()?;
        if cli.verbosity.is_verbose() {
            println!(
                "{} Output written to: {}",
                "Success:".green().bold(),
                output_file.display()
            );
        }
    } else {
        print!("{output_content}");
    }

    // Print statistics
    if cli.verbosity.is_verbose() {
        println!();
        println!("{}", "Transcription Statistics:".green().bold());
        println!("Audio duration: {:.2}s", result.stats.audio_duration);
        println!("Processing time: {:.2}s", result.stats.processing_time);
        println!("Real-time factor: {:.2}x", result.stats.real_time_factor());
        println!("Segments: {}", result.stats.segment_count);
        println!("Words: {}", result.stats.word_count);
        println!("Words per minute: {:.1}", result.stats.words_per_minute());
        println!("Acceleration: {}", result.accel.selected.pretty_name());
    }

    Ok(())
}
