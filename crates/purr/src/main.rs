//! Whisper UI CLI - Audio transcription command-line interface
mod cli;
mod fmt;

use crate::fmt::{MyFormatter, Verbosity, VerbosityLevel};
use clap::Parser as _;
use cli::{Cli, Commands, ModelCommands, OutputFormat, TranscriptionConfigWithCliExt, ASCII_ART};
use indicatif::{HumanBytes, HumanDuration, ProgressBar, ProgressStyle};
use miette::IntoDiagnostic as _;
use owo_colors::OwoColorize as _;
use purr_core::{
    dev::{FeatureStatus, WhisperGpuBackend},
    install_logging_hooks, is_valid_url, is_file_path, list_devices, transcribe_file_stream,
    transcribe_file_sync, transcribe_url_stream, transcribe_url_sync, ModelManager,
    StreamingTranscription, TranscriptionConfig, WhisperModel,
};
use purr_core::{
    math::{ByteSpeed, RoundToUnit as _},
    SystemInfo,
};
use shadow_rs::shadow;
use std::path::Path;
use std::process;
use std::str::FromStr as _;
use std::{
    borrow::Cow,
    io::{self, Write},
};
use tracing::{debug, error, info, warn, Level};
use tracing_subscriber::{fmt::time::Uptime, EnvFilter};

const APP_NAME: &str = env!("CARGO_PKG_NAME");

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
        error!("No audio file or URL specified. Please provide an audio file path or URL to transcribe.");
        std::process::exit(1);
    };

    // Determine if input is a URL or file path
    let is_url = is_valid_url(&audio_input);
    let is_path = is_file_path(&audio_input);

    if !is_url && !is_path {
        error!("Invalid input: '{}'. Please provide a valid file path or URL.", audio_input.bold());
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
    if config.use_gpu {
        info!("GPU acceleration: {}", "enabled".green());
    } else {
        info!("GPU acceleration: {}", "disabled".red());
    }
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

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        all_chunks.push(chunk.clone());

        // Format the chunk for real-time output
        let chunk_text: Cow<'_, str> = match cli.output {
            OutputFormat::Text => {
                if cli.timestamps {
                    format!("[{:.2}s -> {:.2}s] {}", chunk.start, chunk.end, chunk.text).into()
                } else {
                    Cow::Borrowed(&chunk.text)
                }
            }
            OutputFormat::Json => serde_json::to_string(&chunk).into_diagnostic()?.into(),
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
        } else if let OutputFormat::Json = cli.output {
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
    model: Option<WhisperModel>,
) -> miette::Result<Option<WhisperModel>> {
    if let Some(model) = model {
        println!();
        println!(
            "{} Model {} is not downloaded!",
            "Notice:".yellow().bold(),
            model.as_str().green().bold()
        );
    } else {
        println!();
        println!("{} No Whisper model found!", "Notice:".yellow().bold());
        println!("To transcribe audio, you need to download a Whisper model first.");
        println!();
        println!(
            "The {} model is recommended for most users:",
            "base".green().bold()
        );
    }
    let model = model.unwrap_or(WhisperModel::Base);
    println!("  • {}", model.description().green());
    println!(
        "  • Size: ~{}",
        HumanBytes(model.size().round_to_unit(1024)).yellow()
    );
    println!(
        "  • Download time: {}",
        HumanDuration(model.estimated_download_time(64u32 * ByteSpeed::MIBPS)).yellow()
    );
    println!();

    print!("Would you like to download the base model now? [Y/n]: ");
    io::stdout().flush().into_diagnostic()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input).into_diagnostic()?;
    let input = input.trim().to_lowercase();

    // Default to 'yes' if user just presses enter
    let should_download = input.is_empty() || input == "y" || input == "yes";

    if should_download {
        println!();
        println!("{} Downloading base model...", "Info:".blue().bold());

        let model_manager = ModelManager::new()?;
        match model_manager.download_model(model).await {
            Ok(model_path) => {
                println!(
                    "{} Model downloaded successfully to: {}",
                    "Success:".green().bold(),
                    model_path.display()
                );
                println!();
                Ok(Some(model))
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
        Commands::Models { command } => handle_model_command(command, verbosity).await,
        Commands::Sys {} => handle_sys_command(verbosity).await,
    }
}

/// Handle model management subcommands
async fn handle_model_command(command: ModelCommands, verbosity: Verbosity) -> miette::Result<()> {
    let model_manager = ModelManager::new()?;

    match command {
        ModelCommands::Download { model, force } => {
            let whisper_model = WhisperModel::from_str(&model).map_err(|e| {
                miette::miette!(
                    "Unknown model: {}. Use 'models list' to see available models. Error: {}",
                    model,
                    e
                )
            })?;

            // check if it is already downloaded
            if !force && model_manager.is_model_downloaded(whisper_model).await {
                println!(
                    "{} Model {} is already downloaded.",
                    "Info:".blue().bold(),
                    whisper_model.as_str()
                );
                return Ok(());
            }

            println!(
                "{} Downloading model: {} ({})",
                "Info:".blue().bold(),
                whisper_model.as_str(),
                whisper_model.description()
            );

            // Create progress bar
            let progress_bar = ProgressBar::new(0);
            progress_bar.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
                    .unwrap()
                    .progress_chars("#>-")
            );

            model_manager.download_model_with_progress(whisper_model, |downloaded, total| {
                if let Some(total) = total {
                    if progress_bar.length().unwrap_or(0) != total {
                        progress_bar.set_length(total);
                    }
                    progress_bar.set_position(downloaded);
                } else {
                    // If total size is unknown, show as spinner with downloaded bytes
                    progress_bar.set_style(
                        ProgressStyle::default_spinner()
                            .template("{spinner:.green} [{elapsed_precise}] {bytes} downloaded... {msg}")
                            .unwrap()
                    );
                    progress_bar.set_position(downloaded);
                }
            }).await?;

            let per_sec = progress_bar
                .length()
                .map(|len| len as f64 / progress_bar.elapsed().as_secs_f64());
            let elapsed = progress_bar.elapsed();
            progress_bar.finish_and_clear();

            println!(
                "{} Model downloaded in {:#}{}.",
                "Success:".green().bold(),
                HumanDuration(elapsed).cyan(),
                if let Some(per_sec) = per_sec {
                    format!(
                        " ({}{} avg)",
                        HumanBytes(per_sec as u64).cyan(),
                        "/s".cyan()
                    )
                } else {
                    String::new()
                }
            );
        }

        ModelCommands::List { available } => {
            if available {
                // List available models
                println!("{}", "Available Whisper Models:".blue().bold());
                println!();

                // Group models by base type and show quantized variants together
                print_model_groups();

                println!();
                println!(
                    "{}{}{}",
                    "Usage: ".dimmed(),
                    env!("CARGO_PKG_NAME").cyan().dimmed(),
                    " models download <model>".cyan().dimmed()
                );
                println!(
                    "{}{}{}",
                    "Example: ".dimmed(),
                    env!("CARGO_PKG_NAME").cyan().dimmed(),
                    " models download base".cyan().dimmed()
                );
            } else {
                // List downloaded models (default behavior)
                let downloaded = model_manager.list_downloaded_models().await?;

                if downloaded.is_empty() {
                    println!("{} No models downloaded yet.", "Info:".blue().bold());
                    println!(
                        "Use {}{} models download <model> to download a model.",
                        env!("CARGO_PKG_NAME").cyan(),
                        " models download base".cyan()
                    );
                } else {
                    println!("{} Downloaded models:", "Info:".blue().bold());
                    println!();

                    for model in downloaded {
                        let path = model_manager.get_model_path(model);
                        let size = if let Ok(metadata) = std::fs::metadata(&path) {
                            format_file_size(metadata.len())
                        } else {
                            "unknown size".to_string()
                        };

                        println!(
                            "  {} - {} ({})",
                            model.as_str().green(),
                            model.description().dimmed(),
                            size.yellow()
                        );

                        if verbosity.is_verbose() {
                            println!("    Path: {}", path.display().to_string().dimmed());
                        }
                    }

                    println!();
                    println!(
                        "XDG data directory: {}",
                        model_manager.models_dir().display().to_string().dimmed()
                    );
                }
            }
        }

        ModelCommands::Delete { model } => {
            let whisper_model = WhisperModel::from_str(&model).map_err(|e| {
                miette::miette!(
                    "Unknown model: {}. Use 'models list' to see available models. Error: {}",
                    model,
                    e
                )
            })?;

            if !model_manager.is_model_downloaded(whisper_model).await {
                println!(
                    "{} Model {} is not downloaded.",
                    "Warning:".yellow().bold(),
                    whisper_model.as_str()
                );
                return Ok(());
            }

            model_manager.delete_model(whisper_model).await?;

            println!(
                "{} Model {} deleted successfully.",
                "Success:".green().bold(),
                whisper_model.as_str()
            );
        }

        ModelCommands::Info { model } => {
            let whisper_model = WhisperModel::from_str(&model).map_err(|e| {
                miette::miette!(
                    "Unknown model: {}. Use 'models list' to see available models. Error: {}",
                    model,
                    e
                )
            })?;

            println!("{} Model Information", "Info:".blue().bold());
            println!();
            println!("Name: {}", whisper_model.as_str().green().bold());
            println!("Description: {}", whisper_model.description());
            println!("Filename: {}", whisper_model.filename().yellow());

            let is_downloaded = model_manager.is_model_downloaded(whisper_model).await;
            println!(
                "Downloaded: {}",
                if is_downloaded {
                    "yes".green().to_string()
                } else {
                    "no".red().to_string()
                }
            );

            if is_downloaded {
                let path = model_manager.get_model_path(whisper_model);
                println!("Path: {}", path.display());

                if let Ok(metadata) = std::fs::metadata(&path) {
                    println!("Size: {}", format_file_size(metadata.len()).yellow());
                }
            }
        }
    }

    Ok(())
}

/// Handle system subcommands
async fn handle_sys_command(_verbosity: Verbosity) -> miette::Result<()> {
    let sys = SystemInfo::get();

    fn feature_status(feature: &FeatureStatus) -> String {
        match feature {
            FeatureStatus::Disabled => "Disabled".yellow().to_string(),
            FeatureStatus::EnabledButNotAvailable => "Enabled but not available".red().to_string(),
            FeatureStatus::Available(_) => "Enabled".green().to_string(),
        }
    }

    println!("{}", "Backends:".blue().bold());
    println!();
    let mut gpu_backends = WhisperGpuBackend::ALL
        .iter()
        .map(|b| (b.pretty_name(), &sys.backends[b]))
        .collect::<Vec<_>>();
    gpu_backends.sort_by(|a, b| a.1.cmp(b.1));
    let longest = gpu_backends
        .iter()
        .map(|(name, _)| name.len())
        .max()
        .unwrap_or(0);

    for (name, status) in gpu_backends {
        println!("  * {:longest$} - {}", name, feature_status(status));
    }

    println!("{}", "Devices:".blue().bold());
    println!();

    let devices = list_devices();

    if devices.is_empty() {
        warn!("{} No devices found.", "Info:".blue().bold());
        warn!("{} To enable GPU support, ensure:", "Info:".blue().bold());
        warn!("  • Vulkan drivers are installed");
        warn!("  • Compatible GPU hardware is available");
        warn!("  • Vulkan feature is enabled (use --features vulkan)");
    } else {
        for device in devices {
            println!(
                "{} - {} {} {}",
                format_args!("Device {}", device.id.bold()).green(),
                device.name.bold(),
                if device.description.is_empty() {
                    String::new()
                } else {
                    device.description
                },
                match device.tpe {
                    purr_core::dev::DeviceType::Cpu =>
                        format_args!("({})", "CPU".green()).dimmed().to_string(),
                    purr_core::dev::DeviceType::Gpu =>
                        format_args!("({})", "GPU".blue()).dimmed().to_string(),
                    purr_core::dev::DeviceType::Accel =>
                        format_args!("({})", "Accel".yellow()).dimmed().to_string(),
                    purr_core::dev::DeviceType::Unknown =>
                        format_args!("({})", "Unknown").dimmed().to_string(),
                },
            );
            if device.vram_total != 0 {
                if device.vram_free != 0 && device.vram_free <= device.vram_total {
                    println!(
                        "    VRAM available: {}/{}",
                        format_file_size(device.vram_free as u64).green(),
                        format_file_size(device.vram_total as u64).green(),
                    );
                } else {
                    println!(
                        "    VRAM total: {}",
                        format_file_size(device.vram_total as u64).green()
                    );
                }
            }
            println!();
        }
    }

    Ok(())
}

/// Format file size in human readable format
fn format_file_size(size: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = size as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", size as u64, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
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

/// Print grouped model information with quantized variants
fn print_model_groups() {
    // Define model groups with their base models and quantized variants
    let model_groups = vec![
        ModelGroup {
            name: "tiny",
            description: "fastest, lowest accuracy",
            size: "39 MB",
            base_models: vec![WhisperModel::Tiny, WhisperModel::TinyEn],
            quantized: vec![
                WhisperModel::TinyQ5_1,
                WhisperModel::TinyEnQ5_1,
                WhisperModel::TinyQ8_0,
            ],
        },
        ModelGroup {
            name: "base",
            description: "good balance of speed and accuracy (recommended)",
            size: "142 MB",
            base_models: vec![WhisperModel::Base, WhisperModel::BaseEn],
            quantized: vec![
                WhisperModel::BaseQ5_1,
                WhisperModel::BaseEnQ5_1,
                WhisperModel::BaseQ8_0,
            ],
        },
        ModelGroup {
            name: "small",
            description: "good accuracy",
            size: "466 MB",
            base_models: vec![
                WhisperModel::Small,
                WhisperModel::SmallEn,
                WhisperModel::SmallEnTdrz,
            ],
            quantized: vec![
                WhisperModel::SmallQ5_1,
                WhisperModel::SmallEnQ5_1,
                WhisperModel::SmallQ8_0,
            ],
        },
        ModelGroup {
            name: "medium",
            description: "high accuracy",
            size: "1.5 GB",
            base_models: vec![WhisperModel::Medium, WhisperModel::MediumEn],
            quantized: vec![
                WhisperModel::MediumQ5_0,
                WhisperModel::MediumEnQ5_0,
                WhisperModel::MediumQ8_0,
            ],
        },
        ModelGroup {
            name: "large-v3",
            description: "most accurate",
            size: "3.0 GB",
            base_models: vec![WhisperModel::LargeV3],
            quantized: vec![WhisperModel::LargeV3Q5_0],
        },
        ModelGroup {
            name: "large-v3-turbo",
            description: "faster large model",
            size: "1.5 GB",
            base_models: vec![WhisperModel::LargeV3Turbo],
            quantized: vec![
                WhisperModel::LargeV3TurboQ5_0,
                WhisperModel::LargeV3TurboQ8_0,
            ],
        },
    ];

    for group in model_groups {
        // Print main model line
        let quantized_info = if group.quantized.is_empty() {
            String::new()
        } else {
            let mut quantized_types = std::collections::HashSet::new();
            for model in &group.quantized {
                let name = model.as_str();
                // Extract quantization type (q5_0, q5_1, q8_0)
                if let Some(q_part) = name.split('-').next_back() {
                    if q_part.starts_with('q') {
                        quantized_types.insert(q_part.to_string());
                    }
                }
            }
            let mut sorted_types: Vec<String> = quantized_types.into_iter().collect();
            sorted_types.sort();
            format!(" [quantized: {}]", sorted_types.join(", "))
        };

        println!(
            "  {} - {}, {}{}",
            group.name.green().bold(),
            group.description,
            group.size.yellow(),
            quantized_info.dimmed()
        );

        // Show base models (non-quantized variants) only if they're different from the group name
        let mut shown_variants = false;
        for model in &group.base_models {
            let name = model.as_str();
            if name != group.name {
                if !shown_variants {
                    shown_variants = true;
                }
                if name.contains(".en") && name.contains("tdrz") {
                    println!(
                        "    {} - speaker diarization, {}",
                        name.cyan(),
                        group.size.yellow()
                    );
                } else if name.contains(".en") {
                    println!(
                        "    {} - English-only, {}",
                        name.cyan(),
                        group.size.yellow()
                    );
                } else if name.contains("tdrz") {
                    println!(
                        "    {} - speaker diarization, {}",
                        name.cyan(),
                        group.size.yellow()
                    );
                } else {
                    println!("    {} - {}", name.cyan(), group.size.yellow());
                }
            }
        }

        // Add newline for spacing
        println!();
    }
}

async fn setup_config(cli: &Cli) -> miette::Result<TranscriptionConfig> {
    // Build transcription config
    let mut config = TranscriptionConfig::new();

    config = setup_model_config(cli, config).await?;

    config = config.with_cli_options(cli);

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
                let model = WhisperModel::from_str(model_string)?;
                if model_manager.is_model_downloaded(model).await {
                    config = config.with_model_path(model_manager.get_model_path(model));
                } else {
                    // If not downloaded, prompt user to download
                    if let Some(model) = prompt_for_model_download(Some(model)).await? {
                        model_manager.assign_model_path(&mut config, model);
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
        if let Some(model) = model_manager.find_first_available_model().await {
            config = config.with_model_path(model);
        } else {
            // Prompt user to download the base model
            if let Some(model) = prompt_for_model_download(None).await? {
                // If user agrees, download the base model
                model_manager.assign_model_path(&mut config, model);
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
        OutputFormat::Json => serde_json::to_string_pretty(&result).into_diagnostic()?,
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
    }

    Ok(())
}

/// Helper struct for organizing model information
struct ModelGroup {
    name: &'static str,
    description: &'static str,
    size: &'static str,
    base_models: Vec<WhisperModel>,
    quantized: Vec<WhisperModel>,
}
