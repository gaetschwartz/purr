//! Whisper UI CLI - Audio transcription command-line interface
mod fmt;

use clap::{Parser, Subcommand};
use indicatif::{HumanBytes, HumanDuration, ProgressBar, ProgressStyle};
use owo_colors::OwoColorize as _;
use purr_core::{
    check_gpu_status, list_devices, transcribe_audio_file, transcribe_audio_file_streaming,
    ModelManager, TranscriptionConfig, WhisperError, WhisperModel,
};
use std::io::{self, Write};
use std::path::PathBuf;
use std::process;
use std::str::FromStr as _;
use tracing::{debug, error, info};
use tracing_subscriber::EnvFilter;

use crate::fmt::MyFormatter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Setup logging
    if cli.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_writer(std::io::stderr)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_env_filter(
                EnvFilter::builder().parse("info,whisper_rs::whisper_logging_hook=warn")?,
            )
            .compact()
            .without_time()
            .with_file(false)
            .with_line_number(false)
            .with_thread_ids(false)
            .with_thread_names(false)
            .with_target(false)
            .event_format(MyFormatter)
            .with_writer(std::io::stderr)
            .init();
    }
    debug!("Command line arguments: {:?}", cli);

    // Handle subcommands
    if let Some(command) = cli.command {
        return handle_command(command, cli.verbose).await;
    }

    // Handle transcription (original behavior)
    let Some(audio_file) = cli.audio_file.clone() else {
        println!("{}\n", ASCII_ART);
        error!("No audio file specified. Please provide an audio file to transcribe.",);
        std::process::exit(1);
    };

    // Validate audio file exists
    if !audio_file.exists() {
        error!("Audio file not found: {}", audio_file.display());
        process::exit(1);
    }

    // Build transcription config
    let mut config = TranscriptionConfig::new()
        .with_gpu(!cli.no_gpu)
        .with_sample_rate(16000); // Whisper's preferred sample rate

    if let Some(ref model_path) = cli.model {
        config = config.with_model_path(model_path.clone());
    }

    if let Some(ref language) = cli.language {
        config = config.with_language(language.clone());
    }

    config = config.with_threads(cli.threads.unwrap_or_else(num_cpus::get));

    config.temperature = cli.temperature;
    config.output_format.include_timestamps = cli.timestamps;
    config.output_format.word_timestamps = cli.word_timestamps;
    config = config.with_verbose(cli.verbose);

    // Print startup info
    if cli.verbose {
        println!("{}", "Whisper UI - Audio Transcription".blue().bold());
        if config.use_gpu {
            println!("GPU acceleration: {}", "enabled".green());
        } else {
            println!("GPU acceleration: {}", "disabled".red());
        }
        if let Some(lang) = &config.language {
            println!("Language: {}", lang);
        }
        println!();
    }

    // Perform transcription (streaming or batch)
    if !cli.no_stream {
        info!("Streaming transcription...");

        // Handle streaming transcription
        let mut receiver = match transcribe_audio_file_streaming(&audio_file, Some(config)).await {
            Ok(receiver) => receiver,
            Err(e) => {
                // Check if this is a "no model found" error

                if matches!(e, WhisperError::WhisperModel(_)) {
                    // Prompt user to download base model
                    if prompt_for_model_download().await? {
                        // Retry streaming transcription after downloading model
                        let mut config = TranscriptionConfig::new()
                            .with_gpu(!cli.no_gpu)
                            .with_sample_rate(16000);

                        if let Some(ref model_path) = cli.model {
                            config = config.with_model_path(model_path.clone());
                        }

                        if let Some(ref language) = cli.language {
                            config = config.with_language(language.clone());
                        }

                        config = config.with_threads(cli.threads.unwrap_or_else(num_cpus::get));

                        config.temperature = cli.temperature;
                        config.output_format.include_timestamps = cli.timestamps;
                        config.output_format.word_timestamps = cli.word_timestamps;
                        config = config.with_verbose(cli.verbose);

                        println!("{} Streaming transcription...", "Info:".blue().bold());
                        match transcribe_audio_file_streaming(&audio_file, Some(config)).await {
                            Ok(receiver) => receiver,
                            Err(e) => {
                                error!("Streaming transcription failed: {}", e);
                                process::exit(1);
                            }
                        }
                    } else {
                        error!(
                            "No model available. Download one with: {}{}",
                            env!("CARGO_PKG_NAME").cyan(),
                            " models download base".cyan()
                        );
                        process::exit(1);
                    }
                } else {
                    error!("Streaming transcription failed: {}", e);
                    process::exit(1);
                }
            }
        };

        // Process streaming results
        handle_streaming_output(&mut receiver, &cli).await?;

        return Ok(());
    }

    println!("{} Transcribing audio...", "Info:".blue().bold());

    let result = match transcribe_audio_file(&audio_file, Some(config)).await {
        Ok(result) => result,
        Err(e) => {
            // Check if this is a "no model found" error
            let error_str = e.to_string();
            if error_str.contains("No Whisper model found") {
                // Prompt user to download base model
                if prompt_for_model_download().await? {
                    // Retry transcription after downloading model
                    let mut config = TranscriptionConfig::new()
                        .with_gpu(!cli.no_gpu)
                        .with_sample_rate(16000);

                    if let Some(ref model_path) = cli.model {
                        config = config.with_model_path(model_path.clone());
                    }

                    if let Some(ref language) = cli.language {
                        config = config.with_language(language.clone());
                    }

                    config = config.with_threads(cli.threads.unwrap_or_else(num_cpus::get));

                    config.temperature = cli.temperature;
                    config.output_format.include_timestamps = cli.timestamps;
                    config.output_format.word_timestamps = cli.word_timestamps;
                    config = config.with_verbose(cli.verbose);

                    println!("{} Transcribing audio...", "Info:".blue().bold());
                    match transcribe_audio_file(&audio_file, Some(config)).await {
                        Ok(result) => result,
                        Err(e) => {
                            error!("Transcription failed: {}", e);
                            process::exit(1);
                        }
                    }
                } else {
                    error!(
                        "No model available. Download one with: {}{}",
                        env!("CARGO_PKG_NAME").cyan(),
                        " models download base".cyan()
                    );
                    process::exit(1);
                }
            } else {
                error!("Transcription failed: {}", e);
                process::exit(1);
            }
        }
    };

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
        OutputFormat::Json => serde_json::to_string_pretty(&result)?,
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
    if let Some(output_file) = cli.output_file {
        use std::fs;
        fs::write(&output_file, &output_content)?;
        if cli.verbose {
            println!(
                "{} Output written to: {}",
                "Success:".green().bold(),
                output_file.display()
            );
        }
    } else {
        print!("{}", output_content);
    }

    // Print summary if verbose
    if cli.verbose {
        println!();
        println!("{}", "Transcription Summary:".green().bold());
        println!("Audio duration: {:.2}s", result.audio_duration);
        println!("Processing time: {:.2}s", result.processing_time);
        println!(
            "Real-time factor: {:.2}x",
            result.processing_time / result.audio_duration as f64
        );
        if let Some(lang) = &result.language {
            println!("Detected language: {}", lang);
        }
        println!("Segments: {}", result.segments.len());
    }

    Ok(())
}

const ABOUT: &str = "😸 Transcribe audio files using Whisper AI";
#[derive(Parser, Debug)]
#[command(name = env!("CARGO_PKG_NAME"), author = env!("CARGO_PKG_AUTHORS"))]
#[command(about = ABOUT)]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to the audio file to transcribe (when no subcommand)
    #[arg(value_name = "AUDIO_FILE")]
    audio_file: Option<PathBuf>,

    /// Path to the Whisper model file
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Language code (e.g., en, es, fr). Auto-detect if not specified
    #[arg(short, long)]
    language: Option<String>,

    /// Disable GPU acceleration
    #[arg(long)]
    no_gpu: bool,

    /// Number of threads to use
    #[arg(short, long)]
    threads: Option<usize>,

    /// Output format: text, json, srt, txt
    #[arg(short, long, default_value = "text")]
    output: OutputFormat,

    /// Output file path (writes to file instead of stdout)
    #[arg(short = 'f', long = "output-file")]
    output_file: Option<PathBuf>,

    /// Include timestamps in output (text format only)
    #[arg(long)]
    timestamps: bool,

    /// Include word-level timestamps (if supported)
    #[arg(long)]
    word_timestamps: bool,

    /// Stream transcription results in real-time
    #[arg(short = 'S', long)]
    no_stream: bool,

    /// Temperature for sampling (0.0 = deterministic)
    #[arg(long, default_value = "0.0")]
    temperature: f32,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Model management commands
    Models {
        #[command(subcommand)]
        command: ModelCommands,
    },
    /// GPU acceleration commands
    #[clap(alias = "dev")]
    Device {
        #[command(subcommand)]
        command: GpuCommands,
    },
}

#[derive(Subcommand, Debug)]
enum ModelCommands {
    /// Download a Whisper model
    Download {
        /// Model to download (e.g., base, small, large-v3)
        #[arg(value_name = "MODEL")]
        model: String,

        /// Force download even if the model is already downloaded
        #[arg(short, long)]
        force: bool,
    },
    /// List models (downloaded by default, use --available to list all available models)
    List {
        /// List all available models instead of downloaded models
        #[arg(short, long)]
        available: bool,
    },
    /// Delete a downloaded model
    Delete {
        /// Model to delete (e.g., base, small, large-v3)
        #[arg(value_name = "MODEL")]
        model: String,
    },
    /// Show model information
    Info {
        /// Model to show info for (e.g., base, small, large-v3)
        #[arg(value_name = "MODEL")]
        model: String,
    },
}

#[derive(Subcommand, Debug)]
enum GpuCommands {
    /// List GPU devices available for acceleration
    List,
    /// Check GPU acceleration status
    Status,
}

/// Output format options
#[derive(Clone, Debug, clap::ValueEnum)]
enum OutputFormat {
    /// Plain text output with optional timestamps
    Text,
    /// JSON output with metadata
    Json,
    /// SRT subtitle format
    Srt,
    /// Plain text output (clean, no timestamps)
    Txt,
}

/// Handle streaming transcription output
async fn handle_streaming_output(
    receiver: &mut purr_core::StreamingReceiver,
    cli: &Cli,
) -> anyhow::Result<()> {
    use std::fs;

    let mut all_chunks = Vec::new();
    let mut output_buffer = String::new();
    let mut stdout = io::stdout();

    while let Some(chunk) = receiver.recv().await {
        all_chunks.push(chunk.clone());

        // Format the chunk for real-time output
        let chunk_text = match cli.output {
            OutputFormat::Text => {
                if cli.timestamps {
                    format!("[{:.2}s -> {:.2}s] {}", chunk.start, chunk.end, chunk.text)
                } else {
                    chunk.text.clone()
                }
            }
            OutputFormat::Json => serde_json::to_string(&chunk)?,
            OutputFormat::Srt => {
                format!(
                    "{}\n{} --> {}\n{}\n",
                    chunk.chunk_index + 1,
                    format_srt_time(chunk.start),
                    format_srt_time(chunk.end),
                    chunk.text
                )
            }
            OutputFormat::Txt => chunk.text.clone(),
        };

        // Print to stdout or accumulate for file output
        if cli.output_file.is_some() {
            output_buffer.push_str(&chunk_text);
            if !matches!(cli.output, OutputFormat::Txt) {
                output_buffer.push('\n');
            }
        } else {
            // IMMEDIATE real-time output to stdout
            if matches!(cli.output, OutputFormat::Json) {
                write!(stdout, "{}", chunk_text)?;
            } else {
                write!(stdout, "{}", chunk_text)?;
                if !chunk.text.is_empty() && !chunk.text.ends_with('\n') {
                    if matches!(cli.output, OutputFormat::Srt) {
                        writeln!(stdout)?;
                    } else {
                        write!(stdout, " ")?;
                    }
                }
                // CRITICAL: Flush immediately to show real-time output
                stdout.flush()?;
            }
        }
    }

    // Write to file if specified
    if let Some(output_file) = &cli.output_file {
        fs::write(output_file, &output_buffer)?;
        if cli.verbose {
            info!(
                "\n{} Streaming output written to: {}",
                "Success:".green().bold(),
                output_file.display()
            );
        }
    } else {
        println!(); // Final newline for stdout
    }

    if cli.verbose {
        debug!("Processed {} chunks", all_chunks.len());
    }

    Ok(())
}

/// Prompt user to download base model when none is found
async fn prompt_for_model_download() -> anyhow::Result<bool> {
    println!();
    println!("{} No Whisper model found!", "Notice:".yellow().bold());
    println!("To transcribe audio, you need to download a Whisper model first.");
    println!();
    println!(
        "The {} model is recommended for most users:",
        "base".green().bold()
    );
    println!("  • Good balance of speed and accuracy");
    println!("  • Size: ~142 MB");
    println!("  • Download time: ~1-2 minutes");
    println!();

    print!("Would you like to download the base model now? [Y/n]: ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim().to_lowercase();

    // Default to 'yes' if user just presses enter
    let should_download = input.is_empty() || input == "y" || input == "yes";

    if should_download {
        println!();
        println!("{} Downloading base model...", "Info:".blue().bold());

        let model_manager = ModelManager::new()?;
        match model_manager.download_model(WhisperModel::Base).await {
            Ok(model_path) => {
                println!(
                    "{} Model downloaded successfully to: {}",
                    "Success:".green().bold(),
                    model_path.display()
                );
                println!();
                Ok(true)
            }
            Err(e) => {
                error!("Failed to download model: {}", e);
                Ok(false)
            }
        }
    } else {
        println!();
        println!("Model download cancelled. You can download a model later with:");
        println!(
            "  {}{}",
            env!("CARGO_PKG_NAME").cyan(),
            " models download base".cyan()
        );
        Ok(false)
    }
}

/// Handle subcommands
async fn handle_command(command: Commands, verbose: bool) -> anyhow::Result<()> {
    match command {
        Commands::Models { command } => handle_model_command(command, verbose).await,
        Commands::Device { command } => handle_devices_command(command, verbose).await,
    }
}

/// Handle model management subcommands
async fn handle_model_command(command: ModelCommands, verbose: bool) -> anyhow::Result<()> {
    let model_manager = ModelManager::new()?;

    match command {
        ModelCommands::Download { model, force } => {
            let whisper_model = WhisperModel::from_str(&model).map_err(|e| {
                anyhow::anyhow!(
                    "Unknown model: {}. Use 'models list' to see available models. Error: {}",
                    model,
                    e
                )
            })?;

            // check if it is already downloaded
            if !force && model_manager.is_model_downloaded(&whisper_model).await {
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
                    "".to_string()
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
                        let path = model_manager.get_model_path(&model);
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

                        if verbose {
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
                anyhow::anyhow!(
                    "Unknown model: {}. Use 'models list' to see available models. Error: {}",
                    model,
                    e
                )
            })?;

            if !model_manager.is_model_downloaded(&whisper_model).await {
                println!(
                    "{} Model {} is not downloaded.",
                    "Warning:".yellow().bold(),
                    whisper_model.as_str()
                );
                return Ok(());
            }

            model_manager.delete_model(&whisper_model).await?;

            println!(
                "{} Model {} deleted successfully.",
                "Success:".green().bold(),
                whisper_model.as_str()
            );
        }

        ModelCommands::Info { model } => {
            let whisper_model = WhisperModel::from_str(&model).map_err(|e| {
                anyhow::anyhow!(
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

            let is_downloaded = model_manager.is_model_downloaded(&whisper_model).await;
            println!(
                "Downloaded: {}",
                if is_downloaded {
                    "yes".green().to_string()
                } else {
                    "no".red().to_string()
                }
            );

            if is_downloaded {
                let path = model_manager.get_model_path(&whisper_model);
                println!("Path: {}", path.display());

                if let Ok(metadata) = std::fs::metadata(&path) {
                    println!("Size: {}", format_file_size(metadata.len()).yellow());
                }
            }
        }
    }

    Ok(())
}

/// Handle GPU acceleration subcommands
async fn handle_devices_command(command: GpuCommands, verbose: bool) -> anyhow::Result<()> {
    match command {
        GpuCommands::List => {
            println!("{}", "Devices:".blue().bold());
            println!();

            let devices = list_devices();

            if devices.is_empty() {
                println!(
                    "{} No GPU devices found or Vulkan support not available.",
                    "Info:".blue().bold()
                );
                println!("To enable GPU support, ensure:");
                println!("  • Vulkan drivers are installed");
                println!("  • Compatible GPU hardware is available");
                println!("  • Vulkan feature is enabled (use --features vulkan)");
            } else {
                for device in devices {
                    println!(
                        "{} - {} {} {}",
                        format_args!("Device {}", device.id.bold()).green(),
                        device.name.bold(),
                        if device.description.is_empty() {
                            "".to_string()
                        } else {
                            format_args!("{}", device.description).to_string()
                        },
                        match device.tpe {
                            purr_core::dev::DeviceType::Cpu =>
                                format_args!("({})", "CPU".green()).dimmed().to_string(),
                            purr_core::dev::DeviceType::Gpu =>
                                format_args!("({})", "GPU".blue()).dimmed().to_string(),
                            purr_core::dev::DeviceType::Accel =>
                                format_args!("({})", "Accel".yellow()).dimmed().to_string(),
                        },
                    );
                    if device.vram_total != 0 {
                        if verbose {
                            println!(
                                "    VRAM: {} / {} ({} free)",
                                format_file_size(
                                    device.vram_total as u64 - device.vram_free as u64
                                )
                                .yellow(),
                                format_file_size(device.vram_total as u64).yellow(),
                                format_file_size(device.vram_free as u64).green()
                            );
                        } else {
                            println!(
                                "    VRAM: {}",
                                format_file_size(device.vram_total as u64).yellow()
                            );
                        }
                    }
                    println!();
                }
            }
        }

        GpuCommands::Status => {
            let status = check_gpu_status();

            println!("{}", "GPU Acceleration Status:".blue().bold());
            println!();

            println!(
                "Vulkan support: {}",
                if status.vulkan_available {
                    "Available".green().bold().to_string()
                } else {
                    "Not available".red().bold().to_string()
                }
            );

            println!(
                "CUDA support: {}",
                if status.cuda_available {
                    "Available".green().bold().to_string()
                } else {
                    "Not available".red().bold().to_string()
                }
            );

            println!(
                "CoreML support: {}",
                if status.coreml_available {
                    "Available".green().bold().to_string()
                } else {
                    "Not available".red().bold().to_string()
                }
            );

            println!(
                "GPU devices: {}",
                if status.devices.is_empty() {
                    "None detected".red().bold().to_string()
                } else {
                    format!("{} found", status.devices.len())
                        .green()
                        .bold()
                        .to_string()
                }
            );

            if verbose && !status.devices.is_empty() {
                println!();
                println!("Detected devices:");
                for device in status.devices {
                    println!("  • {} (ID: {})", device.name, device.id);
                }
            }

            println!();
            if status.vulkan_available || status.cuda_available || status.coreml_available {
                println!(
                    "{} GPU acceleration is available for faster transcription.",
                    "Success:".green().bold()
                );
            } else {
                println!(
                    "{} GPU acceleration is not available. Transcription will use CPU.",
                    "Warning:".yellow().bold()
                );
            }
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

    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, secs, millis)
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

/// Helper struct for organizing model information
struct ModelGroup {
    name: &'static str,
    description: &'static str,
    size: &'static str,
    base_models: Vec<WhisperModel>,
    quantized: Vec<WhisperModel>,
}

const ASCII_ART: &str = r#"
             *     ,MMM8&&&.            *
                  MMMM88&&&&&    .
                 MMMM88&&&&&&&
     *           MMM88&&&&&&&&
                 MMM88&&&&&&&&
                 'MMM88&&&&&&'
                   'MMM8&&&'      *
          |\___/|
          )     (             .              '
         =\     /=
           )===(       *
          /     \
          |     |
         /       \
         \       /
  _/\_/\_/\__  _/_/\_/\_/\_/\_/\_/\_/\_/\_/\_
  |  |  |  |( (  |  |  |  |  |  |  |  |  |  |
  |  |  |  | ) ) |  |  |  |  |  |  |  |  |  |
  |  |  |  |(_(  |  |  |  |  |  |  |  |  |  |
  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srt_time_formatting() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(61.5), "00:01:01,500");
        assert_eq!(format_srt_time(3661.123), "01:01:01,123");
    }
}
