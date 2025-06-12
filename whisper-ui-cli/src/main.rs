//! Whisper UI CLI - Audio transcription command-line interface

use clap::{Parser, Subcommand};
use colored::*;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process;
use whisper_ui_core::{transcribe_audio_file, TranscriptionConfig, ModelManager, WhisperModel};

/// Audio transcription CLI using Whisper
#[derive(Parser)]
#[command(name = "whisper-ui")]
#[command(about = "Transcribe audio files using Whisper AI")]
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

    /// Output format: text, json, srt
    #[arg(short, long, default_value = "text")]
    output: OutputFormat,

    /// Include timestamps in output (text format only)
    #[arg(long)]
    timestamps: bool,

    /// Include word-level timestamps (if supported)
    #[arg(long)]
    word_timestamps: bool,

    /// Temperature for sampling (0.0 = deterministic)
    #[arg(long, default_value = "0.0")]
    temperature: f32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Model management commands
    Models {
        #[command(subcommand)]
        command: ModelCommands,
    },
}

#[derive(Subcommand)]
enum ModelCommands {
    /// Download a Whisper model
    Download {
        /// Model to download (e.g., base, small, large-v3)
        #[arg(value_name = "MODEL")]
        model: String,
    },
    /// List available models
    List,
    /// List downloaded models
    Downloaded,
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

/// Output format options
#[derive(Clone, Debug, clap::ValueEnum)]
enum OutputFormat {
    /// Plain text output
    Text,
    /// JSON output with metadata
    Json,
    /// SRT subtitle format
    Srt,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Setup logging
    if cli.verbose {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();
    }

    // Handle subcommands
    if let Some(command) = cli.command {
        return handle_command(command, cli.verbose).await;
    }

    // Handle transcription (original behavior)
    let audio_file = cli.audio_file.ok_or("Audio file is required when no subcommand is provided")?;
    
    // Validate audio file exists
    if !audio_file.exists() {
        eprintln!("{} Audio file not found: {}", 
                 "Error:".red().bold(), 
                 audio_file.display());
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

    if let Some(threads) = cli.threads {
        config = config.with_threads(threads);
    }

    config.temperature = cli.temperature;
    config.output_format.include_timestamps = cli.timestamps;
    config.output_format.word_timestamps = cli.word_timestamps;

    // Print startup info
    if cli.verbose {
        println!("{}", "Whisper UI - Audio Transcription".blue().bold());
        println!("Audio file: {}", audio_file.display());
        println!("GPU acceleration: {}", 
                if config.use_gpu { "enabled".green() } else { "disabled".red() });
        if let Some(lang) = &config.language {
            println!("Language: {}", lang);
        }
        println!();
    }

    // Perform transcription
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

                    if let Some(threads) = cli.threads {
                        config = config.with_threads(threads);
                    }

                    config.temperature = cli.temperature;
                    config.output_format.include_timestamps = cli.timestamps;
                    config.output_format.word_timestamps = cli.word_timestamps;

                    println!("{} Transcribing audio...", "Info:".blue().bold());
                    match transcribe_audio_file(&audio_file, Some(config)).await {
                        Ok(result) => result,
                        Err(e) => {
                            eprintln!("{} Transcription failed: {}", "Error:".red().bold(), e);
                            process::exit(1);
                        }
                    }
                } else {
                    eprintln!("{} No model available. Download one with: whisper-ui models download base", 
                             "Error:".red().bold());
                    process::exit(1);
                }
            } else {
                eprintln!("{} Transcription failed: {}", "Error:".red().bold(), e);
                process::exit(1);
            }
        }
    };

    // Output results
    match cli.output {
        OutputFormat::Text => {
            if cli.timestamps {
                for segment in &result.segments {
                    println!("[{:.2}s -> {:.2}s] {}", 
                            segment.start, segment.end, segment.text);
                }
            } else {
                println!("{}", result.text);
            }
        },
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&result)?;
            println!("{}", json);
        },
        OutputFormat::Srt => {
            for (i, segment) in result.segments.iter().enumerate() {
                println!("{}", i + 1);
                println!("{} --> {}", 
                        format_srt_time(segment.start), 
                        format_srt_time(segment.end));
                println!("{}\n", segment.text);
            }
        },
    }

    // Print summary if verbose
    if cli.verbose {
        println!();
        println!("{}", "Transcription Summary:".green().bold());
        println!("Audio duration: {:.2}s", result.audio_duration);
        println!("Processing time: {:.2}s", result.processing_time);
        println!("Real-time factor: {:.2}x", 
                result.processing_time / result.audio_duration as f64);
        if let Some(lang) = &result.language {
            println!("Detected language: {}", lang);
        }
        println!("Segments: {}", result.segments.len());
    }

    Ok(())
}

/// Prompt user to download base model when none is found
async fn prompt_for_model_download() -> Result<bool, Box<dyn std::error::Error>> {
    println!();
    println!("{} No Whisper model found!", "Notice:".yellow().bold());
    println!("To transcribe audio, you need to download a Whisper model first.");
    println!();
    println!("The {} model is recommended for most users:", "base".green().bold());
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
                println!("{} Model downloaded successfully to: {}", 
                        "Success:".green().bold(), 
                        model_path.display());
                println!();
                return Ok(true);
            },
            Err(e) => {
                eprintln!("{} Failed to download model: {}", 
                         "Error:".red().bold(), e);
                return Ok(false);
            }
        }
    } else {
        println!();
        println!("Model download cancelled. You can download a model later with:");
        println!("  {}", "whisper-ui models download base".cyan());
        return Ok(false);
    }
}

/// Handle subcommands
async fn handle_command(command: Commands, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Models { command } => {
            handle_model_command(command, verbose).await
        }
    }
}

/// Handle model management subcommands
async fn handle_model_command(command: ModelCommands, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    let model_manager = ModelManager::new()?;

    match command {
        ModelCommands::Download { model } => {
            let whisper_model = WhisperModel::from_str(&model)
                .ok_or_else(|| format!("Unknown model: {}. Use 'models list' to see available models.", model))?;
            
            println!("{} Downloading model: {} ({})", 
                    "Info:".blue().bold(), 
                    whisper_model.as_str(), 
                    whisper_model.description());
            
            let model_path = model_manager.download_model(whisper_model).await?;
            
            println!("{} Model downloaded successfully to: {}", 
                    "Success:".green().bold(), 
                    model_path.display());
        },
        
        ModelCommands::List => {
            println!("{}", "Available Whisper Models:".blue().bold());
            println!();
            
            let models = WhisperModel::all_models();
            let mut current_family = "";
            
            for model in models {
                let family = model.as_str().split('-').next().unwrap_or("");
                if family != current_family {
                    if !current_family.is_empty() {
                        println!();
                    }
                    println!("{} models:", family.to_uppercase().yellow().bold());
                    current_family = family;
                }
                
                println!("  {} - {}", 
                        model.as_str().green(), 
                        model.description().dimmed());
            }
            
            println!();
            println!("{}", "Usage: whisper-ui models download <model>".dimmed());
        },
        
        ModelCommands::Downloaded => {
            let downloaded = model_manager.list_downloaded_models().await?;
            
            if downloaded.is_empty() {
                println!("{} No models downloaded yet.", "Info:".blue().bold());
                println!("Use 'whisper-ui models download <model>' to download a model.");
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
                    
                    println!("  {} - {} ({})", 
                            model.as_str().green(), 
                            model.description().dimmed(),
                            size.yellow());
                    
                    if verbose {
                        println!("    Path: {}", path.display().to_string().dimmed());
                    }
                }
                
                println!();
                println!("XDG data directory: {}", 
                        model_manager.models_dir().display().to_string().dimmed());
            }
        },
        
        ModelCommands::Delete { model } => {
            let whisper_model = WhisperModel::from_str(&model)
                .ok_or_else(|| format!("Unknown model: {}. Use 'models downloaded' to see available models.", model))?;
            
            if !model_manager.is_model_downloaded(&whisper_model).await {
                println!("{} Model {} is not downloaded.", 
                        "Warning:".yellow().bold(), 
                        whisper_model.as_str());
                return Ok(());
            }
            
            model_manager.delete_model(&whisper_model).await?;
            
            println!("{} Model {} deleted successfully.", 
                    "Success:".green().bold(), 
                    whisper_model.as_str());
        },
        
        ModelCommands::Info { model } => {
            let whisper_model = WhisperModel::from_str(&model)
                .ok_or_else(|| format!("Unknown model: {}. Use 'models list' to see available models.", model))?;
            
            println!("{} Model Information", "Info:".blue().bold());
            println!();
            println!("Name: {}", whisper_model.as_str().green().bold());
            println!("Description: {}", whisper_model.description());
            println!("Filename: {}", whisper_model.filename().yellow());
            
            let is_downloaded = model_manager.is_model_downloaded(&whisper_model).await;
            println!("Downloaded: {}", 
                    if is_downloaded { 
                        "Yes".green().bold() 
                    } else { 
                        "No".red().bold() 
                    });
            
            if is_downloaded {
                let path = model_manager.get_model_path(&whisper_model);
                println!("Path: {}", path.display());
                
                if let Ok(metadata) = std::fs::metadata(&path) {
                    println!("Size: {}", format_file_size(metadata.len()).yellow());
                }
            }
        },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srt_time_formatting() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(61.5), "00:01:01,500");
        assert_eq!(format_srt_time(3661.123), "01:01:01,123");
    }

    #[test]
    fn test_cli_parsing() {
        use clap::Parser;
        
        let cli = Cli::try_parse_from(&[
            "whisper-ui",
            "test.wav",
            "--model", "model.bin",
            "--language", "en",
            "--output", "json"
        ]).unwrap();
        
        assert_eq!(cli.audio_file, Some(PathBuf::from("test.wav")));
        assert_eq!(cli.model, Some(PathBuf::from("model.bin")));
        assert_eq!(cli.language, Some("en".to_string()));
        assert!(matches!(cli.output, OutputFormat::Json));
    }
}
