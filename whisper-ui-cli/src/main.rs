//! Whisper UI CLI - Audio transcription command-line interface

use clap::Parser;
use colored::*;
use std::path::PathBuf;
use std::process;
use whisper_ui_core::{transcribe_audio_file, TranscriptionConfig};

/// Audio transcription CLI using Whisper
#[derive(Parser)]
#[command(name = "whisper-ui")]
#[command(about = "Transcribe audio files using Whisper AI")]
#[command(version = "0.1.0")]
struct Cli {
    /// Path to the audio file to transcribe
    #[arg(value_name = "AUDIO_FILE")]
    audio_file: PathBuf,

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

    // Validate audio file exists
    if !cli.audio_file.exists() {
        eprintln!("{} Audio file not found: {}", 
                 "Error:".red().bold(), 
                 cli.audio_file.display());
        process::exit(1);
    }

    // Build transcription config
    let mut config = TranscriptionConfig::new()
        .with_gpu(!cli.no_gpu)
        .with_sample_rate(16000); // Whisper's preferred sample rate

    if let Some(model_path) = cli.model {
        config = config.with_model_path(model_path);
    }

    if let Some(language) = cli.language {
        config = config.with_language(language);
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
        println!("Audio file: {}", cli.audio_file.display());
        println!("GPU acceleration: {}", 
                if config.use_gpu { "enabled".green() } else { "disabled".red() });
        if let Some(lang) = &config.language {
            println!("Language: {}", lang);
        }
        println!();
    }

    // Perform transcription
    println!("{} Transcribing audio...", "Info:".blue().bold());
    
    let result = match transcribe_audio_file(&cli.audio_file, Some(config)).await {
        Ok(result) => result,
        Err(e) => {
            eprintln!("{} Transcription failed: {}", "Error:".red().bold(), e);
            process::exit(1);
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
        
        assert_eq!(cli.audio_file, PathBuf::from("test.wav"));
        assert_eq!(cli.model, Some(PathBuf::from("model.bin")));
        assert_eq!(cli.language, Some("en".to_string()));
        assert!(matches!(cli.output, OutputFormat::Json));
    }
}
