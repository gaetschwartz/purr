use crate::fmt::Verbosity;
use clap::builder::{
    styling::{AnsiColor, Effects, Style},
    Styles,
};
use clap::{Parser, Subcommand};
use purr_common::Language;
use std::path::PathBuf;

const ABOUT: &str = "😸 Transcribe audio files and URLs using Whisper AI";
#[derive(Parser, Debug)]
#[command(name = env!("CARGO_PKG_NAME"), author = env!("CARGO_PKG_AUTHORS"))]
#[command(about = ABOUT)]
#[command(version = "0.1.0")]
#[command(styles = CLAP_STYLING)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Path to the audio file or URL to transcribe (when no subcommand)
    #[arg(value_name = "AUDIO_FILE_OR_URL")]
    pub audio_input: Option<String>,

    /// Path to the Whisper model file
    #[arg(short, long)]
    pub model: Option<String>,

    /// Language code (e.g., en, es, fr). Auto-detect if not specified
    #[arg(short, long)]
    pub language: Option<Language>,

    /// Translate to English (like whisper.cpp --translate)
    #[arg(long)]
    pub translate: bool,

    /// Disable GPU acceleration
    #[arg(long)]
    pub no_gpu: bool,

    /// Number of threads to use
    #[arg(short, long)]
    pub threads: Option<usize>,

    /// Output format: text, json, srt, txt
    #[arg(short, long, default_value = "text")]
    pub output: OutputFormat,

    /// Output file path (writes to file instead of stdout)
    #[arg(short = 'f', long = "output-file")]
    pub output_file: Option<PathBuf>,

    /// Include timestamps in output (text format only)
    #[arg(long)]
    pub timestamps: bool,

    /// Include word-level timestamps (if supported)
    #[arg(long)]
    pub word_timestamps: bool,

    /// Stream transcription results in real-time
    #[arg(short = 'S', long)]
    pub no_stream: bool,

    /// Temperature for sampling (0.0 = deterministic)
    #[arg(long, default_value = "0.0")]
    pub temperature: f32,

    /// Verbose output
    #[clap(flatten)]
    pub verbosity: Verbosity,

    /// Sample rate to use in Hz. 16'000 is the recommended and default value.
    #[arg(long, default_value = "16000")]
    pub sample_rate: u32,

    /// Chunk size for streaming (in seconds)
    #[arg(long, default_value = "4.0")]
    pub chunk_size: f32,

    /// Chunk overlap for streaming (in seconds)
    #[arg(long, default_value = "0.5")]
    pub chunk_overlap: f32,

    /// HTTP timeout for URL requests (in seconds)
    #[arg(long, default_value = "30")]
    pub http_timeout: u64,

    /// HTTP connection timeout for URL requests (in seconds)
    #[arg(long, default_value = "10")]
    pub http_connect_timeout: u64,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Model management commands
    Models {
        #[command(subcommand)]
        command: ModelCommands,
    },
    /// System commands
    #[clap(alias = "s")]
    Sys {},
}

#[derive(Subcommand, Debug)]
pub enum ModelCommands {
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
pub enum SysCommands {
    /// List GPU devices available for acceleration
    Info,
}

/// Output format options
#[derive(Clone, Debug, clap::ValueEnum)]
pub enum OutputFormat {
    /// Plain text output with optional timestamps
    Text,
    /// JSON output with metadata
    Json,
    /// SRT subtitle format
    Srt,
    /// Plain text output (clean, no timestamps)
    Txt,
}

pub trait TranscriptionConfigWithCliExt {
    fn with_cli_options(self, cli: &Cli) -> Self;
}

impl TranscriptionConfigWithCliExt for purr_core::TranscriptionConfig {
    fn with_cli_options(self, cli: &Cli) -> Self {
        self.with_verbose(cli.verbosity.is_verbose())
            .with_opt_language(cli.language.as_ref().map(Language::code))
            .with_translate(cli.translate)
            .with_threads(cli.threads.unwrap_or_else(num_cpus::get))
            .with_temperature(cli.temperature)
            .with_sample_rate(cli.sample_rate)
            .with_chunk_overlap(cli.chunk_overlap)
            .with_chunk_size(cli.chunk_size)
            .without_gpu(cli.no_gpu)
            .with_http_timeout(cli.http_timeout)
            .with_http_connect_timeout(cli.http_connect_timeout)
            .apply_output_format(|f| {
                f.with_timestamps(cli.timestamps)
                    .with_word_timestamps(cli.word_timestamps)
            })
    }
}

pub const ASCII_ART: &str = r"
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
  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |";

pub const HEADER: Style = AnsiColor::Green.on_default().effects(Effects::BOLD);
pub const USAGE: Style = AnsiColor::Green.on_default().effects(Effects::BOLD);
pub const LITERAL: Style = AnsiColor::Cyan.on_default().effects(Effects::BOLD);
pub const PLACEHOLDER: Style = AnsiColor::Cyan.on_default().italic();
pub const ERROR: Style = AnsiColor::Red.on_default().effects(Effects::BOLD);
pub const VALID: Style = AnsiColor::Cyan.on_default().effects(Effects::BOLD);
pub const INVALID: Style = AnsiColor::Yellow.on_default().effects(Effects::BOLD);

const CLAP_STYLING: Styles = Styles::styled()
    .header(HEADER)
    .usage(USAGE)
    .literal(LITERAL)
    .placeholder(PLACEHOLDER)
    .error(ERROR)
    .valid(VALID)
    .invalid(INVALID);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lang_parsing() {
        for code in Language::VARIANTS_CODES {
            let lang: Language = code.parse().unwrap();
            assert_eq!(lang.code(), *code);
        }
        for lang in Language::VARIANTS_LANGUAGES {
            let lang_enum: Language = lang.parse().unwrap();
            assert_eq!(lang_enum.name(), *lang);
        }
        for (lang, code) in Language::ADDITIONAL_MAPPINGS {
            let lang_enum: Language = lang.parse().unwrap();
            assert_eq!(lang_enum.code(), *code);
        }
    }

    #[test]
    fn clap_value_enum() {
        #[derive(clap::Parser, Debug)]
        struct Opts {
            #[clap(value_enum)]
            lang: Language,
        }

        for code in Language::VARIANTS_CODES {
            let opts = Opts::try_parse_from(["test", code]).unwrap_or_else(|e| {
                panic!("Failed to parse '{}': {}", code, e);
            });
            assert_eq!(opts.lang.code(), *code);
        }
        for lang in Language::VARIANTS_LANGUAGES {
            let opts = Opts::try_parse_from(["test", lang]).unwrap_or_else(|e| {
                panic!("Failed to parse '{}': {}", lang, e);
            });
            assert_eq!(opts.lang.name(), *lang);
        }
        for (lang, code) in Language::ADDITIONAL_MAPPINGS {
            let opts = Opts::try_parse_from(["test", lang]).unwrap_or_else(|e| {
                panic!("Failed to parse '{}': {}", lang, e);
            });
            assert_eq!(opts.lang.code(), *code);
        }
    }

    #[test]
    fn auto_lang() {
        #[derive(clap::Parser, Debug)]
        struct Opts {
            #[clap(value_enum)]
            lang: Language,
        }

        let opts = Opts::try_parse_from(["test", "auto"]).unwrap_or_else(|e| {
            panic!("Failed to parse 'auto': {}", e);
        });
        assert_eq!(opts.lang.code(), "auto");
        assert_eq!(opts.lang.name(), "auto");
    }
}
