use crate::fmt::Verbosity;
use clap::builder::{
    Styles,
    styling::{AnsiColor, Effects, Style},
};
use clap::{
    Parser, Subcommand,
    builder::{PossibleValue, TypedValueParser},
    error::ErrorKind,
};
use purr_common::Language;
use purr_core::accel::{backend::AccelBackend, config::AccelConfig};
use std::{ffi::OsStr, path::PathBuf, str::FromStr as _};

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

    /// Acceleration backend. `auto` tries the priority list from best to worst, moving to the
    /// next backend when one is unavailable or fails (the CPU is the last resort). A named
    /// backend is forced: there is no fallback, and it is an error if it cannot be used
    /// (see --accel-fallback). Run `purr sys` to see what this build supports
    #[arg(
        long,
        value_name = "BACKEND",
        value_parser = AccelChoiceParser,
        conflicts_with = "no_gpu"
    )]
    pub accel: Option<AccelChoice>,

    /// With --accel BACKEND: when it is unavailable or fails, fall back to the priority list
    /// instead of failing
    #[arg(long, requires = "accel", conflicts_with = "no_gpu")]
    pub accel_fallback: bool,

    /// Priority order for `auto` and --accel-fallback, best first, comma separated
    /// (e.g. vulkan,cuda). The CPU is appended when absent
    #[arg(
        long,
        value_name = "BACKENDS",
        value_delimiter = ',',
        value_parser = AccelBackendParser,
        conflicts_with = "no_gpu"
    )]
    pub accel_priority: Vec<AccelBackend>,

    /// Disable GPU acceleration (same as --accel cpu)
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

    /// Seconds of new audio to gather before looking for a pause to decode up to while streaming
    /// (only with --vad-model)
    #[arg(long, default_value = "4.0")]
    pub chunk_size: f32,

    /// Silero voice activity model (ggml-silero-*.bin from whisper.cpp): streaming decodes up to
    /// pauses in speech and skips audio without speech
    #[arg(long, value_name = "PATH")]
    pub vad_model: Option<PathBuf>,

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
    /// Verify the SHA-256 of downloaded models (all downloaded models by default)
    Verify {
        /// Model to verify (e.g., base, small, large-v3)
        #[arg(value_name = "MODEL")]
        model: Option<String>,
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

/// Value of `--accel`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccelChoice {
    Auto,
    Backend(AccelBackend),
}

fn backend_names() -> impl Iterator<Item = &'static str> {
    AccelBackend::ALL
        .iter()
        .map(|backend| <&'static str>::from(*backend))
}

fn parse_backend(
    cmd: &clap::Command,
    arg: Option<&clap::Arg>,
    value: &OsStr,
    extra: &[&'static str],
) -> Result<AccelBackend, clap::Error> {
    let text = value
        .to_str()
        .ok_or_else(|| clap::Error::new(ErrorKind::InvalidUtf8).with_cmd(cmd))?;
    AccelBackend::from_str(text).map_err(|_unknown| {
        let expected = extra
            .iter()
            .copied()
            .chain(backend_names())
            .collect::<Vec<_>>()
            .join(", ");
        let name = arg.map_or_else(|| "backend".to_owned(), ToString::to_string);
        cmd.clone().error(
            ErrorKind::InvalidValue,
            format!("invalid value '{text}' for '{name}'\n  [possible values: {expected}]"),
        )
    })
}

fn possible_backends(extra: &'static [&'static str]) -> Box<dyn Iterator<Item = PossibleValue>> {
    Box::new(
        extra
            .iter()
            .copied()
            .chain(backend_names())
            .map(PossibleValue::new),
    )
}

#[derive(Clone)]
struct AccelBackendParser;

impl TypedValueParser for AccelBackendParser {
    type Value = AccelBackend;

    fn parse_ref(
        &self,
        cmd: &clap::Command,
        arg: Option<&clap::Arg>,
        value: &OsStr,
    ) -> Result<Self::Value, clap::Error> {
        parse_backend(cmd, arg, value, &[])
    }

    fn possible_values(&self) -> Option<Box<dyn Iterator<Item = PossibleValue> + '_>> {
        Some(possible_backends(&[]))
    }
}

#[derive(Clone)]
struct AccelChoiceParser;

impl TypedValueParser for AccelChoiceParser {
    type Value = AccelChoice;

    fn parse_ref(
        &self,
        cmd: &clap::Command,
        arg: Option<&clap::Arg>,
        value: &OsStr,
    ) -> Result<Self::Value, clap::Error> {
        if value == "auto" {
            return Ok(AccelChoice::Auto);
        }
        parse_backend(cmd, arg, value, &["auto"]).map(AccelChoice::Backend)
    }

    fn possible_values(&self) -> Option<Box<dyn Iterator<Item = PossibleValue> + '_>> {
        Some(possible_backends(&["auto"]))
    }
}

impl Cli {
    /// Acceleration settings from `--accel`, `--accel-fallback`, `--accel-priority` and
    /// `--no-gpu`.
    pub fn accel_config(&self) -> AccelConfig {
        let config = if self.no_gpu {
            AccelConfig::cpu_only()
        } else {
            match self.accel {
                None | Some(AccelChoice::Auto) => AccelConfig::default(),
                Some(AccelChoice::Backend(backend)) => {
                    AccelConfig::forced(backend, self.accel_fallback)
                }
            }
        };
        if self.accel_priority.is_empty() {
            config
        } else {
            config.with_priority(self.accel_priority.clone())
        }
    }
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
            .with_chunk_size(cli.chunk_size)
            .with_vad_model(cli.vad_model.clone())
            .with_accel(cli.accel_config())
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
                panic!("Failed to parse '{code}': {e}");
            });
            assert_eq!(opts.lang.code(), *code);
        }
        for lang in Language::VARIANTS_LANGUAGES {
            let opts = Opts::try_parse_from(["test", lang]).unwrap_or_else(|e| {
                panic!("Failed to parse '{lang}': {e}");
            });
            assert_eq!(opts.lang.name(), *lang);
        }
        for (lang, code) in Language::ADDITIONAL_MAPPINGS {
            let opts = Opts::try_parse_from(["test", lang]).unwrap_or_else(|e| {
                panic!("Failed to parse '{lang}': {e}");
            });
            assert_eq!(opts.lang.code(), *code);
        }
    }

    fn accel_of(args: &[&str]) -> AccelConfig {
        let args = ["purr", "audio.wav"]
            .into_iter()
            .chain(args.iter().copied());
        Cli::try_parse_from(args).unwrap().accel_config()
    }

    fn parse_error(args: &[&str]) -> ErrorKind {
        let args = ["purr", "audio.wav"]
            .into_iter()
            .chain(args.iter().copied());
        Cli::try_parse_from(args).unwrap_err().kind()
    }

    #[test]
    fn accel_defaults_to_auto() {
        assert_eq!(accel_of(&[]), AccelConfig::default());
        assert_eq!(accel_of(&["--accel", "auto"]), AccelConfig::default());
    }

    #[test]
    fn named_accel_is_forced_without_fallback() {
        assert_eq!(
            accel_of(&["--accel", "vulkan"]),
            AccelConfig::forced(AccelBackend::Vulkan, false)
        );
    }

    #[test]
    fn accel_fallback_allows_fallback_for_a_forced_backend() {
        assert_eq!(
            accel_of(&["--accel", "cuda", "--accel-fallback"]),
            AccelConfig::forced(AccelBackend::Cuda, true)
        );
    }

    #[test]
    fn no_gpu_forces_the_cpu() {
        assert_eq!(accel_of(&["--no-gpu"]), AccelConfig::cpu_only());
    }

    #[test]
    fn accel_priority_overrides_the_priority_list() {
        assert_eq!(
            accel_of(&["--accel-priority", "vulkan,cuda"]).priority,
            [AccelBackend::Vulkan, AccelBackend::Cuda]
        );
    }

    #[test]
    fn accel_fallback_needs_a_forced_backend() {
        assert_eq!(
            parse_error(&["--accel-fallback"]),
            ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn no_gpu_conflicts_with_the_accel_flags() {
        assert_eq!(
            parse_error(&["--no-gpu", "--accel", "cuda"]),
            ErrorKind::ArgumentConflict
        );
        assert_eq!(
            parse_error(&["--no-gpu", "--accel-priority", "cuda"]),
            ErrorKind::ArgumentConflict
        );
    }

    #[test]
    fn unknown_accel_backend_is_rejected() {
        assert_eq!(parse_error(&["--accel", "tpu"]), ErrorKind::InvalidValue);
        assert_eq!(
            parse_error(&["--accel-priority", "cuda,tpu"]),
            ErrorKind::InvalidValue
        );
    }

    #[test]
    fn auto_lang() {
        #[derive(clap::Parser, Debug)]
        struct Opts {
            #[clap(value_enum)]
            lang: Language,
        }

        let opts = Opts::try_parse_from(["test", "auto"]).unwrap_or_else(|e| {
            panic!("Failed to parse 'auto': {e}");
        });
        assert_eq!(opts.lang.code(), "auto");
        assert_eq!(opts.lang.name(), "auto");
    }
}
