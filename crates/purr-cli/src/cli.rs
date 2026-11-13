use crate::color::ColorChoice;
use crate::logging::level::LogLevel;
use crate::theme::CLAP_STYLES;
use crate::verbosity::Verbosity;
use clap::{
    CommandFactory as _, FromArgMatches as _, Parser, Subcommand,
    builder::{PossibleValue, TypedValueParser},
    error::ErrorKind,
};
use purr_common::devices::{DeviceDefaults, DeviceOrder};
use purr_common::{Language, beams::BeamSwitch};
use purr_core::accel::{
    backend::AccelBackend,
    devices::DetectedDevice,
    selector::{DeviceSelector, SelectorError},
};
use std::{ffi::OsStr, path::PathBuf, str::FromStr as _};

const ABOUT: &str = "😸 Transcribe audio files and URLs using Whisper AI";
#[derive(Parser, Debug)]
#[command(name = env!("CARGO_BIN_NAME"), bin_name = env!("CARGO_BIN_NAME"), author = env!("CARGO_PKG_AUTHORS"))]
#[command(about = ABOUT)]
#[command(version = crate::build_info::version())]
#[command(styles = CLAP_STYLES)]
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

    /// Which backend's devices to try first. `auto` keeps the usual order (CUDA, ROCm, CoreML,
    /// Metal, SYCL, Vulkan, CPU). A named backend goes first and the other devices follow as
    /// fallbacks, so `--accel cpu` runs on the CPU alone. Run `purr sys` to see what this build
    /// supports
    #[arg(
        long,
        value_name = "BACKEND",
        value_parser = AccelChoiceParser,
        conflicts_with = "no_gpu"
    )]
    pub accel: Option<AccelChoice>,

    /// Backends whose devices are tried first, best first, comma separated (e.g. vulkan,cuda);
    /// the other devices follow in the usual order
    #[arg(
        long,
        value_name = "BACKENDS",
        value_delimiter = ',',
        value_parser = AccelBackendParser,
        conflicts_with = "no_gpu"
    )]
    pub accel_priority: Vec<AccelBackend>,

    /// Run on one device first; the others stay as fallbacks. Give a device index as `purr sys`
    /// prints it (`--device 2`), or describe it with tokens joined by `/` in any order: a
    /// backend (cuda, vulkan, rocm, ...), a type (gpu, igpu, cpu, accel) and a device id such as
    /// 0000:01:00.0 (`--device vulkan/igpu`, `--device cuda/0000:01:00.0`, `--device gpu`).
    /// The explicit `backend=`, `type=`, `id=` and `index=` forms also work. It
    /// must match exactly one device; otherwise the candidates are listed
    #[arg(long, value_name = "DEVICE", conflicts_with = "no_gpu")]
    pub device: Option<DeviceSelector>,

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

    /// Time every word, and add each segment's words (text, start, end, score) to --output json
    #[arg(long)]
    pub word_timestamps: bool,

    /// While streaming, show how far the transcription has come, as position / duration
    /// (`2:51 / 7:30`; the duration shows once the whole audio has been read), in a line under
    /// the text. Only on a terminal: with --plain or a redirected stdout there is no progress
    /// and the text is not broken into lines
    #[arg(long)]
    pub progress: bool,

    /// While streaming, print on stderr (in plain mode too) one line per decoding step (at most
    /// ten a second): the window, the step, and each of the decoder's beams with its log
    /// probability and newest text
    #[arg(long)]
    pub trace_beams: bool,

    /// Stream transcription results in real-time
    #[arg(short = 'S', long)]
    pub no_stream: bool,

    /// Temperature for sampling (0.0 = deterministic)
    #[arg(long, default_value = "0.0")]
    pub temperature: f32,

    #[clap(flatten)]
    pub verbosity: Verbosity,

    /// When to colour the output. `always` and `never` override the environment. `auto` follows,
    /// in this order: NO_COLOR (any non-empty value disables colours), CLICOLOR_FORCE (any value
    /// but 0 enables them), CLICOLOR=0 (disables them), then whether the stream is a terminal.
    /// Standard output and standard error are decided separately
    #[arg(long, global = true, value_enum, default_value_t, value_name = "WHEN")]
    pub color: ColorChoice,

    /// Print only the command's result on stdout: no banner, status lines or progress bars. This
    /// is already the case whenever stdout is not a terminal (a pipe or a file); the flag
    /// forces it on a terminal too
    #[arg(long, global = true)]
    pub plain: bool,

    /// Append the diagnostic log to PATH as plain text, never coloured, whatever the terminal
    /// shows. Each line is written as it happens, so nothing is lost if the run ends abruptly
    #[arg(long, global = true, value_name = "PATH")]
    pub log_file: Option<PathBuf>,

    /// Level of what --log-file records: error, warn, info, debug or trace. It does not depend
    /// on -v, which only controls the terminal [default: info]
    #[arg(
        long,
        global = true,
        value_enum,
        value_name = "LEVEL",
        requires = "log_file"
    )]
    pub log_level: Option<LogLevel>,

    /// Sample rate to use in Hz. 16'000 is the recommended and default value.
    #[arg(long, default_value = "16000")]
    pub sample_rate: u32,

    /// Seconds of new audio to gather before looking for a pause to decode up to while streaming
    /// (only with a voice activity model)
    #[arg(long, default_value = "4.0")]
    pub chunk_size: f32,

    /// Silero voice activity model (ggml-silero-*.bin from whisper.cpp): streaming decodes up to
    /// pauses in speech and skips audio without speech. Defaults to the one fetched alongside
    /// downloaded models, when present
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
    Sys {
        /// Print one JSON document instead of text; each backend is "Disabled",
        /// "EnabledButNotAvailable", {"Available":null} or {"RuntimeLibraryMissing":{library,searched}}
        #[arg(long)]
        json: bool,
    },
    /// GPU runtime libraries: where they are looked for and where to put them
    Runtime {
        #[command(subcommand)]
        command: RuntimeCommands,
    },
}

#[derive(Subcommand, Debug)]
pub enum RuntimeCommands {
    /// Show where GPU runtime libraries are searched, in order, and what each place holds
    Status,
    /// Print the directory a runtime pack should be unpacked into
    Path,
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
    /// Parses the process arguments; `--help` and usage errors honour `--color` too.
    pub fn from_env() -> Self {
        let args: Vec<std::ffi::OsString> = std::env::args_os().collect();
        let color = Self::color_choice_in(&args);
        Self::command()
            .color(color.into())
            .try_get_matches_from(&args)
            .and_then(|matches| Self::from_arg_matches(&matches))
            .unwrap_or_else(|error| error.exit())
    }

    /// `--color` read ahead of the full parse, so that the help text it triggers is already
    /// coloured accordingly. Whatever is wrong with the rest of the command line is ignored.
    fn color_choice_in<S: AsRef<OsStr>>(args: &[S]) -> ColorChoice {
        Self::command()
            .disable_help_flag(true)
            .disable_version_flag(true)
            .ignore_errors(true)
            .try_get_matches_from(args.iter().map(AsRef::as_ref))
            .ok()
            .and_then(|matches| matches.get_one::<ColorChoice>("color").copied())
            .unwrap_or_default()
    }

    /// The devices `--accel`, `--accel-priority` and `--no-gpu` ask for, before `--device`.
    pub fn device_order(&self) -> DeviceOrder {
        if self.no_gpu || self.accel == Some(AccelChoice::Backend(AccelBackend::Cpu)) {
            return DeviceOrder::cpu_only();
        }
        let mut backends_first: Vec<AccelBackend> = Vec::new();
        if let Some(AccelChoice::Backend(backend)) = self.accel {
            backends_first.push(backend);
        }
        for backend in &self.accel_priority {
            if !backends_first.contains(backend) {
                backends_first.push(*backend);
            }
        }
        DeviceOrder::automatic().with_defaults(DeviceDefaults {
            gpus_enabled: true,
            backends_first: backends_first.into_iter().map(Into::into).collect(),
        })
    }

    /// [`Self::device_order`] with the `--device` device in front, found among `detected`.
    pub fn device_order_among(
        &self,
        detected: &[DetectedDevice],
    ) -> Result<DeviceOrder, SelectorError> {
        let order = self.device_order();
        match &self.device {
            None => Ok(order),
            Some(selector) => {
                let device = selector.resolve(detected)?;
                Ok(order.with_first(Some(device.key.clone())))
            }
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
            .with_beam_telemetry(BeamSwitch::new(cli.trace_beams))
            .with_devices(cli.device_order())
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

    fn cli_of(args: &[&str]) -> Cli {
        let args = ["purr", "audio.wav"]
            .into_iter()
            .chain(args.iter().copied());
        Cli::try_parse_from(args).unwrap()
    }

    fn order_of(args: &[&str]) -> DeviceOrder {
        cli_of(args).device_order()
    }

    fn first_backends(order: &DeviceOrder) -> Vec<purr_common::accel::AccelBackend> {
        order.defaults().backends_first.clone()
    }

    fn parse_error(args: &[&str]) -> ErrorKind {
        let args = ["purr", "audio.wav"]
            .into_iter()
            .chain(args.iter().copied());
        Cli::try_parse_from(args).unwrap_err().kind()
    }

    fn detected_workstation() -> Vec<DetectedDevice> {
        use purr_common::platform::DeviceType;
        use purr_core::{accel::backend::BuildStatus, dev::Device};
        let device = |id, name: &str, backend, tpe| Device {
            id,
            name: format!("dev{id}"),
            description: name.to_owned(),
            device_id: None,
            vram_free: 0,
            vram_total: 0,
            tpe,
            backend: Some(backend),
            caps: None,
            vulkan_driver: None,
        };
        let inventory = purr_core::accel::inventory::Inventory {
            devices: vec![
                device(0, "RTX", AccelBackend::Cuda, DeviceType::Gpu),
                device(1, "RTX", AccelBackend::Vulkan, DeviceType::Gpu),
                device(2, "Radeon", AccelBackend::Vulkan, DeviceType::Igpu),
                device(3, "Ryzen", AccelBackend::Cpu, DeviceType::Cpu),
            ],
            build: AccelBackend::ALL
                .iter()
                .map(|backend| (*backend, BuildStatus::Built))
                .collect(),
            ..Default::default()
        };
        purr_core::accel::devices::detect(&inventory)
    }

    #[test]
    fn accel_defaults_to_the_usual_order() {
        assert_eq!(order_of(&[]), DeviceOrder::automatic());
        assert_eq!(order_of(&["--accel", "auto"]), DeviceOrder::automatic());
    }

    #[test]
    fn a_named_accel_puts_its_devices_first_and_keeps_the_rest_as_fallbacks() {
        assert_eq!(
            first_backends(&order_of(&["--accel", "vulkan"])),
            [purr_common::accel::AccelBackend::Vulkan]
        );
        assert!(order_of(&["--accel", "vulkan"]).defaults().gpus_enabled);
        assert!(!order_of(&["--accel", "vulkan"]).is_cpu_only());
    }

    #[test]
    fn accel_cpu_and_no_gpu_run_on_the_cpu_alone() {
        assert_eq!(order_of(&["--accel", "cpu"]), DeviceOrder::cpu_only());
        assert_eq!(order_of(&["--no-gpu"]), DeviceOrder::cpu_only());
    }

    #[test]
    fn accel_priority_lists_the_backends_to_try_first() {
        use purr_common::accel::AccelBackend::{Cuda, Vulkan};
        assert_eq!(
            first_backends(&order_of(&["--accel-priority", "vulkan,cuda"])),
            [Vulkan, Cuda]
        );
        assert_eq!(
            first_backends(&order_of(&[
                "--accel",
                "cuda",
                "--accel-priority",
                "vulkan,cuda"
            ])),
            [Cuda, Vulkan]
        );
    }

    #[test]
    fn device_takes_an_index_or_a_selector() {
        assert_eq!(
            cli_of(&["--device", "2"]).device,
            Some(DeviceSelector {
                index: Some(2),
                ..DeviceSelector::default()
            })
        );
        assert_eq!(
            cli_of(&["--device", "vulkan/igpu"]).device,
            Some(DeviceSelector {
                backend: Some(AccelBackend::Vulkan),
                kind: Some(purr_common::platform::DeviceType::Igpu),
                ..DeviceSelector::default()
            })
        );
    }

    #[test]
    fn a_malformed_device_selector_is_rejected() {
        assert_eq!(
            parse_error(&["--device", "colour=red"]),
            ErrorKind::ValueValidation
        );
    }

    #[test]
    fn the_selected_device_goes_first_among_the_detected_ones() {
        let detected = detected_workstation();
        let order = cli_of(&["--device", "type=igpu"])
            .device_order_among(&detected)
            .unwrap();
        let igpu = detected
            .iter()
            .find(|d| d.kind == Some(purr_common::platform::DeviceType::Igpu))
            .unwrap();
        assert_eq!(order.first(), Some(&igpu.key));
    }

    #[test]
    fn an_ambiguous_device_is_an_error_not_a_guess() {
        let detected = detected_workstation();
        assert!(matches!(
            cli_of(&["--device", "backend=vulkan"]).device_order_among(&detected),
            Err(SelectorError::Ambiguous { .. })
        ));
    }

    #[test]
    fn the_accel_fallback_option_is_gone() {
        assert_eq!(
            parse_error(&["--accel", "cuda", "--accel-fallback"]),
            ErrorKind::UnknownArgument
        );
    }

    #[test]
    fn no_gpu_conflicts_with_the_device_flags() {
        assert_eq!(
            parse_error(&["--no-gpu", "--device", "1"]),
            ErrorKind::ArgumentConflict
        );
    }

    #[test]
    fn color_is_a_global_flag_in_either_position() {
        for args in [
            ["purr", "--color", "never", "sys"],
            ["purr", "sys", "--color", "never"],
        ] {
            assert_eq!(Cli::try_parse_from(args).unwrap().color, ColorChoice::Never);
        }
        assert_eq!(
            Cli::try_parse_from(["purr", "sys"]).unwrap().color,
            ColorChoice::Auto
        );
        assert_eq!(
            Cli::try_parse_from(["purr", "--color", "sometimes", "sys"])
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidValue
        );
    }

    #[test]
    fn the_flag_is_found_wherever_it_stands() {
        let table: [(&[&str], ColorChoice); 7] = [
            (&["purr"], ColorChoice::Auto),
            (&["purr", "--color", "never", "sys"], ColorChoice::Never),
            (&["purr", "sys", "--color", "never"], ColorChoice::Never),
            (&["purr", "sys", "--color=always"], ColorChoice::Always),
            (
                &["purr", "a.wav", "-m", "base", "--color", "always", "-vv"],
                ColorChoice::Always,
            ),
            (&["purr", "--color", "sometimes", "sys"], ColorChoice::Auto),
            (&["purr", "--help"], ColorChoice::Auto),
        ];
        for (args, expected) in table {
            assert_eq!(Cli::color_choice_in(args), expected, "{args:?}");
        }
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
