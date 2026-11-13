//! Configuration options for transcription

use purr_common::beams::BeamSwitch;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};

use crate::accel::{
    backend::AccelBackend,
    config::{AccelConfig, AccelMode},
};

/// The threads a transcription uses when none is asked for: every one the processor runs at once.
pub fn auto_threads() -> std::io::Result<usize> {
    std::thread::available_parallelism().map(std::num::NonZeroUsize::get)
}

/// Configuration for transcription operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionConfig {
    /// Path to the Whisper model file
    pub model_path: Option<PathBuf>,

    /// Language code (e.g., "en", "es", "fr")
    pub language: Option<Cow<'static, str>>,

    /// Translate to English (like whisper.cpp --translate flag)
    pub translate: bool,

    /// Acceleration backend selection and fallback
    pub accel: AccelConfig,

    /// Number of threads to use
    pub num_threads: Option<usize>,

    /// Sample rate to use in Hz. 16'000 is the recommended and default value.
    pub sample_rate: u32,

    /// Maximum audio duration in seconds
    pub max_duration: Option<f32>,

    /// Temperature for sampling (0.0 = deterministic)
    pub temperature: f32,

    /// Beam size for beam search
    pub beam_size: Option<usize>,

    /// Output format options
    pub output_format: OutputFormat,

    /// Enable verbose debug output
    pub verbose: bool,

    /// Seconds of new audio a stream with a voice activity model gathers before it looks for a
    /// pause to decode up to
    pub chunk_size: f32,

    /// Silero voice activity model (ggml): streaming windows end at pauses in speech, and windows
    /// without speech are not decoded
    pub vad_model_path: Option<PathBuf>,

    /// Whether and when the stream shows text it has not committed yet
    pub tentative: TentativeText,

    /// Sends the decoder's beams as stream events while on, read at the start of each window
    #[serde(skip)]
    pub beam_telemetry: BeamSwitch,

    /// HTTP timeout for URL requests (in seconds)
    pub http_timeout: u64,

    /// HTTP connection timeout for URL requests (in seconds)
    pub http_connect_timeout: u64,
}

/// When a stream shows text it has not committed yet, as tentative chunks.
///
/// A tentative chunk carries the stream's whole uncommitted tail, with the index the next
/// committed chunk will take. A later tentative chunk replaces it, and so does any committed
/// chunk with the same or a higher index; the stream's last chunk is committed, so no
/// tentative text outlives a stream that ends normally.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TentativeText {
    /// Only committed text
    Off,
    /// Also the end of a full window that was decoded but not committed, which costs nothing
    Decoded,
    /// As `Decoded`, and also decode the pending audio while the stream waits for a slow (live)
    /// source, once at least `every_seconds` of new audio arrived and only when the next window
    /// cannot be ready before that decode ends; each such decode costs a whole encoder pass
    WhileWaiting { every_seconds: f32 },
}

/// Output format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputFormat {
    /// Include timestamps in the output
    pub include_timestamps: bool,

    /// Include word-level timestamps
    pub word_timestamps: bool,

    /// Include confidence scores
    pub include_confidence: bool,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            model_path: None, // Will use default model
            language: None,   // Auto-detect
            translate: false,
            accel: AccelConfig::default(),
            num_threads: None,  // Use system default
            sample_rate: 16000, // Whisper's preferred sample rate
            max_duration: None,
            temperature: 0.0,
            beam_size: Some(5), // Default to beam search with 5 beams for better quality
            output_format: OutputFormat::default(),
            verbose: false,
            chunk_size: 4.0, // 4 seconds
            vad_model_path: None,
            tentative: TentativeText::Off,
            beam_telemetry: BeamSwitch::default(),
            http_timeout: 30,         // 30 seconds
            http_connect_timeout: 10, // 10 seconds
        }
    }
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self {
            include_timestamps: true,
            word_timestamps: false,
            include_confidence: false,
        }
    }
}

impl TranscriptionConfig {
    /// Create a new configuration with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model path
    pub fn with_model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.model_path = Some(path.as_ref().into());
        self
    }

    /// Set the language
    pub fn with_language<S: Into<Cow<'static, str>>>(mut self, language: S) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Optionally set the language
    pub fn with_opt_language<S: Into<Cow<'static, str>>>(mut self, language: Option<S>) -> Self {
        self.language = language.map(Into::into);
        self
    }

    /// Set the acceleration backend selection
    #[must_use]
    pub fn with_accel(mut self, accel: AccelConfig) -> Self {
        self.accel = accel;
        self
    }

    /// `true` selects automatic backend choice with fallback, `false` forces the CPU
    #[must_use]
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.accel.mode = if use_gpu {
            AccelMode::Auto
        } else {
            AccelMode::Force {
                backend: AccelBackend::Cpu,
                allow_fallback: false,
            }
        };
        self
    }

    /// Set the number of threads
    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.num_threads = Some(threads);
        self
    }

    /// Set the sample rate
    #[must_use]
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Enable or disable verbose output
    #[must_use]
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Enable or disable translation to English
    #[must_use]
    pub fn with_translate(mut self, translate: bool) -> Self {
        self.translate = translate;
        self
    }

    /// Set the temperature for sampling
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the beam size for beam search
    #[must_use]
    pub fn with_beam_size(mut self, beam_size: usize) -> Self {
        self.beam_size = Some(beam_size);
        self
    }

    /// Set the seconds of new audio a stream gathers before it looks for a pause
    #[must_use]
    pub fn with_chunk_size(mut self, chunk_size: f32) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Optionally set the voice activity model used while streaming
    #[must_use]
    pub fn with_vad_model(mut self, path: Option<PathBuf>) -> Self {
        self.vad_model_path = path;
        self
    }

    /// Set when the stream shows text it has not committed yet
    #[must_use]
    pub fn with_tentative(mut self, tentative: TentativeText) -> Self {
        self.tentative = tentative;
        self
    }

    /// Share `switch`, which turns the beam telemetry on and off while the stream runs
    #[must_use]
    pub fn with_beam_telemetry(mut self, switch: BeamSwitch) -> Self {
        self.beam_telemetry = switch;
        self
    }

    /// Set the HTTP timeout for URL requests (in seconds)
    #[must_use]
    pub fn with_http_timeout(mut self, timeout: u64) -> Self {
        self.http_timeout = timeout;
        self
    }

    /// Set the HTTP connection timeout for URL requests (in seconds)
    #[must_use]
    pub fn with_http_connect_timeout(mut self, timeout: u64) -> Self {
        self.http_connect_timeout = timeout;
        self
    }

    /// Edit output format options
    #[must_use]
    pub fn apply_output_format<F>(mut self, f: F) -> Self
    where
        F: FnOnce(OutputFormat) -> OutputFormat,
    {
        self.output_format = f(self.output_format);
        self
    }
}

impl OutputFormat {
    /// Create a new output format configuration with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable timestamps in the output
    #[must_use]
    pub fn with_timestamps(mut self, include: bool) -> Self {
        self.include_timestamps = include;
        self
    }

    /// Enable or disable word-level timestamps
    #[must_use]
    pub fn with_word_timestamps(mut self, include: bool) -> Self {
        self.word_timestamps = include;
        self
    }

    /// Enable or disable confidence scores in the output
    #[must_use]
    pub fn with_confidence(mut self, include: bool) -> Self {
        self.include_confidence = include;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accelerates_automatically_by_default() {
        assert_eq!(TranscriptionConfig::new().accel.mode, AccelMode::Auto);
    }

    #[test]
    fn disabling_gpu_forces_the_cpu_without_fallback() {
        assert_eq!(
            TranscriptionConfig::new()
                .with_gpu(false)
                .accel
                .attempt_order(),
            [AccelBackend::Cpu]
        );
    }

    #[test]
    fn enabling_gpu_restores_automatic_mode_and_keeps_the_priority() {
        let accel = AccelConfig::cpu_only().with_priority(vec![AccelBackend::Vulkan]);
        let config = TranscriptionConfig::new().with_accel(accel).with_gpu(true);
        assert_eq!(config.accel.mode, AccelMode::Auto);
        assert_eq!(config.accel.priority, [AccelBackend::Vulkan]);
    }
}
