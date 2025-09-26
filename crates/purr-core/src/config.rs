//! Configuration options for transcription

use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};

/// Configuration for transcription operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionConfig {
    /// Path to the Whisper model file
    pub model_path: Option<PathBuf>,

    /// Language code (e.g., "en", "es", "fr")
    pub language: Option<Cow<'static, str>>,

    /// Translate to English (like whisper.cpp --translate flag)
    pub translate: bool,

    /// Use GPU acceleration if available
    pub use_gpu: bool,

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

    /// Chunk size for streaming (in seconds)
    pub chunk_size: f32,

    /// Chunk overlap for streaming (in seconds)
    pub chunk_overlap: f32,
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
            use_gpu: true,
            num_threads: None,  // Use system default
            sample_rate: 16000, // Whisper's preferred sample rate
            max_duration: None,
            temperature: 0.0,
            beam_size: Some(5), // Default to beam search with 5 beams for better quality
            output_format: OutputFormat::default(),
            verbose: false,
            chunk_size: 4.0,    // 4 seconds
            chunk_overlap: 0.5, // 0.5 seconds
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

    /// Enable or disable GPU acceleration
    #[must_use]
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Disable GPU acceleration
    #[must_use]
    pub fn without_gpu(mut self, no_gpu: bool) -> Self {
        self.use_gpu = !no_gpu;
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

    /// Set the chunk size for streaming (in seconds)
    #[must_use]
    pub fn with_chunk_size(mut self, chunk_size: f32) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Set the chunk overlap for streaming (in seconds)
    #[must_use]
    pub fn with_chunk_overlap(mut self, chunk_overlap: f32) -> Self {
        self.chunk_overlap = chunk_overlap;
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
