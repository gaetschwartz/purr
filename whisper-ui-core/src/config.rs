//! Configuration options for transcription

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for transcription operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionConfig {
    /// Path to the Whisper model file
    pub model_path: Option<PathBuf>,
    
    /// Language code (e.g., "en", "es", "fr")
    pub language: Option<String>,
    
    /// Use GPU acceleration if available
    pub use_gpu: bool,
    
    /// Number of threads to use
    pub num_threads: Option<usize>,
    
    /// Audio sample rate to convert to
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
            use_gpu: true,
            num_threads: None, // Use system default
            sample_rate: 16000, // Whisper's preferred sample rate
            max_duration: None,
            temperature: 0.0,
            beam_size: None,
            output_format: OutputFormat::default(),
            verbose: false,
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
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the model path
    pub fn with_model_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.model_path = Some(path.into());
        self
    }
    
    /// Set the language
    pub fn with_language<S: Into<String>>(mut self, language: S) -> Self {
        self.language = Some(language.into());
        self
    }
    
    /// Enable or disable GPU acceleration
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }
    
    /// Set the number of threads
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.num_threads = Some(threads);
        self
    }
    
    /// Set the sample rate
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }
    
    /// Enable or disable verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}
