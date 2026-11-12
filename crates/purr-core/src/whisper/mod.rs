pub mod beams;
mod commit;
mod language;
pub mod loader;
pub mod logging;
mod segments;
mod speech;
pub mod streaming;
pub mod sync;
mod tentative;
pub mod words;

use purr_common::{beams::BeamSnapshot, platform::ChunkKind, progress::ProgressFrame, word::Word};
use serde::{Deserialize, Serialize};
use std::future::Future;

use crate::{TranscriptionConfig, accel::report::AccelReport, error::Result};

pub trait TranscriptionResult {}

/// Transcription statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionStats {
    /// Total processing time in seconds
    pub processing_time: f64,

    /// Audio duration in seconds
    pub audio_duration: f32,

    /// Number of segments produced
    pub segment_count: usize,

    /// Total number of words
    pub word_count: usize,
}

impl TranscriptionStats {
    #[must_use]
    pub fn real_time_factor(&self) -> f32 {
        if self.processing_time > 0.0 {
            self.audio_duration / self.processing_time as f32
        } else {
            0.0
        }
    }

    #[must_use]
    pub fn words_per_minute(&self) -> f32 {
        if self.audio_duration > 0.0 {
            (self.word_count as f32 * 60.0) / self.audio_duration
        } else {
            0.0
        }
    }
}

/// Transcription result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncTranscriptionResult {
    /// The transcribed text
    pub text: String,

    /// Segments with timestamps
    pub segments: Vec<TranscriptionSegment>,

    /// Processing time in seconds
    pub processing_time: f64,

    /// Audio duration in seconds
    pub audio_duration: f32,

    /// Transcription statistics
    pub stats: TranscriptionStats,

    /// Acceleration backend the transcription ran on
    pub accel: AccelReport,
}

impl TranscriptionResult for SyncTranscriptionResult {}

/// A transcription segment with timestamps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    /// Segment text, trimmed
    pub text: String,

    /// Start time in seconds
    pub start: f64,

    /// End time in seconds
    pub end: f64,

    /// The words of `text`, with their scores, and their times when word timestamps were asked for
    pub words: Vec<Word>,
}

/// Streaming transcription chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChunk {
    /// Chunk text
    pub text: String,

    /// Start time in seconds
    pub start: f64,

    /// End time in seconds  
    pub end: f64,

    /// Whether this is the stream's last chunk
    pub is_final: bool,

    /// Chunk index; a committed chunk replaces a tentative chunk of the same or a lower index
    pub chunk_index: usize,

    /// Final statistics (only present on the very last chunk)
    pub final_stats: Option<TranscriptionStats>,

    /// The language Whisper recognised in this chunk (a Whisper code), when it was asked to detect it
    pub language: Option<String>,

    /// Whether the text is final, or only the uncommitted tail shown early
    pub kind: ChunkKind,

    /// The words of `text`, timed on the source timeline when word timestamps were asked for
    pub words: Vec<Word>,
}

/// What a streaming transcription reports, in order.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Text the stream will not change any more, or tentative text it will replace
    Chunk(StreamingChunk),
    /// How far the stream has come, sent before and after each decoded window
    Progress(StreamProgress),
    /// The decoder's hypotheses, at most ten times a second while the beam switch is on
    Beams(BeamSnapshot),
}

impl StreamEvent {
    /// The chunk this event carries, if it is one.
    #[must_use]
    pub fn into_chunk(self) -> Option<StreamingChunk> {
        match self {
            Self::Chunk(chunk) => Some(chunk),
            Self::Progress(_) | Self::Beams(_) => None,
        }
    }
}

/// Progress of a stream, with what it knows of its source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamProgress {
    pub frame: ProgressFrame,
    /// Seconds of audio received from the source so far
    pub received: f64,
    /// Whether the source has delivered all its audio, which makes `received` its duration
    pub source_ended: bool,
}

pub trait WhisperTranscriber {
    type TranscriberResult: TranscriptionResult;
    type InputData;

    /// Create a new transcriber with the given configuration
    fn from_config(config: TranscriptionConfig) -> impl Future<Output = Result<Self>>
    where
        Self: Sized;

    /// Transcribe audio data
    fn transcribe(
        self,
        input: Self::InputData,
    ) -> impl Future<Output = Result<Self::TranscriberResult>>;
}
