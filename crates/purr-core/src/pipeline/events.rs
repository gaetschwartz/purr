//! What a transcription reports while it runs.

use std::{
    pin::Pin,
    task::{Context, Poll},
};

use futures::Stream;
use purr_common::{
    language::tag::LanguageTag, platform::ChunkKind, progress::ProgressFrame, word::Word,
};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;

use crate::{SourceLength, accel::report::AccelReport, engine::cancel::CancelToken, error::Error};

use super::driver::{EndSlot, Request};

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

    /// The language the engine recognised in this chunk, when it was asked to detect it
    pub language: Option<LanguageTag>,

    /// Whether the text is final, or only the uncommitted tail shown early
    pub kind: ChunkKind,

    /// The words of `text`, timed on the source timeline when word timestamps were asked for
    pub words: Vec<Word>,
}

/// What a transcription reports, in order.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Text the stream will not change any more, or tentative text it will replace
    Chunk(StreamingChunk),
    /// How far the stream has come, sent before and after each decoded window
    Progress(StreamProgress),
}

impl StreamEvent {
    /// The chunk this event carries, if it is one.
    #[must_use]
    pub fn into_chunk(self) -> Option<StreamingChunk> {
        match self {
            Self::Chunk(chunk) => Some(chunk),
            Self::Progress(_) => None,
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

/// Why the source of a transcription ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    /// The source delivered all its audio
    SourceEnded,
    /// The run took as much audio as it was asked to listen for from a source of unknown length
    DurationReached,
    /// [`Transcription::stop`] ended the source
    Stopped,
}

/// A running transcription: its events, and the device it runs on.
///
/// Dropping it cancels the transcription, which stops the engine inside the window it decodes.
pub struct Transcription {
    stream: Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send>>,
    accel: AccelReport,
    cancel: CancelToken,
    request: watch::Sender<Request>,
    length: SourceLength,
    end: EndSlot,
}

impl Transcription {
    pub(crate) fn new(
        stream: Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send>>,
        accel: AccelReport,
        cancel: CancelToken,
        request: watch::Sender<Request>,
        length: SourceLength,
        end: EndSlot,
    ) -> Self {
        Self {
            stream,
            accel,
            cancel,
            request,
            length,
            end,
        }
    }

    /// Acceleration backend the transcription runs on
    #[must_use]
    pub const fn accel(&self) -> &AccelReport {
        &self.accel
    }

    /// What the source says of its length.
    #[must_use]
    pub const fn length(&self) -> SourceLength {
        self.length
    }

    /// Why the source ended; set once the source has ended, which is before the last chunk is
    /// sent, and left `None` when the transcription failed or was cancelled first.
    #[must_use]
    pub fn end_reason(&self) -> Option<EndReason> {
        self.end.get()
    }

    /// Ends the source now: the audio that has arrived is decoded and the stream finishes with
    /// its last chunk. Nothing happens when the source has already ended.
    pub fn stop(&self) {
        self.request.send_replace(Request::Stop);
    }

    /// Stops the transcription at once. A window being decoded ends with what the engine finished
    /// of it, committed as the stream's last chunk; otherwise the stream just ends.
    pub fn cancel(&self) {
        self.cancel.cancel();
        self.request.send_replace(Request::Cancel);
    }
}

impl Drop for Transcription {
    fn drop(&mut self) {
        self.cancel();
    }
}

impl Stream for Transcription {
    type Item = Result<StreamEvent, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}
