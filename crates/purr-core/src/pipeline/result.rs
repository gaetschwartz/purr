//! A whole transcript, gathered from the events of a transcription.

use futures::StreamExt;
use purr_common::{platform::ChunkKind, word::Word};
use serde::{Deserialize, Serialize};

use crate::{accel::report::AccelReport, error::Result};

use super::events::{StreamEvent, Transcription, TranscriptionStats};

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

/// Runs `transcription` to its end and gathers its committed text.
pub async fn collect(mut transcription: Transcription) -> Result<SyncTranscriptionResult> {
    let mut segments = Vec::new();
    let mut final_stats = None;
    while let Some(event) = transcription.next().await {
        let chunk = match event? {
            StreamEvent::Chunk(chunk) => chunk,
            StreamEvent::Progress(_) => continue,
        };
        match chunk.kind {
            ChunkKind::Committed => {}
            ChunkKind::Tentative => continue,
        }
        if chunk.final_stats.is_some() {
            final_stats = chunk.final_stats;
        }
        if !chunk.text.is_empty() {
            segments.push(TranscriptionSegment {
                text: chunk.text,
                start: chunk.start,
                end: chunk.end,
                words: chunk.words,
            });
        }
    }

    let text = segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let (processing_time, audio_duration) = final_stats.map_or((0.0, 0.0), |stats| {
        (stats.processing_time, stats.audio_duration)
    });
    let stats = TranscriptionStats {
        processing_time,
        audio_duration,
        segment_count: segments.len(),
        word_count: text.split_whitespace().count(),
    };
    Ok(SyncTranscriptionResult {
        text,
        segments,
        processing_time,
        audio_duration,
        stats,
        accel: transcription.accel().clone(),
    })
}
