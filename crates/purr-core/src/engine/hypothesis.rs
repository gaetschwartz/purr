//! What an engine recognised in one window of audio.

use purr_common::{language::tag::LanguageTag, word::Word};

/// A stretch of a window, in samples from the window start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// One piece of recognised text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// The trimmed text
    pub text: String,
    /// Where the engine places the text in its window; `None` when it does not say
    pub span: Option<Span>,
    /// The words of `text`, with byte ranges into it and times from the window start
    pub words: Vec<Word>,
}

/// Whether the engine's segments end where its text ends sentences or pauses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segmentation {
    /// The engine's own segments, which the pipeline keeps as they are
    Native,
    /// Long stretches the pipeline splits at sentence ends and pauses, from the word times
    Synthesize,
}

/// The language an engine recognised.
#[derive(Debug, Clone, PartialEq)]
pub struct LanguageEstimate {
    pub tag: LanguageTag,
    /// How sure the engine is, from 0 to 1, when it says
    pub confidence: Option<f32>,
}

/// Everything an engine recognised in a window.
#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    pub segments: Vec<Segment>,
    /// The language the engine detected, when it was asked to
    pub language: Option<LanguageEstimate>,
    pub segmentation: Segmentation,
}

impl Hypothesis {
    /// Nothing recognised.
    #[must_use]
    pub const fn empty(segmentation: Segmentation) -> Self {
        Self {
            segments: Vec::new(),
            language: None,
            segmentation,
        }
    }
}

/// Whether a decode ran to its end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Completion {
    Complete,
    /// Stopped by its cancel token; the hypothesis holds what the engine finished before
    Cancelled,
    /// Stopped before the end of the window because it ran out of room for its output; the
    /// hypothesis holds the text so far, which says nothing about how far into the window it got
    Truncated,
    /// Stopped because its output began repeating itself; the hypothesis holds the text before
    /// the loop, which says nothing about how far into the window it got
    Repeating,
}

/// The outcome of one decode.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    pub hypothesis: Hypothesis,
    pub completion: Completion,
}
