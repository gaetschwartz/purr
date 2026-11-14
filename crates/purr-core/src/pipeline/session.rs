//! One transcription: the audio not yet committed, and what the stream has settled.

use std::time::{Duration, Instant};

use purr_common::{
    language::tag::LanguageTag,
    model::profile::Timing,
    platform::ChunkKind,
    progress::{ProgressFrame, ProgressPhase},
};
use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::{
    config::{RunConfig, TentativeText},
    engine::{
        cancel::CancelToken,
        decoder::{DecodeRequest, LanguageRequest, WindowDecoder},
        hypothesis::{Completion, LanguageEstimate, Span},
        speech::SpeechDetector,
    },
    error::{Result, TranscriptionError},
};

use super::{
    commit::{WindowKind, commit},
    events::{StreamEvent, StreamProgress, StreamingChunk, TranscriptionStats},
    language::StreamLanguage,
    pause::last_pause,
    policy::{Margins, SAMPLE_RATE, WindowPolicy},
    retry::shorter,
    segmenter::{Part, Segmenter},
    tentative::{self, Waiting},
};

const SAMPLES_PER_CENTISECOND: usize = SAMPLE_RATE / 100;

pub type Events = mpsc::UnboundedSender<Result<StreamEvent>>;

/// What one decode of a window gave.
struct Recognised {
    parts: Vec<Part>,
    /// The language the engine detected, when it was asked to
    language: Option<LanguageEstimate>,
    completion: Completion,
}

/// What a session decodes with.
pub struct Decoding {
    pub decoder: Box<dyn WindowDecoder>,
    pub speech: Option<Box<dyn SpeechDetector>>,
    pub policy: WindowPolicy,
    pub cancel: CancelToken,
}

/// One transcription: the audio not yet committed, and what the stream has settled.
pub struct Session {
    out: Events,
    tentative_text: TentativeText,
    timing: Timing,
    policy: WindowPolicy,
    margins: Margins,
    decoder: Box<dyn WindowDecoder>,
    speech: Option<Box<dyn SpeechDetector>>,
    segmenter: Segmenter,
    cancel: CancelToken,
    /// Audio from `origin` on that no committed segment covers yet
    pending: Vec<f32>,
    origin: usize,
    /// How much of `pending` a window has already looked at
    seen: usize,
    /// How much of `pending` the last tentative decode covered
    tentative_mark: usize,
    received: usize,
    first_audio: Option<Instant>,
    last_decode: Option<Duration>,
    /// Whether the source has delivered all its audio
    pub ended: bool,
    language: StreamLanguage,
    chunk_index: usize,
    segment_count: usize,
    word_count: usize,
    pub finished: bool,
    started: Instant,
}

impl Session {
    pub fn new(run: &RunConfig, decoding: Decoding, out: Events) -> Self {
        let language = run
            .language
            .clone()
            .map_or(StreamLanguage::Detecting, StreamLanguage::Chosen);
        Self {
            out,
            tentative_text: run.tentative,
            timing: if run.word_timestamps {
                Timing::Word
            } else {
                Timing::Segment
            },
            policy: decoding.policy,
            margins: Margins::default(),
            decoder: decoding.decoder,
            speech: decoding.speech,
            segmenter: Segmenter::default(),
            cancel: decoding.cancel,
            pending: Vec::new(),
            origin: 0,
            seen: 0,
            tentative_mark: 0,
            received: 0,
            first_audio: None,
            last_decode: None,
            ended: false,
            language,
            chunk_index: 0,
            segment_count: 0,
            word_count: 0,
            finished: false,
            started: Instant::now(),
        }
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    pub fn push(&mut self, samples: &[f32]) {
        if !samples.is_empty() {
            self.first_audio.get_or_insert_with(Instant::now);
        }
        self.received += samples.len();
        self.pending.extend_from_slice(samples);
    }

    /// The window to decode now, if any.
    pub fn next_window(&self, complete: bool) -> Option<WindowKind> {
        let len = self.pending.len();
        let last = complete && !self.finished;
        match self.policy {
            WindowPolicy::Whole => last.then_some(WindowKind::Last),
            WindowPolicy::Full { max } | WindowPolicy::AtPauses { max, .. } if len >= max => {
                Some(WindowKind::Full)
            }
            WindowPolicy::Full { .. } | WindowPolicy::AtPauses { .. } if last => {
                Some(WindowKind::Last)
            }
            WindowPolicy::AtPauses { step, .. } if len - self.seen >= step => {
                Some(WindowKind::Growing)
            }
            WindowPolicy::Full { .. } | WindowPolicy::AtPauses { .. } => None,
        }
    }

    fn sample_rate() -> f64 {
        SAMPLE_RATE as f64
    }

    /// Where the stream stands, while it decodes the samples `decoding` of the source.
    fn progress(&self, decoding: Option<(usize, usize)>) -> StreamEvent {
        let sr = Self::sample_rate();
        StreamEvent::Progress(StreamProgress {
            frame: ProgressFrame {
                transcribed_to: self.origin.min(self.received) as f64 / sr,
                decoding: decoding.map(|(start, end)| (start as f64 / sr, end as f64 / sr)),
                phase: match self.language {
                    StreamLanguage::Detecting if self.segment_count == 0 => {
                        ProgressPhase::DetectingLanguage
                    }
                    StreamLanguage::Detecting
                    | StreamLanguage::Chosen(_)
                    | StreamLanguage::Locked(_) => ProgressPhase::Transcribing,
                },
            },
            received: self.received as f64 / sr,
            source_ended: self.ended,
        })
    }

    fn send(&self, event: StreamEvent) {
        if self.out.send(Ok(event)).is_err() {
            debug!("Receiver dropped, event not delivered");
        }
    }

    fn decode(&mut self, window: &[f32], language: &LanguageRequest) -> Result<Recognised> {
        let started = Instant::now();
        let decoded = self
            .decoder
            .decode(DecodeRequest {
                pcm: window,
                language,
                timing: self.timing,
                cancel: &self.cancel,
            })
            .map_err(|source| TranscriptionError::Decode { source })?;
        self.last_decode = Some(started.elapsed());
        let language = decoded.hypothesis.language.clone();
        Ok(Recognised {
            parts: self.segmenter.parts(decoded.hypothesis, window.len()),
            language,
            completion: decoded.completion,
        })
    }

    /// Drops `n` samples from the front of the pending audio.
    fn consume(&mut self, n: usize) {
        self.pending.drain(..n);
        self.origin += n;
        self.seen = self.seen.saturating_sub(n);
        self.tentative_mark = self.tentative_mark.saturating_sub(n);
    }

    /// Whether the stream, waiting for its source, should decode its pending audio tentatively.
    pub fn tentative_due(&self) -> bool {
        let sr = Self::sample_rate();
        let len = self.pending.len();
        let ready_at = match self.policy {
            WindowPolicy::AtPauses { step, max } => max.min(self.seen + step),
            WindowPolicy::Full { max } => max,
            WindowPolicy::Whole => return false,
        };
        let arrival_rate = self.first_audio.and_then(|first| {
            let wall = first.elapsed().as_secs_f64();
            (wall > 0.0).then(|| self.received as f64 / sr / wall)
        });
        tentative::due(
            self.tentative_text,
            Waiting {
                new_audio: len.saturating_sub(self.tentative_mark) as f64 / sr,
                pending: len as f64 / sr,
                until_window: ready_at.saturating_sub(len) as f64 / sr,
                arrival_rate,
                last_decode: self.last_decode.map(|decode| decode.as_secs_f64()),
            },
        )
    }

    /// One tentative chunk holding the text of `parts`, decoded from the first `len` pending
    /// samples; it takes the index of the next committed chunk, which replaces it.
    fn tentative_chunk(
        &self,
        parts: &[&Part],
        len: usize,
        language: Option<LanguageTag>,
    ) -> StreamingChunk {
        let sr = Self::sample_rate();
        let origin_cs = u32::try_from(self.origin / SAMPLES_PER_CENTISECOND).unwrap_or(u32::MAX);
        let mut text = String::new();
        let mut words = Vec::new();
        for part in parts {
            if !text.is_empty() {
                text.push(' ');
            }
            let offset = u32::try_from(text.len()).unwrap_or(u32::MAX);
            text.push_str(&part.text);
            words.extend(
                part.words
                    .iter()
                    .cloned()
                    .map(|word| word.shifted(offset).delayed(origin_cs)),
            );
        }
        let start = parts.first().map_or(len, |part| part.span.start.min(len));
        let end = parts
            .last()
            .map_or(len, |part| part.span.end.min(len))
            .max(start);
        StreamingChunk {
            text,
            start: (self.origin + start) as f64 / sr,
            end: (self.origin + end) as f64 / sr,
            is_final: false,
            chunk_index: self.chunk_index,
            final_stats: None,
            language,
            kind: ChunkKind::Tentative,
            words,
        }
    }

    /// Decodes all the pending audio only to show it; the stream's position, counters and
    /// language lock stay as they were.
    pub fn tentative(&mut self) -> Result<StreamEvent> {
        let window = self.pending.clone();
        let recognised = self.decode(&window, &self.language.request())?;
        self.tentative_mark = window.len();
        let parts: Vec<&Part> = recognised
            .parts
            .iter()
            .filter(|d| !d.text.is_empty())
            .collect();
        let reported = match &self.language {
            StreamLanguage::Locked(tag) => Some(tag.clone()),
            StreamLanguage::Chosen(_) | StreamLanguage::Detecting => None,
        };
        Ok(StreamEvent::Chunk(self.tentative_chunk(
            &parts,
            window.len(),
            reported,
        )))
    }

    pub fn process(&mut self, kind: WindowKind) -> Result<Vec<StreamEvent>> {
        let mut kind = kind;
        let mut len = match (kind, self.policy) {
            (WindowKind::Full, WindowPolicy::Full { max } | WindowPolicy::AtPauses { max, .. }) => {
                max
            }
            (WindowKind::Full, WindowPolicy::Whole)
            | (WindowKind::Growing | WindowKind::Paused | WindowKind::Last, _) => {
                self.pending.len()
            }
        };
        let mut window: Vec<f32> = self.pending.iter().take(len).copied().collect();
        let mut spoke = len >= self.margins.min;
        let mut speech_spans: Option<Vec<Span>> = None;

        if let Some(detector) = self.speech.as_mut() {
            let speech = if spoke {
                detector
                    .speech(&window)
                    .map_err(|source| TranscriptionError::Detect { source })?
            } else {
                Vec::new()
            };
            self.seen = self.seen.max(len);
            spoke = !speech.is_empty() && len >= self.margins.min;
            let paused = if spoke && kind != WindowKind::Last {
                last_pause(&speech, len, self.margins.min_paused, self.margins.commit)
            } else {
                None
            };
            speech_spans = Some(speech);
            if spoke && kind != WindowKind::Last {
                match paused {
                    Some(cut) => {
                        len = cut;
                        window.truncate(cut);
                        kind = WindowKind::Paused;
                    }
                    None if kind == WindowKind::Growing => return Ok(Vec::new()),
                    None => {}
                }
            }
        }

        let (mut decoded, language) = if spoke {
            let recognised = loop {
                self.send(self.progress(Some((self.origin, self.origin + len))));
                let recognised = self.decode(&window, &self.language.request())?;
                match recognised.completion {
                    Completion::Complete => break recognised,
                    Completion::Cancelled => {
                        kind = WindowKind::Last;
                        break recognised;
                    }
                    Completion::Truncated | Completion::Repeating => {
                        let shorter = shorter(
                            speech_spans.as_deref(),
                            len,
                            self.margins.retry_floor,
                            self.margins.commit,
                        );
                        let Some(shorter) = shorter else {
                            warn!(
                                "The engine stopped early on a window of {:.1}s ({:?}); keeping what it gave",
                                len as f64 / Self::sample_rate(),
                                recognised.completion
                            );
                            break recognised;
                        };
                        debug!(
                            "The engine stopped early on a window of {:.1}s ({:?}); decoding its first {:.1}s",
                            len as f64 / Self::sample_rate(),
                            recognised.completion,
                            shorter.len as f64 / Self::sample_rate()
                        );
                        len = shorter.len;
                        kind = shorter.kind;
                        window.truncate(len);
                    }
                }
            };
            let spoken = recognised.parts.iter().any(|d| !d.text.is_empty());
            let language = recognised.language.map(|estimate| {
                let confidence = estimate.confidence;
                if let Some(locked) = self.language.observe(&estimate.tag, confidence, spoken) {
                    debug!("Streaming language locked to {locked} (confidence {confidence:?})");
                }
                estimate.tag
            });
            (recognised.parts, language)
        } else {
            (Vec::new(), None)
        };

        let spans: Vec<Span> = decoded.iter().map(|d| d.span).collect();
        let decision = commit(&spans, len, self.margins.commit, kind);
        debug!(
            "Window at {:.2}s of {:.2}s ({kind:?}): {} of {} segments committed, next at {:.2}s",
            self.origin as f64 / Self::sample_rate(),
            len as f64 / Self::sample_rate(),
            decision.segments,
            decoded.len(),
            (self.origin + decision.advance) as f64 / Self::sample_rate()
        );

        let tail = decoded.split_off(decision.segments);

        let reported = match &self.language {
            StreamLanguage::Locked(tag) => Some(tag.clone()),
            StreamLanguage::Detecting if kind == WindowKind::Last => language,
            StreamLanguage::Chosen(_) | StreamLanguage::Detecting => None,
        };
        let sr = Self::sample_rate();
        let origin_cs = u32::try_from(self.origin / SAMPLES_PER_CENTISECOND).unwrap_or(u32::MAX);
        let whole = self.policy == WindowPolicy::Whole;
        let mut chunks = Vec::new();
        for d in decoded.iter().filter(|d| !d.text.is_empty()) {
            let (start, end) = if whole {
                (d.span.start, d.span.end)
            } else {
                let end = d.span.end.min(len);
                (d.span.start.min(end), end)
            };
            chunks.push(StreamingChunk {
                text: d.text.clone(),
                start: (self.origin + start) as f64 / sr,
                end: (self.origin + end) as f64 / sr,
                is_final: false,
                chunk_index: self.chunk_index,
                final_stats: None,
                language: reported.clone(),
                kind: ChunkKind::Committed,
                words: d
                    .words
                    .iter()
                    .cloned()
                    .map(|word| word.delayed(origin_cs))
                    .collect(),
            });
            self.chunk_index += 1;
            self.segment_count += 1;
            self.word_count += d.text.split_whitespace().count();
        }

        if kind == WindowKind::Full && self.tentative_text != TentativeText::Off {
            let parts: Vec<&Part> = tail.iter().filter(|d| !d.text.is_empty()).collect();
            chunks.push(self.tentative_chunk(&parts, len, reported.clone()));
        }

        self.consume(decision.advance);

        if kind == WindowKind::Last {
            self.finished = true;
            let end = self.received as f64 / sr;
            if chunks.is_empty() && self.received > 0 {
                chunks.push(StreamingChunk {
                    text: String::new(),
                    start: end,
                    end,
                    is_final: false,
                    chunk_index: self.chunk_index,
                    final_stats: None,
                    language: reported,
                    kind: ChunkKind::Committed,
                    words: Vec::new(),
                });
                self.chunk_index += 1;
            }
            if let Some(last) = chunks.last_mut() {
                last.is_final = true;
                last.final_stats = Some(TranscriptionStats {
                    processing_time: self.started.elapsed().as_secs_f64(),
                    audio_duration: end as f32,
                    segment_count: self.segment_count,
                    word_count: self.word_count,
                });
            }
        }
        let mut events: Vec<StreamEvent> = chunks.into_iter().map(StreamEvent::Chunk).collect();
        events.push(self.progress(None));
        Ok(events)
    }
}
