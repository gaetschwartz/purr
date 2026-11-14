//! The pipeline driven by scripted engines.

use std::{collections::VecDeque, sync::Arc, time::Duration};

use futures::StreamExt;
use parking_lot::Mutex;
use purr_common::{
    language::tag::LanguageTag, model::profile::InputLimit, platform::ChunkKind, word::Word,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::{
    AudioStream,
    config::{RunConfig, Windowing},
    engine::{
        cancel::CancelToken,
        decoder::{DecodeRequest, LanguageRequest, WindowDecoder, WindowLimits},
        error::{DecodeError, DetectError},
        hypothesis::{
            Completion, Decoded, Hypothesis, LanguageEstimate, Segment, Segmentation, Span,
        },
        speech::SpeechDetector,
    },
    error::{Error, Result, TranscriptionError},
};

use super::{
    commit::WindowKind,
    driver::drive,
    events::{StreamEvent, StreamingChunk},
    policy::{SAMPLE_RATE, WindowPolicy, samples},
    session::{Decoding, Events, Session},
};

const S: usize = SAMPLE_RATE;

/// What a scripted decoder was asked.
#[derive(Debug, Clone, PartialEq)]
struct Asked {
    samples: usize,
    language: LanguageRequest,
}

/// Answers each decode with the next scripted outcome, and records what it was asked.
struct ScriptedDecoder {
    script: VecDeque<std::result::Result<Decoded, DecodeError>>,
    asked: Arc<Mutex<Vec<Asked>>>,
    limits: WindowLimits,
}

const UNLIMITED: WindowLimits = WindowLimits {
    input: InputLimit::Unbounded,
    window: Duration::from_secs(30),
};

impl WindowDecoder for ScriptedDecoder {
    fn limits(&self) -> WindowLimits {
        self.limits
    }

    fn supports(&self, _language: &LanguageTag) -> bool {
        true
    }

    fn decode(&mut self, request: DecodeRequest<'_>) -> std::result::Result<Decoded, DecodeError> {
        self.asked.lock().push(Asked {
            samples: request.pcm.len(),
            language: request.language.clone(),
        });
        self.script
            .pop_front()
            .unwrap_or_else(|| Ok(decoded(Vec::new())))
    }
}

struct ScriptedDetector {
    script: VecDeque<Vec<Span>>,
}

impl SpeechDetector for ScriptedDetector {
    fn speech(&mut self, _pcm: &[f32]) -> std::result::Result<Vec<Span>, DetectError> {
        Ok(self.script.pop_front().unwrap_or_default())
    }
}

fn at(seconds: f32) -> usize {
    (seconds * S as f32) as usize
}

fn span(start: f32, end: f32) -> Span {
    Span {
        start: at(start),
        end: at(end),
    }
}

fn segment(text: &str, start: f32, end: f32) -> Segment {
    Segment {
        text: text.to_owned(),
        span: Some(span(start, end)),
        words: Vec::new(),
    }
}

fn decoded(segments: Vec<Segment>) -> Decoded {
    Decoded {
        hypothesis: Hypothesis {
            segments,
            language: None,
            segmentation: Segmentation::Native,
        },
        completion: Completion::Complete,
    }
}

fn detected(segments: Vec<Segment>, tag: &str, confidence: Option<f32>) -> Decoded {
    let mut decoded = decoded(segments);
    decoded.hypothesis.language = Some(LanguageEstimate {
        tag: tag.parse().unwrap(),
        confidence,
    });
    decoded
}

struct Harness {
    session: Session,
    asked: Arc<Mutex<Vec<Asked>>>,
    tx: Events,
    events: mpsc::UnboundedReceiver<Result<StreamEvent>>,
}

fn harness(
    policy: WindowPolicy,
    script: Vec<std::result::Result<Decoded, DecodeError>>,
    speech: Option<Vec<Vec<Span>>>,
    run: &RunConfig,
) -> Harness {
    limited_harness(UNLIMITED, policy, script, speech, run)
}

fn limited_harness(
    limits: WindowLimits,
    policy: WindowPolicy,
    script: Vec<std::result::Result<Decoded, DecodeError>>,
    speech: Option<Vec<Vec<Span>>>,
    run: &RunConfig,
) -> Harness {
    let asked = Arc::new(Mutex::new(Vec::new()));
    let (tx, events) = mpsc::unbounded_channel();
    let session = Session::new(
        run,
        Decoding {
            decoder: Box::new(ScriptedDecoder {
                script: script.into(),
                asked: Arc::clone(&asked),
                limits,
            }),
            speech: speech.map(|script| {
                Box::new(ScriptedDetector {
                    script: script.into(),
                }) as Box<dyn SpeechDetector>
            }),
            policy,
            cancel: CancelToken::new(),
        },
        tx.clone(),
    );
    Harness {
        session,
        asked,
        tx,
        events,
    }
}

fn chunks(events: Vec<StreamEvent>) -> Vec<StreamingChunk> {
    events
        .into_iter()
        .filter_map(StreamEvent::into_chunk)
        .collect()
}

fn texts(chunks: &[StreamingChunk]) -> Vec<(&str, ChunkKind, f64, f64)> {
    chunks
        .iter()
        .map(|chunk| (chunk.text.as_str(), chunk.kind, chunk.start, chunk.end))
        .collect()
}

const FULL: WindowPolicy = WindowPolicy::Full { max: 10 * S };
const AT_PAUSES: WindowPolicy = WindowPolicy::AtPauses {
    step: 4 * S,
    max: 30 * S,
};

#[test]
fn a_full_window_commits_what_ends_before_the_margin_and_keeps_the_rest() -> Result<()> {
    let mut h = harness(
        FULL,
        vec![Ok(decoded(vec![
            segment("one", 0.0, 3.0),
            segment("two", 3.0, 6.5),
            segment("three", 6.5, 9.8),
        ]))],
        None,
        &RunConfig::new(),
    );
    h.session.push(&vec![0.0; 10 * S]);
    assert_eq!(h.session.next_window(false), Some(WindowKind::Full));
    let committed = chunks(h.session.process(WindowKind::Full)?);
    assert_eq!(
        texts(&committed),
        [
            ("one", ChunkKind::Committed, 0.0, 3.0),
            ("two", ChunkKind::Committed, 3.0, 6.5)
        ]
    );

    h.session.push(&vec![0.0; 2 * S]);
    assert_eq!(h.session.next_window(true), Some(WindowKind::Last));
    let last = chunks(h.session.process(WindowKind::Last)?);
    assert_eq!(
        h.asked.lock().last().map(|asked| asked.samples),
        Some(at(5.5))
    );
    assert!(last.last().is_some_and(|chunk| chunk.is_final));
    Ok(())
}

#[test]
fn a_growing_window_without_a_pause_waits_without_decoding() -> Result<()> {
    let mut h = harness(
        AT_PAUSES,
        Vec::new(),
        Some(vec![vec![span(0.2, 4.9)]]),
        &RunConfig::new(),
    );
    h.session.push(&vec![0.0; 5 * S]);
    assert_eq!(h.session.next_window(false), Some(WindowKind::Growing));
    assert!(h.session.process(WindowKind::Growing)?.is_empty());
    assert!(h.asked.lock().is_empty());
    assert_eq!(h.session.next_window(false), None);
    Ok(())
}

#[test]
fn a_window_cut_at_a_pause_commits_everything_before_the_cut() -> Result<()> {
    let mut h = harness(
        AT_PAUSES,
        vec![Ok(decoded(vec![
            segment("hello", 0.5, 2.0),
            segment("there", 3.0, 4.4),
        ]))],
        Some(vec![vec![span(0.5, 2.0), span(3.0, 4.0)]]),
        &RunConfig::new(),
    );
    h.session.push(&vec![0.0; 5 * S]);
    let committed = chunks(h.session.process(WindowKind::Growing)?);
    assert_eq!(
        h.asked.lock().first().map(|asked| asked.samples),
        Some(at(4.5))
    );
    assert_eq!(
        texts(&committed),
        [
            ("hello", ChunkKind::Committed, 0.5, 2.0),
            ("there", ChunkKind::Committed, 3.0, 4.4)
        ]
    );
    Ok(())
}

#[test]
fn untimed_text_in_a_window_cut_at_a_pause_covers_the_window() -> Result<()> {
    let untimed = Decoded {
        hypothesis: Hypothesis {
            segments: vec![Segment {
                text: "no times here".to_owned(),
                span: None,
                words: Vec::new(),
            }],
            language: None,
            segmentation: Segmentation::Synthesize,
        },
        completion: Completion::Complete,
    };
    let mut h = harness(
        AT_PAUSES,
        vec![Ok(untimed)],
        Some(vec![vec![span(0.5, 2.0), span(3.0, 4.0)]]),
        &RunConfig::new(),
    );
    h.session.push(&vec![0.0; 5 * S]);
    let committed = chunks(h.session.process(WindowKind::Growing)?);
    assert_eq!(
        texts(&committed),
        [("no times here", ChunkKind::Committed, 0.0, 4.5)]
    );
    Ok(())
}

#[test]
fn the_last_window_ends_the_stream_even_without_text() -> Result<()> {
    let mut h = harness(FULL, Vec::new(), None, &RunConfig::new());
    h.session.push(&vec![0.0; 3 * S]);
    assert_eq!(h.session.next_window(true), Some(WindowKind::Last));
    let last = chunks(h.session.process(WindowKind::Last)?);
    assert_eq!(texts(&last), [("", ChunkKind::Committed, 3.0, 3.0)]);
    assert!(
        last.iter()
            .all(|chunk| chunk.is_final && chunk.final_stats.is_some())
    );
    assert!(h.session.finished);
    assert_eq!(h.session.next_window(true), None);
    Ok(())
}

#[test]
fn a_whole_source_is_decoded_once_it_ended_and_keeps_the_engine_spans() -> Result<()> {
    let mut h = harness(
        WindowPolicy::Whole,
        vec![Ok(decoded(vec![
            segment("early", 0.0, 30.0),
            segment("late", 59.0, 61.0),
        ]))],
        None,
        &RunConfig::new(),
    );
    h.session.push(&vec![0.0; 60 * S]);
    assert_eq!(h.session.next_window(false), None);
    assert_eq!(h.session.next_window(true), Some(WindowKind::Last));
    let all = chunks(h.session.process(WindowKind::Last)?);
    assert_eq!(
        h.asked.lock().first().map(|asked| asked.samples),
        Some(60 * S)
    );
    assert_eq!(
        texts(&all),
        [
            ("early", ChunkKind::Committed, 0.0, 30.0),
            ("late", ChunkKind::Committed, 59.0, 61.0)
        ]
    );
    Ok(())
}

#[test]
fn a_confident_detection_locks_the_language_for_the_next_windows() -> Result<()> {
    let mut h = harness(
        FULL,
        vec![
            Ok(detected(vec![segment("hello", 0.0, 2.0)], "en", Some(0.9))),
            Ok(decoded(vec![segment("again", 0.0, 2.0)])),
        ],
        None,
        &RunConfig::new(),
    );
    h.session.push(&vec![0.0; 10 * S]);
    let first = chunks(h.session.process(WindowKind::Full)?);
    h.session.push(&vec![0.0; 2 * S]);
    let last = chunks(h.session.process(WindowKind::Last)?);
    let en: LanguageTag = "en".parse().unwrap();
    assert_eq!(
        first.first().and_then(|chunk| chunk.language.clone()),
        Some(en.clone())
    );
    assert_eq!(
        h.asked
            .lock()
            .iter()
            .map(|asked| asked.language.clone())
            .collect::<Vec<_>>(),
        [LanguageRequest::Detect, LanguageRequest::Use(en.clone())]
    );
    assert_eq!(
        last.first().and_then(|chunk| chunk.language.clone()),
        Some(en)
    );
    Ok(())
}

#[test]
fn an_unsure_detection_keeps_detecting_and_a_silent_confidence_locks() -> Result<()> {
    let mut h = harness(
        FULL,
        vec![
            Ok(detected(vec![segment("hm", 0.0, 2.0)], "nn", Some(0.4))),
            Ok(detected(vec![segment("bonjour", 0.0, 2.0)], "fr", None)),
        ],
        None,
        &RunConfig::new(),
    );
    h.session.push(&vec![0.0; 10 * S]);
    let unsure = chunks(h.session.process(WindowKind::Full)?);
    assert_eq!(
        unsure.first().and_then(|chunk| chunk.language.clone()),
        None
    );
    h.session.push(&vec![0.0; 10 * S]);
    let locked = chunks(h.session.process(WindowKind::Full)?);
    assert_eq!(
        locked.first().and_then(|chunk| chunk.language.clone()),
        Some("fr".parse().unwrap())
    );
    assert_eq!(
        h.asked
            .lock()
            .iter()
            .map(|asked| asked.language.clone())
            .collect::<Vec<_>>(),
        [LanguageRequest::Detect, LanguageRequest::Detect]
    );
    Ok(())
}

#[test]
fn a_cancelled_decode_ends_the_stream_with_what_the_engine_finished() -> Result<()> {
    let mut partial = decoded(vec![segment("so far", 0.0, 4.0)]);
    partial.completion = Completion::Cancelled;
    let mut h = harness(FULL, vec![Ok(partial)], None, &RunConfig::new());
    h.session.push(&vec![0.0; 20 * S]);
    let last = chunks(h.session.process(WindowKind::Full)?);
    assert_eq!(texts(&last), [("so far", ChunkKind::Committed, 0.0, 4.0)]);
    assert!(last.iter().all(|chunk| chunk.is_final));
    assert!(h.session.finished);
    Ok(())
}

#[test]
fn words_are_moved_onto_the_source_timeline() -> Result<()> {
    let word = Word {
        range: 0..5,
        start_cs: Some(50),
        end_cs: Some(120),
        score: Some(200),
    };
    let mut h = harness(
        FULL,
        vec![
            Ok(decoded(vec![segment("first", 0.0, 8.0)])),
            Ok(decoded(vec![Segment {
                words: vec![word.clone()],
                ..segment("later", 0.5, 1.2)
            }])),
        ],
        None,
        &RunConfig::new(),
    );
    h.session.push(&vec![0.0; 10 * S]);
    h.session.process(WindowKind::Full)?;
    h.session.push(&vec![0.0; S]);
    let last = chunks(h.session.process(WindowKind::Last)?);
    assert_eq!(
        last.first().map(|chunk| chunk.words.clone()),
        Some(vec![word.delayed(800)])
    );
    Ok(())
}

#[tokio::test]
async fn an_engine_error_mid_stream_ends_it_with_the_typed_error() {
    let h = harness(
        FULL,
        vec![
            Ok(decoded(vec![segment("fine", 0.0, 3.0)])),
            Err(DecodeError::OutOfMemory),
        ],
        None,
        &RunConfig::new(),
    );
    let Harness {
        session,
        tx,
        events,
        ..
    } = h;
    drive(session, AudioStream::from_samples(&vec![0.0; 25 * S]), tx).await;
    let received: Vec<Result<StreamEvent>> = UnboundedReceiverStream::new(events).collect().await;
    let chunks: Vec<String> = received
        .iter()
        .filter_map(|event| match event {
            Ok(StreamEvent::Chunk(chunk)) => Some(chunk.text.clone()),
            Ok(StreamEvent::Progress(_)) | Err(_) => None,
        })
        .collect();
    assert_eq!(chunks, ["fine"]);
    assert!(matches!(
        received.last(),
        Some(Err(Error::Transcription {
            source: TranscriptionError::Decode {
                source: DecodeError::OutOfMemory
            }
        }))
    ));
}

fn stopped(completion: Completion, partial: &str) -> Decoded {
    Decoded {
        hypothesis: Hypothesis {
            segments: vec![Segment {
                text: partial.to_owned(),
                span: None,
                words: Vec::new(),
            }],
            language: None,
            segmentation: Segmentation::Native,
        },
        completion,
    }
}

fn asked_samples(h: &Harness) -> Vec<usize> {
    h.asked.lock().iter().map(|asked| asked.samples).collect()
}

#[test]
fn a_window_the_engine_stopped_early_on_is_decoded_again_at_half_the_length() -> Result<()> {
    let mut h = harness(
        FULL,
        vec![
            Ok(stopped(Completion::Truncated, "never shown")),
            Ok(decoded(vec![segment("hello", 0.0, 4.0)])),
        ],
        None,
        &RunConfig::new(),
    );
    h.session.push(&vec![0.0; 10 * S]);
    let committed = chunks(h.session.process(WindowKind::Full)?);
    assert_eq!(asked_samples(&h), [10 * S, 5 * S]);
    assert_eq!(
        texts(&committed),
        [("hello", ChunkKind::Committed, 0.0, 4.0)]
    );
    assert!(!h.session.finished);
    Ok(())
}

#[test]
fn a_window_too_short_to_cut_again_commits_the_partial_text_and_is_consumed() -> Result<()> {
    let mut h = harness(
        FULL,
        vec![Ok(stopped(Completion::Repeating, "so far"))],
        None,
        &RunConfig::new(),
    );
    h.session.push(&vec![0.0; 8 * S]);
    let last = chunks(h.session.process(WindowKind::Last)?);
    assert_eq!(asked_samples(&h), [8 * S]);
    assert_eq!(texts(&last), [("so far", ChunkKind::Committed, 0.0, 8.0)]);
    assert!(last.iter().all(|chunk| chunk.is_final));
    assert!(h.session.finished);
    Ok(())
}

#[test]
fn a_last_window_that_is_cut_again_does_not_end_the_stream_before_the_rest_is_decoded() -> Result<()>
{
    let mut h = harness(
        WindowPolicy::Full { max: 20 * S },
        vec![
            Ok(stopped(Completion::Truncated, "never shown")),
            Ok(decoded(vec![segment("first", 0.0, 7.4)])),
            Ok(decoded(vec![segment("second", 0.0, 7.6)])),
        ],
        None,
        &RunConfig::new(),
    );
    h.session.push(&vec![0.0; 15 * S]);
    assert_eq!(h.session.next_window(true), Some(WindowKind::Last));
    let first = chunks(h.session.process(WindowKind::Last)?);
    assert_eq!(texts(&first), [("first", ChunkKind::Committed, 0.0, 7.4)]);
    assert!(first.iter().all(|chunk| !chunk.is_final));
    assert!(!h.session.finished);
    assert_eq!(h.session.next_window(true), Some(WindowKind::Last));
    let second = chunks(h.session.process(WindowKind::Last)?);
    assert_eq!(second.len(), 1);
    assert!(second.iter().all(|chunk| chunk.is_final));
    assert_eq!(asked_samples(&h), [15 * S, at(7.5), at(7.6)]);
    assert!(h.session.finished);
    Ok(())
}

#[test]
fn a_window_cut_again_ends_in_the_last_pause_of_its_first_half() -> Result<()> {
    let speech = vec![span(0.5, 6.0), span(8.0, 14.0), span(14.5, 25.0)];
    let mut h = harness(
        AT_PAUSES,
        vec![
            Ok(stopped(Completion::Truncated, "never shown")),
            Ok(decoded(vec![
                segment("one", 0.5, 6.0),
                segment("two", 8.0, 6.9),
            ])),
        ],
        Some(vec![speech]),
        &RunConfig::new(),
    );
    h.session.push(&vec![0.0; 30 * S]);
    let committed = chunks(h.session.process(WindowKind::Full)?);
    assert_eq!(asked_samples(&h), [at(27.5), at(7.0)]);
    assert_eq!(committed.len(), 2);
    Ok(())
}

#[tokio::test]
async fn a_whole_source_goes_to_a_decoder_with_a_limit_in_windows_no_longer_than_it() {
    let limit = Duration::from_secs(20);
    let limits = WindowLimits {
        input: InputLimit::UpTo(limit),
        window: Duration::from_secs(30),
    };
    let run = RunConfig::new().with_windowing(Windowing::Whole);
    let policy = WindowPolicy::choose(run.windowing, limits, false, samples(run.chunk_size));
    assert_eq!(policy, WindowPolicy::Full { max: 20 * S });
    let h = limited_harness(limits, policy, Vec::new(), None, &run);
    let Harness {
        session,
        asked,
        tx,
        events,
    } = h;
    drive(session, AudioStream::from_samples(&vec![0.0; 70 * S]), tx).await;
    let received: Vec<Result<StreamEvent>> = UnboundedReceiverStream::new(events).collect().await;
    let sizes: Vec<usize> = asked.lock().iter().map(|asked| asked.samples).collect();
    assert!(sizes.len() >= 4, "{sizes:?}");
    assert!(sizes.iter().all(|samples| *samples <= 20 * S), "{sizes:?}");
    let last = received
        .into_iter()
        .filter_map(|event| match event {
            Ok(event) => event.into_chunk(),
            Err(_) => None,
        })
        .last();
    assert!(last.is_some_and(|chunk| chunk.is_final));
}
