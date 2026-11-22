//! The pipeline driven by scripted engines.

use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use futures::StreamExt;
use parking_lot::Mutex;
use purr_common::{
    devices::DeviceOrder,
    language::tag::LanguageTag,
    model::{engine::EngineId, profile::InputLimit},
    platform::ChunkKind,
    word::Word,
};
use tokio::{
    sync::{mpsc, watch},
    task::JoinHandle,
};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::{
    AudioChunk, AudioStream, Pacing, SourceLength,
    accel::build_info::BuildInfo,
    config::{ModelRef, RunConfig, TentativeText, Windowing},
    dev::Device,
    engine::{
        Engine, OpenSpec,
        cancel::CancelToken,
        decoder::{DecodeRequest, LanguageRequest, WindowDecoder, WindowLimits},
        error::{DecodeError, DetectError, OpenError},
        hypothesis::{
            Completion, Decoded, Hypothesis, LanguageEstimate, Segment, Segmentation, Span,
        },
        speech::SpeechDetector,
    },
    error::{Error, Result, TranscriptionError},
};

use super::{
    commit::WindowKind,
    driver::{EndSlot, Ending, Request, drive},
    events::{EndReason, StreamEvent, StreamingChunk, Transcription, TranscriptionStats},
    model::LoadedModel,
    policy::{SAMPLE_RATE, WindowPolicy, samples},
    run::Input,
    session::{Decoding, Events, MAX_PENDING_SAMPLES, Session},
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

struct BlockDetector;

impl SpeechDetector for BlockDetector {
    fn speech(&mut self, pcm: &[f32]) -> std::result::Result<Vec<Span>, DetectError> {
        let len = pcm.len();
        Ok((0..len.div_ceil(6 * S))
            .map(|block| block * 6 * S + S / 2)
            .filter(|start| *start < len)
            .map(|start| Span {
                start,
                end: (start + 9 * S / 2).min(len),
            })
            .collect())
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
    paced_harness(
        Pacing::Live,
        limits,
        policy,
        script,
        speech.map(|script| {
            Box::new(ScriptedDetector {
                script: script.into(),
            }) as Box<dyn SpeechDetector>
        }),
        run,
    )
}

fn paced_harness(
    pacing: Pacing,
    limits: WindowLimits,
    policy: WindowPolicy,
    script: Vec<std::result::Result<Decoded, DecodeError>>,
    speech: Option<Box<dyn SpeechDetector>>,
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
            speech,
            policy,
            pacing,
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
        mut session,
        tx,
        events,
        ..
    } = h;
    let (_request, ending) = ending(None);
    drive(
        &mut session,
        AudioStream::from_samples(&vec![0.0; 25 * S]),
        &tx,
        ending,
    )
    .await;
    drop(tx);
    drop(session);
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
        mut session,
        asked,
        tx,
        events,
    } = h;
    let (_request, ending) = ending(None);
    drive(
        &mut session,
        AudioStream::from_samples(&vec![0.0; 70 * S]),
        &tx,
        ending,
    )
    .await;
    drop(tx);
    drop(session);
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

#[test]
fn wants_audio_is_false_when_streaming_and_pending_is_full() {
    for policy in [FULL, AT_PAUSES] {
        let mut h = harness(policy, Vec::new(), None, &RunConfig::new());
        h.session.push(&vec![0.0; MAX_PENDING_SAMPLES - 1]);
        assert!(h.session.wants_audio());
        h.session.push(&[0.0]);
        assert!(!h.session.wants_audio());
    }
}

#[test]
fn wants_audio_is_true_for_whole_windowing_whatever_is_pending() {
    let mut h = harness(WindowPolicy::Whole, Vec::new(), None, &RunConfig::new());
    h.session.push(&vec![0.0; MAX_PENDING_SAMPLES + S]);
    assert!(h.session.wants_audio());
}

fn ending(listen: Option<usize>) -> (watch::Sender<Request>, Ending) {
    let (request, receiver) = watch::channel(Request::Run);
    (
        request,
        Ending {
            request: receiver,
            listen,
            reason: EndSlot::default(),
        },
    )
}

struct EchoDecoder {
    panics: bool,
}

impl WindowDecoder for EchoDecoder {
    fn limits(&self) -> WindowLimits {
        UNLIMITED
    }

    fn supports(&self, _language: &LanguageTag) -> bool {
        true
    }

    fn decode(&mut self, request: DecodeRequest<'_>) -> std::result::Result<Decoded, DecodeError> {
        assert!(!self.panics, "the engine broke");
        let seconds = request.pcm.len() / S;
        Ok(decoded(vec![Segment {
            text: format!("{seconds}"),
            span: Some(Span {
                start: 0,
                end: request.pcm.len(),
            }),
            words: Vec::new(),
        }]))
    }
}

struct EchoEngine {
    opens: AtomicUsize,
    panics: bool,
}

impl EchoEngine {
    fn new(panics: bool) -> Arc<Self> {
        Arc::new(Self {
            opens: AtomicUsize::new(0),
            panics,
        })
    }
}

impl Engine for EchoEngine {
    fn id(&self) -> EngineId {
        EngineId::WhisperCpp
    }

    fn build_info(&self) -> BuildInfo {
        BuildInfo::default()
    }

    fn devices(&self) -> Vec<Device> {
        Vec::new()
    }

    fn open(&self, _spec: &OpenSpec<'_>) -> std::result::Result<Box<dyn WindowDecoder>, OpenError> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(EchoDecoder {
            panics: self.panics,
        }))
    }

    fn open_speech_detector(
        &self,
        _model: &Path,
    ) -> std::result::Result<Box<dyn SpeechDetector>, OpenError> {
        Ok(Box::new(BlockDetector))
    }

    fn install_logging(&self) {}
}

const PATIENCE: Duration = Duration::from_secs(20);

fn echo_run() -> RunConfig {
    RunConfig::new()
        .with_devices(DeviceOrder::cpu_only())
        .with_threads(1)
}

async fn loaded(engine: &Arc<EchoEngine>) -> Result<LoadedModel> {
    let engines: Vec<Arc<dyn Engine>> = vec![Arc::clone(engine) as Arc<dyn Engine>];
    let model = ModelRef::new(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
        EngineId::WhisperCpp,
    );
    tokio::time::timeout(PATIENCE, LoadedModel::load(&model, &echo_run(), &engines))
        .await
        .expect("the model loads in time")
}

fn live() -> (mpsc::Sender<Result<AudioChunk>>, AudioStream) {
    let (tx, rx) = mpsc::channel(8);
    (tx, AudioStream::new(rx, SourceLength::Unknown))
}

fn heard(index: usize, seconds: usize, is_final: bool) -> AudioChunk {
    AudioChunk::new(vec![0.0; seconds * S], index, (index * 10) as f32, is_final)
}

struct Finished {
    texts: Vec<String>,
    stats: Option<TranscriptionStats>,
    end: Option<EndReason>,
}

impl Finished {
    fn audio_seconds(&self) -> Option<f32> {
        self.stats.as_ref().map(|stats| stats.audio_duration)
    }
}

async fn finish(mut transcription: Transcription) -> Finished {
    let mut texts = Vec::new();
    let mut stats = None;
    tokio::time::timeout(PATIENCE, async {
        while let Some(event) = transcription.next().await {
            if let Ok(StreamEvent::Chunk(chunk)) = event
                && chunk.kind == ChunkKind::Committed
            {
                if !chunk.text.is_empty() {
                    texts.push(chunk.text);
                }
                stats = chunk.final_stats.or_else(|| stats.take());
            }
        }
    })
    .await
    .expect("the transcription ends in time");
    Finished {
        texts,
        stats,
        end: transcription.end_reason(),
    }
}

async fn start(
    model: &mut LoadedModel,
    audio: AudioStream,
    run: &RunConfig,
) -> Result<Transcription> {
    tokio::time::timeout(PATIENCE, model.transcribe(Input::Audio(audio), run))
        .await
        .expect("the transcription starts in time")
}

async fn run_once(
    model: &mut LoadedModel,
    audio: AudioStream,
    run: &RunConfig,
) -> Result<Finished> {
    Ok(finish(start(model, audio, run).await?).await)
}

fn silence(seconds: usize) -> AudioStream {
    AudioStream::from_samples(&vec![0.0; seconds * S])
}

#[tokio::test]
async fn loaded_model_runs_two_inputs_with_one_load() -> Result<()> {
    let engine = EchoEngine::new(false);
    let mut model = loaded(&engine).await?;
    let run = echo_run();
    let first = run_once(&mut model, silence(12), &run).await?;
    let second = run_once(&mut model, silence(7), &run).await?;
    assert_eq!(engine.opens.load(Ordering::SeqCst), 1);
    assert_eq!(first.texts, ["12"]);
    assert_eq!(second.texts, ["7"]);
    Ok(())
}

#[tokio::test]
async fn decoder_returns_after_a_cancelled_run() -> Result<()> {
    let engine = EchoEngine::new(false);
    let mut model = loaded(&engine).await?;
    let run = echo_run();
    let (feeder, audio) = live();
    feeder.send(Ok(heard(0, 10, false))).await.expect("fed");
    let first = start(&mut model, audio, &run).await?;
    first.cancel();
    let cancelled = finish(first).await;
    assert_eq!(cancelled.end, None);
    let second = run_once(&mut model, silence(5), &run).await?;
    assert_eq!(second.texts, ["5"]);
    assert_eq!(engine.opens.load(Ordering::SeqCst), 1);
    drop(feeder);
    Ok(())
}

#[tokio::test]
async fn a_decoder_lost_with_its_task_is_a_typed_error() -> Result<()> {
    let engine = EchoEngine::new(true);
    let mut model = loaded(&engine).await?;
    let run = echo_run();
    let first = run_once(&mut model, silence(5), &run).await?;
    assert_eq!(first.texts, Vec::<String>::new());
    for _ in 0..2 {
        let next = start(&mut model, silence(1), &run).await;
        assert!(matches!(
            next,
            Err(Error::Transcription {
                source: TranscriptionError::DecoderLost
            })
        ));
    }
    Ok(())
}

#[tokio::test]
async fn end_reason_is_source_ended_for_a_finite_source() -> Result<()> {
    let engine = EchoEngine::new(false);
    let mut model = loaded(&engine).await?;
    let done = run_once(&mut model, silence(12), &echo_run()).await?;
    assert_eq!(done.end, Some(EndReason::SourceEnded));
    Ok(())
}

#[tokio::test]
async fn listen_for_ends_a_source_of_unknown_length() -> Result<()> {
    let engine = EchoEngine::new(false);
    let mut model = loaded(&engine).await?;
    let run = echo_run().with_listen_for(Some(Duration::from_secs(30)));
    let (feeder, audio) = live();
    for index in 0..5 {
        feeder.send(Ok(heard(index, 10, false))).await.expect("fed");
    }
    let done = run_once(&mut model, audio, &run).await?;
    assert_eq!(done.end, Some(EndReason::DurationReached));
    assert_eq!(done.audio_seconds(), Some(30.0));
    drop(feeder);
    Ok(())
}

#[tokio::test]
async fn listen_for_cuts_inside_a_chunk() -> Result<()> {
    let engine = EchoEngine::new(false);
    let mut model = loaded(&engine).await?;
    let run = echo_run().with_listen_for(Some(Duration::from_secs(25)));
    let (feeder, audio) = live();
    for index in 0..4 {
        feeder.send(Ok(heard(index, 10, false))).await.expect("fed");
    }
    let done = run_once(&mut model, audio, &run).await?;
    assert_eq!(done.end, Some(EndReason::DurationReached));
    assert_eq!(done.audio_seconds(), Some(25.0));
    drop(feeder);
    Ok(())
}

#[tokio::test]
async fn listen_for_leaves_a_source_of_known_length_whole() -> Result<()> {
    let engine = EchoEngine::new(false);
    let mut model = loaded(&engine).await?;
    let run = echo_run().with_listen_for(Some(Duration::from_secs(30)));
    let done = run_once(&mut model, silence(60), &run).await?;
    assert_eq!(done.end, Some(EndReason::SourceEnded));
    assert_eq!(done.audio_seconds(), Some(60.0));
    Ok(())
}

#[tokio::test]
async fn listen_for_leaves_an_estimated_length_whole() -> Result<()> {
    let engine = EchoEngine::new(false);
    let mut model = loaded(&engine).await?;
    let run = echo_run().with_listen_for(Some(Duration::from_secs(30)));
    let (tx, rx) = mpsc::channel(8);
    for index in 0..6 {
        tx.send(Ok(heard(index, 10, index == 5)))
            .await
            .expect("fed");
    }
    let audio = AudioStream::new(rx, SourceLength::Estimated(Duration::from_secs(60)));
    let done = run_once(&mut model, audio, &run).await?;
    assert_eq!(done.end, Some(EndReason::SourceEnded));
    assert_eq!(done.audio_seconds(), Some(60.0));
    Ok(())
}

#[tokio::test]
async fn stop_decodes_what_arrived_and_ends() -> Result<()> {
    let engine = EchoEngine::new(false);
    let mut model = loaded(&engine).await?;
    let (feeder, audio) = live();
    feeder.send(Ok(heard(0, 10, false))).await.expect("fed");
    feeder.send(Ok(heard(1, 2, false))).await.expect("fed");
    let transcription = start(&mut model, audio, &echo_run()).await?;
    transcription.stop();
    let done = finish(transcription).await;
    assert_eq!(done.end, Some(EndReason::Stopped));
    assert_eq!(done.audio_seconds(), Some(12.0));
    assert_eq!(done.texts, ["12"]);
    drop(feeder);
    Ok(())
}

#[tokio::test]
async fn stop_wakes_a_run_waiting_for_a_source_that_sends_nothing() -> Result<()> {
    let engine = EchoEngine::new(false);
    let mut model = loaded(&engine).await?;
    let (feeder, audio) = live();
    feeder.send(Ok(heard(0, 3, false))).await.expect("fed");
    let transcription = start(&mut model, audio, &echo_run()).await?;
    tokio::time::sleep(Duration::from_millis(100)).await;
    transcription.stop();
    let done = finish(transcription).await;
    assert_eq!(done.end, Some(EndReason::Stopped));
    assert_eq!(done.audio_seconds(), Some(3.0));
    drop(feeder);
    Ok(())
}

#[tokio::test]
async fn transcription_reports_the_source_length() -> Result<()> {
    let engine = EchoEngine::new(false);
    let mut model = loaded(&engine).await?;
    let run = echo_run();
    let finite = start(&mut model, silence(12), &run).await?;
    assert_eq!(
        finite.length(),
        SourceLength::Exact(Duration::from_secs(12))
    );
    finish(finite).await;
    let (feeder, audio) = live();
    let endless = start(&mut model, audio, &run).await?;
    assert_eq!(endless.length(), SourceLength::Unknown);
    endless.cancel();
    finish(endless).await;
    drop(feeder);
    Ok(())
}

#[test]
fn a_stream_is_paced_by_its_length_unless_told_otherwise() {
    let paced = |length| {
        let (_tx, rx) = mpsc::channel(1);
        AudioStream::new(rx, length)
    };
    let known = Duration::from_secs(60);
    assert_eq!(paced(SourceLength::Exact(known)).pacing(), Pacing::Recorded);
    assert_eq!(
        paced(SourceLength::Estimated(known)).pacing(),
        Pacing::Recorded
    );
    assert_eq!(paced(SourceLength::Unknown).pacing(), Pacing::Live);
    assert_eq!(
        paced(SourceLength::Unknown)
            .with_pacing(Pacing::Recorded)
            .pacing(),
        Pacing::Recorded
    );
    assert_eq!(
        paced(SourceLength::Exact(known))
            .with_pacing(Pacing::Live)
            .pacing(),
        Pacing::Live
    );
    assert_eq!(silence(3).pacing(), Pacing::Recorded);
}

fn block_harness(pacing: Pacing, run: &RunConfig) -> Harness {
    paced_harness(
        pacing,
        UNLIMITED,
        AT_PAUSES,
        Vec::new(),
        Some(Box::new(BlockDetector)),
        run,
    )
}

#[test]
fn a_recorded_source_decides_a_window_only_when_full_or_ended() {
    let mut recorded = block_harness(Pacing::Recorded, &RunConfig::new());
    recorded.session.push(&vec![0.0; 5 * S]);
    assert_eq!(recorded.session.next_window(false), None);
    assert_eq!(recorded.session.next_window(true), Some(WindowKind::Last));
    recorded.session.push(&vec![0.0; 25 * S]);
    assert_eq!(recorded.session.next_window(false), Some(WindowKind::Full));

    let mut live = block_harness(Pacing::Live, &RunConfig::new());
    live.session.push(&vec![0.0; 5 * S]);
    assert_eq!(live.session.next_window(false), Some(WindowKind::Growing));
    live.session.push(&vec![0.0; 25 * S]);
    assert_eq!(live.session.next_window(false), Some(WindowKind::Full));
}

#[test]
fn a_recorded_stream_has_no_growing_window_to_wait_for_before_a_tentative_decode() {
    let run = RunConfig::new().with_tentative(TentativeText::WhileWaiting { every_seconds: 1.0 });
    let due = |pacing| {
        let mut h = block_harness(pacing, &run);
        h.session.push(&vec![0.0; 2 * S]);
        std::thread::sleep(Duration::from_millis(300));
        h.session.tentative_due()
    };
    assert!(!due(Pacing::Live));
    assert!(due(Pacing::Recorded));
}

struct Arrival {
    pieces: Vec<(usize, Duration)>,
    final_marked: bool,
}

impl Arrival {
    fn in_pieces(total: usize, piece: usize, gap: Duration, final_marked: bool) -> Self {
        let pieces = (0..total.div_ceil(piece))
            .map(|index| {
                let delay = if index == 0 { Duration::ZERO } else { gap };
                (piece.min(total - index * piece), delay)
            })
            .collect();
        Self {
            pieces,
            final_marked,
        }
    }

    fn feed(self, length: SourceLength) -> (AudioStream, JoinHandle<()>) {
        let (tx, rx) = mpsc::channel(8);
        let feeder = tokio::spawn(async move {
            let count = self.pieces.len();
            for (index, (samples, delay)) in self.pieces.into_iter().enumerate() {
                tokio::time::sleep(delay).await;
                let chunk = AudioChunk::new(
                    vec![0.0; samples],
                    index,
                    0.0,
                    self.final_marked && index + 1 == count,
                );
                tx.send(Ok(chunk)).await.expect("fed");
            }
        });
        (AudioStream::new(rx, length), feeder)
    }
}

fn pausing_run() -> RunConfig {
    echo_run().with_vad_model(Some(PathBuf::from("vad.bin")))
}

#[tokio::test]
async fn a_recorded_source_is_cut_into_the_same_windows_however_its_audio_arrives() -> Result<()> {
    let engine = EchoEngine::new(false);
    let mut model = loaded(&engine).await?;
    let run = pausing_run();
    let total = 75 * S;
    let length = SourceLength::Exact(Duration::from_secs(75));
    let patterns = [
        Arrival::in_pieces(total, total, Duration::ZERO, false),
        Arrival::in_pieces(total, total, Duration::ZERO, true),
        Arrival {
            pieces: vec![
                (10 * S, Duration::ZERO),
                (65 * S, Duration::from_millis(100)),
            ],
            final_marked: true,
        },
        Arrival::in_pieces(total, S / 2, Duration::from_millis(1), false),
        Arrival::in_pieces(total, 4 * S, Duration::from_millis(20), true),
        Arrival::in_pieces(total, 10 * S, Duration::from_millis(5), false),
    ];
    for (index, arrival) in patterns.into_iter().enumerate() {
        let (audio, feeder) = arrival.feed(length);
        let done = run_once(&mut model, audio, &run).await?;
        feeder.await.expect("the feeder ends");
        assert_eq!(done.texts, ["29", "29", "16"], "pattern {index}");
    }
    Ok(())
}

async fn first_text(transcription: &mut Transcription) -> Option<String> {
    while let Some(event) = transcription.next().await {
        if let Ok(StreamEvent::Chunk(chunk)) = event
            && chunk.kind == ChunkKind::Committed
            && !chunk.text.is_empty()
        {
            return Some(chunk.text);
        }
    }
    None
}

#[tokio::test]
async fn a_live_source_with_a_step_of_audio_pending_is_decoded_before_it_ends() -> Result<()> {
    let engine = EchoEngine::new(false);
    let mut model = loaded(&engine).await?;
    let (feeder, audio) = live();
    feeder.send(Ok(heard(0, 10, false))).await.expect("fed");
    let mut transcription = start(&mut model, audio, &pausing_run()).await?;
    let early = tokio::time::timeout(PATIENCE, first_text(&mut transcription))
        .await
        .expect("a window is decoded before the source ends");
    assert_eq!(early.as_deref(), Some("5"));
    drop(feeder);
    finish(transcription).await;
    Ok(())
}

#[tokio::test]
async fn a_recorded_source_that_stalls_is_not_decoded_before_its_window_is_full() -> Result<()> {
    let engine = EchoEngine::new(false);
    let mut model = loaded(&engine).await?;
    let (feeder, rx) = mpsc::channel(8);
    let audio = AudioStream::new(rx, SourceLength::Exact(Duration::from_secs(75)));
    feeder.send(Ok(heard(0, 10, false))).await.expect("fed");
    let mut transcription = start(&mut model, audio, &pausing_run()).await?;
    let stalled =
        tokio::time::timeout(Duration::from_millis(300), first_text(&mut transcription)).await;
    assert!(stalled.is_err());
    drop(feeder);
    let done = finish(transcription).await;
    assert_eq!(done.texts, ["10"]);
    Ok(())
}
