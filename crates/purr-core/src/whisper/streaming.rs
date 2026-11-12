use crate::{
    AudioStream, ModelManager, TranscriptionConfig,
    accel::report::AccelReport,
    config::TentativeText,
    error::TranscriptionError,
    whisper::{
        StreamEvent, StreamProgress, StreamingChunk, TranscriptionResult, TranscriptionStats,
        WhisperTranscriber,
        beams::{self, BeamCollector},
        commit::{Span, WindowKind, commit},
        language::{LanguageCode, StreamLanguage},
        loader::{SpecialTokens, load_model},
        segments,
        speech::{SpeechDetector, last_pause},
        tentative::{self, Waiting},
    },
};
use futures::{FutureExt, Stream, StreamExt};
use purr_common::{
    platform::ChunkKind,
    progress::{ProgressFrame, ProgressPhase},
    word::Word,
};
use std::{
    ops::ControlFlow,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, warn};
use whisper_rs::{FullParams, SamplingStrategy, WhisperState};

const SAMPLES_PER_CENTISECOND: usize = 160;
/// Whisper's encoder always sees 30 s, so a shorter window costs about as much to decode
const MAX_WINDOW_SECONDS: f32 = 30.0;
const COMMIT_MARGIN_SECONDS: f32 = 1.0;
const MIN_WINDOW_SECONDS: f32 = 0.1;
const MIN_PAUSED_WINDOW_SECONDS: f32 = 1.0;

pub struct StreamWhisperTranscriber {
    state: WhisperState,
    tokens: SpecialTokens,
    accel: AccelReport,
    config: Arc<TranscriptionConfig>,
}

/// Window lengths of one stream, in samples.
struct Windows {
    step: usize,
    max: usize,
    min: usize,
    min_paused: usize,
    margin: usize,
}

impl Windows {
    fn new(config: &TranscriptionConfig) -> Self {
        let samples = |seconds: f32| (config.sample_rate as f32 * seconds) as usize;
        Self {
            step: samples(config.chunk_size),
            max: samples(MAX_WINDOW_SECONDS),
            min: samples(MIN_WINDOW_SECONDS),
            min_paused: samples(MIN_PAUSED_WINDOW_SECONDS),
            margin: samples(COMMIT_MARGIN_SECONDS),
        }
    }
}

struct Decoded {
    span: Span,
    text: String,
    /// Timed from the window start
    words: Vec<Word>,
}

type Events = mpsc::UnboundedSender<crate::Result<StreamEvent>>;

/// One streaming transcription: the audio not yet committed, and what the stream has settled.
struct Session {
    config: Arc<TranscriptionConfig>,
    out: Events,
    beams: Arc<BeamCollector>,
    windows: Windows,
    state: WhisperState,
    tokens: SpecialTokens,
    speech: Option<SpeechDetector>,
    threads: usize,
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
    ended: bool,
    language: StreamLanguage,
    chunk_index: usize,
    segment_count: usize,
    word_count: usize,
    finished: bool,
    started: Instant,
}

impl WhisperTranscriber for StreamWhisperTranscriber {
    type TranscriberResult = StreamingTranscription;
    type InputData = AudioStream;

    async fn from_config(config: crate::TranscriptionConfig) -> crate::Result<Self>
    where
        Self: Sized,
    {
        let config = Arc::new(config);
        let model_manager = ModelManager::new()?;
        let loaded = load_model(&config, &model_manager).await?;

        Ok(Self {
            state: loaded.state,
            tokens: loaded.tokens,
            accel: loaded.accel,
            config,
        })
    }

    async fn transcribe(self, input: AudioStream) -> crate::Result<StreamingTranscription> {
        let (tx, rx) = mpsc::unbounded_channel();
        let Self {
            state,
            tokens,
            accel,
            config,
        } = self;

        tokio::spawn(async move {
            if let Err(e) =
                Self::process_audio_stream(state, tokens, config, input, tx.clone()).await
                && tx.send(Err(e)).is_err()
            {
                debug!("Receiver dropped, error not delivered");
            }
        });

        Ok(StreamingTranscription {
            stream: Box::pin(UnboundedReceiverStream::new(rx)),
            accel,
        })
    }
}

impl Session {
    fn new(
        state: WhisperState,
        tokens: SpecialTokens,
        config: Arc<TranscriptionConfig>,
        out: Events,
    ) -> crate::Result<Self> {
        let threads = match config.num_threads {
            Some(threads) => threads,
            None => std::thread::available_parallelism()?.get().min(4),
        };
        let language = match config.language.as_deref() {
            None | Some("auto") => StreamLanguage::Detecting,
            Some(code) => StreamLanguage::Chosen(
                whisper_rs::get_lang_id(code)
                    .and_then(whisper_rs::get_lang_str)
                    .ok_or_else(|| TranscriptionError::UnknownLanguage {
                        code: code.to_owned(),
                    })?,
            ),
        };
        let speech = config
            .vad_model_path
            .as_deref()
            .map(SpeechDetector::load)
            .transpose()?;
        Ok(Self {
            windows: Windows::new(&config),
            config,
            beams: Arc::new(BeamCollector::new(tokens.eot, out.clone())),
            out,
            state,
            tokens,
            speech,
            threads,
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
        })
    }

    fn push(&mut self, samples: &[f32]) {
        if !samples.is_empty() {
            self.first_audio.get_or_insert_with(Instant::now);
        }
        self.received += samples.len();
        self.pending.extend_from_slice(samples);
    }

    /// The window to decode now, if any.
    ///
    /// Without a voice activity model the stream waits for a full window: Whisper's segment
    /// timestamps in a short window are too loose to commit on.
    fn next_window(&self, complete: bool) -> Option<WindowKind> {
        let len = self.pending.len();
        if len >= self.windows.max {
            Some(WindowKind::Full)
        } else if complete && !self.finished {
            Some(WindowKind::Last)
        } else if self.speech.is_some() && len - self.seen >= self.windows.step {
            Some(WindowKind::Growing)
        } else {
            None
        }
    }

    fn sample_rate(&self) -> f64 {
        f64::from(self.config.sample_rate)
    }

    /// Where the stream stands, while it decodes the samples `decoding` of the source.
    fn progress(&self, decoding: Option<(usize, usize)>) -> StreamEvent {
        let sr = self.sample_rate();
        StreamEvent::Progress(StreamProgress {
            frame: ProgressFrame {
                transcribed_to: self.origin.min(self.received) as f64 / sr,
                decoding: decoding.map(|(start, end)| (start as f64 / sr, end as f64 / sr)),
                phase: match self.language {
                    StreamLanguage::Detecting => ProgressPhase::DetectingLanguage,
                    StreamLanguage::Chosen(_) | StreamLanguage::Locked(_) => {
                        ProgressPhase::Transcribing
                    }
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

    fn detect_language(&mut self, window: &[f32]) -> crate::Result<(LanguageCode, f32)> {
        self.state
            .pcm_to_mel(window, self.threads)
            .map_err(|source| TranscriptionError::Failed { source })?;
        let (id, probabilities) = self
            .state
            .lang_detect(0, self.threads)
            .map_err(|source| TranscriptionError::Failed { source })?;
        let code =
            whisper_rs::get_lang_str(id).ok_or(TranscriptionError::UnknownLanguageId { id })?;
        let probability = usize::try_from(id)
            .ok()
            .and_then(|i| probabilities.get(i).copied())
            .ok_or(TranscriptionError::UnknownLanguageId { id })?;
        Ok((code, probability))
    }

    fn decode(&mut self, window: &[f32], language: LanguageCode) -> crate::Result<Vec<Decoded>> {
        let beam_size = self.config.beam_size.unwrap_or(5) as i32;
        let mut params = if beam_size > 1 {
            FullParams::new(SamplingStrategy::BeamSearch {
                beam_size,
                patience: -1.0,
            })
        } else {
            FullParams::new(SamplingStrategy::Greedy { best_of: 5 })
        };
        params.set_language(Some(language));
        params.set_translate(self.config.translate);
        params.set_temperature(self.config.temperature);
        params.set_n_threads(self.threads as i32);
        params.set_print_timestamps(false);
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_no_context(true);
        params.set_token_timestamps(self.config.output_format.word_timestamps);

        let collector = self.config.beam_telemetry.is_on().then(|| {
            let sr = self.sample_rate();
            let start = self.origin as f64 / sr;
            self.beams
                .start_window((start, start + window.len() as f64 / sr));
            Arc::clone(&self.beams)
        });
        if let Some(collector) = &collector {
            // SAFETY: `on_logits` reads its user data only as a `&BeamCollector`, and `collector`
            // keeps that collector alive until `full()` returns; whisper.cpp calls the callback
            // only during that call.
            unsafe {
                params.set_filter_logits_callback(Some(beams::on_logits));
                params
                    .set_filter_logits_callback_user_data(Arc::as_ptr(collector).cast_mut().cast());
            }
        }

        let started = Instant::now();
        self.state
            .full(params, window)
            .map_err(|source| TranscriptionError::Failed { source })?;
        self.last_decode = Some(started.elapsed());
        drop(collector);

        Ok(segments::read(&self.state, self.tokens.eot)?
            .into_iter()
            .map(|segment| Decoded {
                span: Span {
                    start: segment.start_cs.max(0) as usize * SAMPLES_PER_CENTISECOND,
                    end: segment.end_cs.max(0) as usize * SAMPLES_PER_CENTISECOND,
                },
                text: segment.text,
                words: segment.words,
            })
            .collect())
    }

    /// Drops `n` samples from the front of the pending audio.
    fn consume(&mut self, n: usize) {
        self.pending.drain(..n);
        self.origin += n;
        self.seen = self.seen.saturating_sub(n);
        self.tentative_mark = self.tentative_mark.saturating_sub(n);
    }

    /// Whether the stream, waiting for its source, should decode its pending audio tentatively.
    fn tentative_due(&self) -> bool {
        let sr = self.sample_rate();
        let len = self.pending.len();
        let ready_at = if self.speech.is_some() {
            self.windows.max.min(self.seen + self.windows.step)
        } else {
            self.windows.max
        };
        let arrival_rate = self.first_audio.and_then(|first| {
            let wall = first.elapsed().as_secs_f64();
            (wall > 0.0).then(|| self.received as f64 / sr / wall)
        });
        tentative::due(
            self.config.tentative,
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
        parts: &[&Decoded],
        len: usize,
        language: Option<LanguageCode>,
    ) -> StreamingChunk {
        let sr = self.sample_rate();
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
            language: language.map(str::to_owned),
            kind: ChunkKind::Tentative,
            words,
        }
    }

    /// Decodes all the pending audio only to show it; the stream's position, counters and
    /// language lock stay as they were.
    fn tentative(&mut self) -> crate::Result<StreamEvent> {
        let window = self.pending.clone();
        let language = match self.language {
            StreamLanguage::Chosen(code) | StreamLanguage::Locked(code) => code,
            StreamLanguage::Detecting => self.detect_language(&window)?.0,
        };
        let decoded = self.decode(&window, language)?;
        self.tentative_mark = window.len();
        let parts: Vec<&Decoded> = decoded.iter().filter(|d| !d.text.is_empty()).collect();
        let reported = match self.language {
            StreamLanguage::Locked(code) => Some(code),
            StreamLanguage::Chosen(_) | StreamLanguage::Detecting => None,
        };
        Ok(StreamEvent::Chunk(self.tentative_chunk(
            &parts,
            window.len(),
            reported,
        )))
    }

    fn process(&mut self, kind: WindowKind) -> crate::Result<Vec<StreamEvent>> {
        let mut kind = kind;
        let mut len = match kind {
            WindowKind::Full => self.windows.max,
            WindowKind::Growing | WindowKind::Paused | WindowKind::Last => self.pending.len(),
        };
        let mut window: Vec<f32> = self.pending.iter().take(len).copied().collect();
        let mut spoke = len >= self.windows.min;

        if let Some(detector) = self.speech.as_mut() {
            let speech = if spoke {
                detector.speech(&window)?
            } else {
                Vec::new()
            };
            self.seen = self.seen.max(len);
            spoke = !speech.is_empty() && len >= self.windows.min;
            if spoke && kind != WindowKind::Last {
                match last_pause(&speech, len, self.windows.min_paused, self.windows.margin) {
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
            self.send(self.progress(Some((self.origin, self.origin + len))));
            let (language, probability) = match self.language {
                StreamLanguage::Chosen(code) | StreamLanguage::Locked(code) => (code, 1.0),
                StreamLanguage::Detecting => self.detect_language(&window)?,
            };
            let decoded = self.decode(&window, language)?;
            let spoken = decoded.iter().any(|d| !d.text.is_empty());
            if let Some(locked) = self.language.observe(language, probability, spoken) {
                debug!("Streaming language locked to {locked} (p = {probability:.2})");
            }
            (decoded, Some(language))
        } else {
            (Vec::new(), None)
        };

        let spans: Vec<Span> = decoded.iter().map(|d| d.span).collect();
        let decision = commit(&spans, len, self.windows.margin, kind);
        debug!(
            "Window at {:.2}s of {:.2}s ({kind:?}): {} of {} segments committed, next at {:.2}s",
            self.origin as f64 / self.sample_rate(),
            len as f64 / self.sample_rate(),
            decision.segments,
            decoded.len(),
            (self.origin + decision.advance) as f64 / self.sample_rate()
        );

        let tail = decoded.split_off(decision.segments);

        let reported = match self.language {
            StreamLanguage::Locked(code) => Some(code),
            StreamLanguage::Detecting if kind == WindowKind::Last => language,
            StreamLanguage::Chosen(_) | StreamLanguage::Detecting => None,
        };
        let sr = self.sample_rate();
        let origin_cs = u32::try_from(self.origin / SAMPLES_PER_CENTISECOND).unwrap_or(u32::MAX);
        let mut chunks = Vec::new();
        for d in decoded.iter().filter(|d| !d.text.is_empty()) {
            let end = d.span.end.min(len);
            chunks.push(StreamingChunk {
                text: d.text.clone(),
                start: (self.origin + d.span.start.min(end)) as f64 / sr,
                end: (self.origin + end) as f64 / sr,
                is_final: false,
                chunk_index: self.chunk_index,
                final_stats: None,
                language: reported.map(str::to_owned),
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

        if kind == WindowKind::Full && self.config.tentative != TentativeText::Off {
            let parts: Vec<&Decoded> = tail.iter().filter(|d| !d.text.is_empty()).collect();
            chunks.push(self.tentative_chunk(&parts, len, reported));
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
                    language: reported.map(str::to_owned),
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

impl StreamWhisperTranscriber {
    async fn process_audio_stream(
        state: WhisperState,
        tokens: SpecialTokens,
        config: Arc<TranscriptionConfig>,
        mut input: AudioStream,
        tx: Events,
    ) -> crate::Result<()> {
        let mut session = Session::new(state, tokens, config, tx.clone())?;
        let mut complete = false;

        let receive = |item: Option<crate::Result<crate::AudioChunk>>,
                       session: &mut Session,
                       complete: &mut bool| match item {
            Some(Ok(chunk)) => {
                *complete |= chunk.is_final;
                session.ended = *complete;
                session.push(&chunk.samples);
                ControlFlow::Continue(())
            }
            Some(Err(e)) => {
                warn!("Audio stream error: {}", e);
                if tx.send(Err(e)).is_err() {
                    debug!("Receiver dropped, error not delivered");
                }
                ControlFlow::Break(())
            }
            None => {
                *complete = true;
                session.ended = true;
                ControlFlow::Continue(())
            }
        };

        while !session.finished {
            if !complete {
                if session.next_window(false).is_none() {
                    let item = match input.next().now_or_never() {
                        Some(item) => item,
                        None if session.tentative_due() => {
                            let event = match session.tentative() {
                                Ok(event) => event,
                                Err(e) => {
                                    warn!("Failed to decode audio tentatively: {}", e);
                                    if tx.send(Err(e)).is_err() {
                                        debug!("Receiver dropped, error not delivered");
                                    }
                                    return Ok(());
                                }
                            };
                            if tx.send(Ok(event)).is_err() {
                                debug!("Receiver dropped, stopping stream processing");
                                return Ok(());
                            }
                            continue;
                        }
                        None => input.next().await,
                    };
                    if receive(item, &mut session, &mut complete).is_break() {
                        return Ok(());
                    }
                }
                while !complete && let Some(item) = input.next().now_or_never() {
                    if receive(item, &mut session, &mut complete).is_break() {
                        return Ok(());
                    }
                }
            }

            while let Some(kind) = session.next_window(complete) {
                let events = match session.process(kind) {
                    Ok(events) => events,
                    Err(e) => {
                        warn!("Failed to process audio window: {}", e);
                        if tx.send(Err(e)).is_err() {
                            debug!("Receiver dropped, error not delivered");
                        }
                        return Ok(());
                    }
                };
                for event in events {
                    if tx.send(Ok(event)).is_err() {
                        debug!("Receiver dropped, stopping stream processing");
                        return Ok(());
                    }
                }
                if !complete {
                    break;
                }
            }
        }

        Ok(())
    }
}

pub struct StreamingTranscription {
    stream: Pin<Box<dyn Stream<Item = Result<StreamEvent, crate::WhisperError>> + Send>>,
    accel: AccelReport,
}

impl StreamingTranscription {
    /// Acceleration backend the transcription runs on
    #[must_use]
    pub const fn accel(&self) -> &AccelReport {
        &self.accel
    }
}

impl Stream for StreamingTranscription {
    type Item = Result<StreamEvent, crate::WhisperError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}

impl TranscriptionResult for StreamingTranscription {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioChunk, AudioProcessor};

    const HALF_SECOND: usize = 8_000;

    /// Feeds `samples` in half seconds at the pace of speech, from a thread of its own as a
    /// recording device would.
    fn paced(samples: Vec<f32>) -> AudioStream {
        let (tx, rx) = mpsc::unbounded_channel();
        std::thread::spawn(move || {
            let parts = samples.chunks(HALF_SECOND).count();
            for (i, part) in samples.chunks(HALF_SECOND).enumerate() {
                std::thread::sleep(Duration::from_millis(500));
                let chunk = AudioChunk::new(part.to_vec(), i, i as f32 * 0.5, i + 1 == parts);
                if tx.send(Ok(chunk)).is_err() {
                    return;
                }
            }
        });
        AudioStream::new(rx)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "needs ggml-base.bin and feeds 11 s of audio at the pace of speech"]
    async fn tentative_text_shows_early_and_leaves_the_final_text_alone()
    -> Result<(), Box<dyn std::error::Error>> {
        let model = ModelManager::new()?.models_dir().join("ggml-base.bin");
        let audio = AudioProcessor::load_audio(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples/jfk.wav"),
        )
        .await?;
        let mut final_texts = Vec::new();
        for tentative in [
            TentativeText::Off,
            TentativeText::WhileWaiting { every_seconds: 2.0 },
        ] {
            let config = TranscriptionConfig::new()
                .with_gpu(false)
                .with_threads(4)
                .with_language("en")
                .with_model_path(&model)
                .with_tentative(tentative);
            let transcriber = StreamWhisperTranscriber::from_config(config).await?;
            let started = Instant::now();
            let mut stream = transcriber.transcribe(paced(audio.samples.clone())).await?;
            let mut first_text = None;
            let mut tentative_chunks = 0;
            let mut finals = Vec::new();
            while let Some(event) = stream.next().await {
                let StreamEvent::Chunk(chunk) = event? else {
                    continue;
                };
                if !chunk.text.is_empty() {
                    first_text.get_or_insert_with(|| started.elapsed());
                }
                match chunk.kind {
                    ChunkKind::Tentative => {
                        tentative_chunks += 1;
                        println!(
                            "{:>5.2}s tentative #{} [{:.1}-{:.1}] {}",
                            started.elapsed().as_secs_f64(),
                            chunk.chunk_index,
                            chunk.start,
                            chunk.end,
                            chunk.text
                        );
                    }
                    ChunkKind::Committed => finals.push(chunk.text),
                }
            }
            println!(
                "{tentative:?}: first text after {first_text:?}, {tentative_chunks} tentative \
                 chunks, last text after {:?}",
                started.elapsed()
            );
            assert_eq!(
                tentative_chunks > 0,
                tentative != TentativeText::Off,
                "{tentative:?}"
            );
            final_texts.push(finals);
        }
        assert_eq!(final_texts[0], final_texts[1]);
        Ok(())
    }
}
