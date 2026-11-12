use crate::{
    AudioStream, ModelManager, TranscriptionConfig,
    accel::report::AccelReport,
    error::TranscriptionError,
    whisper::{
        StreamingChunk, TranscriptionResult, TranscriptionStats, WhisperTranscriber,
        commit::{Span, WindowKind, commit},
        language::{LanguageCode, StreamLanguage},
        loader::load_model,
    },
};
use futures::{FutureExt, Stream, StreamExt};
use std::{
    ops::ControlFlow,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
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

pub struct StreamWhisperTranscriber {
    state: WhisperState,
    accel: AccelReport,
    config: Arc<TranscriptionConfig>,
}

/// Window lengths of one stream, in samples.
struct Windows {
    max: usize,
    min: usize,
    margin: usize,
}

impl Windows {
    fn new(config: &TranscriptionConfig) -> Self {
        let samples = |seconds: f32| (config.sample_rate as f32 * seconds) as usize;
        Self {
            max: samples(MAX_WINDOW_SECONDS),
            min: samples(MIN_WINDOW_SECONDS),
            margin: samples(COMMIT_MARGIN_SECONDS),
        }
    }
}

struct Decoded {
    span: Span,
    text: String,
}

/// One streaming transcription: the audio not yet committed, and what the stream has settled.
struct Session {
    config: Arc<TranscriptionConfig>,
    windows: Windows,
    state: WhisperState,
    threads: usize,
    /// Audio from `origin` on that no committed segment covers yet
    pending: Vec<f32>,
    origin: usize,
    received: usize,
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
            accel: loaded.accel,
            config,
        })
    }

    async fn transcribe(self, input: AudioStream) -> crate::Result<StreamingTranscription> {
        let (tx, rx) = mpsc::unbounded_channel();
        let Self {
            state,
            accel,
            config,
        } = self;

        tokio::spawn(async move {
            if let Err(e) = Self::process_audio_stream(state, config, input, tx.clone()).await
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
    fn new(state: WhisperState, config: Arc<TranscriptionConfig>) -> crate::Result<Self> {
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
        Ok(Self {
            windows: Windows::new(&config),
            config,
            state,
            threads,
            pending: Vec::new(),
            origin: 0,
            received: 0,
            language,
            chunk_index: 0,
            segment_count: 0,
            word_count: 0,
            finished: false,
            started: Instant::now(),
        })
    }

    fn push(&mut self, samples: &[f32]) {
        self.received += samples.len();
        self.pending.extend_from_slice(samples);
    }

    /// The window to decode now, if any.
    ///
    /// The stream waits for a full window: Whisper's segment timestamps in a short window are
    /// too loose to commit on.
    fn next_window(&self, complete: bool) -> Option<WindowKind> {
        if self.pending.len() >= self.windows.max {
            Some(WindowKind::Full)
        } else if complete && !self.finished {
            Some(WindowKind::Last)
        } else {
            None
        }
    }

    fn sample_rate(&self) -> f64 {
        f64::from(self.config.sample_rate)
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

        self.state
            .full(params, window)
            .map_err(|source| TranscriptionError::Failed { source })?;

        let mut decoded = Vec::new();
        for segment in self.state.as_iter() {
            let text = segment
                .to_str_lossy()
                .map_err(|source| TranscriptionError::Failed { source })?
                .trim()
                .to_owned();
            decoded.push(Decoded {
                span: Span {
                    start: segment.start_timestamp().max(0) as usize * SAMPLES_PER_CENTISECOND,
                    end: segment.end_timestamp().max(0) as usize * SAMPLES_PER_CENTISECOND,
                },
                text,
            });
        }
        Ok(decoded)
    }

    /// Drops `n` samples from the front of the pending audio.
    fn consume(&mut self, n: usize) {
        self.pending.drain(..n);
        self.origin += n;
    }

    fn process(&mut self, kind: WindowKind) -> crate::Result<Vec<StreamingChunk>> {
        let len = match kind {
            WindowKind::Full => self.windows.max,
            WindowKind::Last => self.pending.len(),
        };
        let window: Vec<f32> = self.pending.iter().take(len).copied().collect();
        let spoke = len >= self.windows.min;

        let (mut decoded, language) = if spoke {
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

        decoded.truncate(decision.segments);

        let reported = match self.language {
            StreamLanguage::Locked(code) => Some(code),
            StreamLanguage::Detecting if kind == WindowKind::Last => language,
            StreamLanguage::Chosen(_) | StreamLanguage::Detecting => None,
        };
        let sr = self.sample_rate();
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
            });
            self.chunk_index += 1;
            self.segment_count += 1;
            self.word_count += d.text.split_whitespace().count();
        }

        self.consume(decision.advance);

        if kind == WindowKind::Last {
            self.finished = true;
            let end = self.received as f64 / sr;
            if chunks.is_empty() {
                chunks.push(StreamingChunk {
                    text: String::new(),
                    start: end,
                    end,
                    is_final: false,
                    chunk_index: self.chunk_index,
                    final_stats: None,
                    language: reported.map(str::to_owned),
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
        Ok(chunks)
    }
}

impl StreamWhisperTranscriber {
    async fn process_audio_stream(
        state: WhisperState,
        config: Arc<TranscriptionConfig>,
        mut input: AudioStream,
        tx: mpsc::UnboundedSender<crate::Result<StreamingChunk>>,
    ) -> crate::Result<()> {
        let mut session = Session::new(state, config)?;
        let mut complete = false;

        let receive = |item: Option<crate::Result<crate::AudioChunk>>,
                       session: &mut Session,
                       complete: &mut bool| match item {
            Some(Ok(chunk)) => {
                *complete |= chunk.is_final;
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
                ControlFlow::Continue(())
            }
        };

        while !session.finished {
            if !complete {
                if session.next_window(false).is_none() {
                    let item = input.next().await;
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
                let chunks = match session.process(kind) {
                    Ok(chunks) => chunks,
                    Err(e) => {
                        warn!("Failed to process audio window: {}", e);
                        if tx.send(Err(e)).is_err() {
                            debug!("Receiver dropped, error not delivered");
                        }
                        return Ok(());
                    }
                };
                for chunk in chunks {
                    if tx.send(Ok(chunk)).is_err() {
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
    stream: Pin<Box<dyn Stream<Item = Result<StreamingChunk, crate::WhisperError>> + Send>>,
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
    type Item = Result<StreamingChunk, crate::WhisperError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}

impl TranscriptionResult for StreamingTranscription {}
