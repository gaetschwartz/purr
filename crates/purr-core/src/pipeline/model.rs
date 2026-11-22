//! A model loaded on its device, which serves one transcription after another.

use std::{path::PathBuf, sync::Arc};

use tokio::{
    sync::{mpsc, oneshot, watch},
    task, try_join,
};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, info};

use crate::{
    AudioStream, SourceLength,
    accel::report::AccelReport,
    config::{ModelRef, RunConfig},
    engine::{
        Engine,
        cancel::CancelToken,
        decoder::WindowDecoder,
        open::{Opened, open_on_best_device},
        speech::SpeechDetector,
    },
    error::{Error, Result, TranscriptionError},
};

use super::{
    driver::{EndSlot, Ending, Request, drive},
    events::Transcription,
    policy::{SAMPLE_RATE, WindowPolicy, samples},
    run::Input,
    session::{Decoding, Session},
};

/// A model on its device that outlives any one transcription.
///
/// A transcription takes the decoder and hands it back when it ends, however it ends, so the
/// next one reuses it. [`LoadedModel::transcribe`] waits for the previous transcription to end.
pub struct LoadedModel {
    engine: Arc<dyn Engine>,
    accel: AccelReport,
    decoder: Option<Box<dyn WindowDecoder>>,
    returning: Option<oneshot::Receiver<Box<dyn WindowDecoder>>>,
}

impl LoadedModel {
    /// Loads `model` on the best device `run` allows, with the engine of `engines` it names.
    pub async fn load(
        model: &ModelRef,
        run: &RunConfig,
        engines: &[Arc<dyn Engine>],
    ) -> Result<Self> {
        let engine = engines
            .iter()
            .find(|engine| engine.id() == model.engine)
            .cloned()
            .ok_or(TranscriptionError::EngineMissing {
                engine: model.engine,
            })?;
        info!("Loading model...");
        debug!("Model: {}", model.path.display());
        let path: PathBuf = model.path.clone();
        let devices = run.devices.clone();
        let threads = run.thread_count();
        let loader = Arc::clone(&engine);
        let Opened { decoder, accel } = task::spawn_blocking(move || {
            open_on_best_device(loader.as_ref(), &path, &devices, threads)
        })
        .await
        .map_err(|source| Error::TaskJoin {
            task: "model loading",
            source,
        })??;
        Ok(Self {
            engine,
            accel,
            decoder: Some(decoder),
            returning: None,
        })
    }

    /// Acceleration backend the model runs on
    #[must_use]
    pub const fn accel(&self) -> &AccelReport {
        &self.accel
    }

    /// Transcribes `input` once the previous transcription has ended.
    ///
    /// Each input has its own language, windows and speech detector; only the model is shared.
    pub async fn transcribe(&mut self, input: Input, run: &RunConfig) -> Result<Transcription> {
        let (audio, ()) = try_join!(input.open(), self.settle())?;
        self.start(audio, run)
    }

    async fn settle(&mut self) -> Result<()> {
        if let Some(returning) = self.returning.as_mut() {
            let outcome = returning.await;
            self.returning = None;
            self.decoder = outcome.ok();
        }
        if self.decoder.is_some() {
            Ok(())
        } else {
            Err(TranscriptionError::DecoderLost.into())
        }
    }

    /// Starts a transcription of `audio`; the previous one must have ended.
    pub(super) fn start(&mut self, audio: AudioStream, run: &RunConfig) -> Result<Transcription> {
        let decoder = self.decoder.take().ok_or(TranscriptionError::DecoderLost)?;
        let (policy, speech) = match plan(decoder.as_ref(), self.engine.as_ref(), run) {
            Ok(plan) => plan,
            Err(error) => {
                self.decoder = Some(decoder);
                return Err(error);
            }
        };
        debug!("Audio stream created, starting transcription...");

        let length = audio.length();
        let listen = match length {
            SourceLength::Unknown => run.listen_for.map(listen_samples),
            SourceLength::Exact(_) | SourceLength::Estimated(_) => None,
        };
        let cancel = CancelToken::new();
        let (tx, rx) = mpsc::unbounded_channel();
        let (request_tx, request_rx) = watch::channel(Request::Run);
        let (back_tx, back_rx) = oneshot::channel();
        let end = EndSlot::default();
        let mut session = Session::new(
            run,
            Decoding {
                decoder,
                speech,
                policy,
                cancel: cancel.clone(),
            },
            tx.clone(),
        );
        let ending = Ending {
            request: request_rx,
            listen,
            reason: end.clone(),
        };
        tokio::spawn(async move {
            drive(&mut session, audio, &tx, ending).await;
            if back_tx.send(session.into_decoder()).is_err() {
                debug!("Model handle dropped, decoder released");
            }
        });
        self.returning = Some(back_rx);

        Ok(Transcription::new(
            Box::pin(UnboundedReceiverStream::new(rx)),
            self.accel.clone(),
            cancel,
            request_tx,
            length,
            end,
        ))
    }
}

fn listen_samples(listen_for: std::time::Duration) -> usize {
    let samples = listen_for.as_micros() * SAMPLE_RATE as u128 / 1_000_000;
    usize::try_from(samples).unwrap_or(usize::MAX)
}

/// The windows an input is cut into, and the speech detector that places their ends.
fn plan(
    decoder: &dyn WindowDecoder,
    engine: &dyn Engine,
    run: &RunConfig,
) -> Result<(WindowPolicy, Option<Box<dyn SpeechDetector>>)> {
    if let Some(tag) = &run.language
        && !decoder.supports(tag)
    {
        return Err(TranscriptionError::UnknownLanguage {
            code: tag.to_string(),
        }
        .into());
    }

    let policy = WindowPolicy::choose(
        run.windowing,
        decoder.limits(),
        run.vad_model.is_some(),
        samples(run.chunk_size),
    );
    let speech = match (&run.vad_model, policy.uses_pauses()) {
        (Some(path), true) => Some(engine.open_speech_detector(path).map_err(|source| {
            TranscriptionError::VadModel {
                path: path.clone(),
                source,
            }
        })?),
        (Some(_) | None, false) | (None, true) => None,
    };
    Ok((policy, speech))
}
