//! Starting a transcription: the model on its device, the audio, and the task between them.

use std::{path::PathBuf, sync::Arc};

use tokio::{sync::mpsc, task, try_join};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, info};
use url::Url;

use crate::{
    AudioProcessor, AudioStream,
    accel::report::AccelReport,
    config::{InputConfig, ModelRef, RunConfig},
    engine::{
        Engine,
        cancel::CancelToken,
        open::{Opened, open_on_best_device},
    },
    error::{Error, Result, TranscriptionError},
};

use super::{
    driver::drive,
    events::Transcription,
    policy::{WindowPolicy, samples},
    session::{Decoding, Session},
};

/// Where the audio of a transcription comes from.
#[derive(Debug)]
pub enum Input {
    File(PathBuf),
    Url { url: Url, config: InputConfig },
    Audio(AudioStream),
}

impl Input {
    /// Audio fetched from `url`.
    pub fn url(url: impl reqwest::IntoUrl, config: InputConfig) -> Result<Self> {
        Ok(Self::Url {
            url: url
                .into_url()
                .map_err(|source| crate::error::Error::InvalidUrl { source })?,
            config,
        })
    }

    async fn open(self) -> Result<AudioStream> {
        match self {
            Self::File(path) => AudioProcessor::stream_file(path).await,
            Self::Url { url, config } => AudioProcessor::stream_url_with_config(url, &config).await,
            Self::Audio(stream) => Ok(stream),
        }
    }
}

/// Transcribes `input` with the model of `model`, run by the engine of `engines` it names.
///
/// The model loads while the audio starts to arrive. The transcription runs on its own task;
/// dropping the returned [`Transcription`] cancels it.
pub async fn transcribe(
    input: Input,
    run: RunConfig,
    model: &ModelRef,
    engines: &[Arc<dyn Engine>],
) -> Result<Transcription> {
    let engine = engines
        .iter()
        .find(|engine| engine.id() == model.engine)
        .cloned()
        .ok_or(TranscriptionError::EngineMissing {
            engine: model.engine,
        })?;
    let ((decoding, accel), audio) =
        try_join!(prepare(engine, model.path.clone(), &run), input.open())?;
    debug!("Audio stream created, starting transcription...");

    let cancel = decoding.cancel.clone();
    let (tx, rx) = mpsc::unbounded_channel();
    let session = Session::new(&run, decoding, tx.clone());
    tokio::spawn(drive(session, audio, tx));

    Ok(Transcription::new(
        Box::pin(UnboundedReceiverStream::new(rx)),
        accel,
        cancel,
    ))
}

/// The model loaded on the best device, with what the run decodes with.
async fn prepare(
    engine: Arc<dyn Engine>,
    path: PathBuf,
    run: &RunConfig,
) -> Result<(Decoding, AccelReport)> {
    info!("Loading model...");
    debug!("Model: {}", path.display());
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

    Ok((
        Decoding {
            decoder,
            speech,
            policy,
            cancel: CancelToken::new(),
        },
        accel,
    ))
}
