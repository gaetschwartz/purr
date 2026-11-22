//! Starting a transcription: the audio, and the model that runs it.

use std::{path::PathBuf, sync::Arc};

use tokio::try_join;
use url::Url;

use crate::{
    AudioProcessor, AudioStream,
    config::{InputConfig, ModelRef, RunConfig},
    engine::Engine,
    error::Result,
};

use super::{events::Transcription, model::LoadedModel};

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

    pub(super) async fn open(self) -> Result<AudioStream> {
        match self {
            Self::File(path) => AudioProcessor::stream_file(path).await,
            Self::Url { url, config } => AudioProcessor::stream_url_with_config(url, &config).await,
            Self::Audio(stream) => Ok(stream),
        }
    }
}

/// Transcribes `input` with the model of `model`, run by the engine of `engines` it names.
///
/// The model loads while the audio starts to arrive, for this transcription alone; use a
/// [`LoadedModel`] to transcribe several inputs with one load. The transcription runs on its own
/// task; dropping the returned [`Transcription`] cancels it.
pub async fn transcribe(
    input: Input,
    run: RunConfig,
    model: &ModelRef,
    engines: &[Arc<dyn Engine>],
) -> Result<Transcription> {
    let (mut loaded, audio) = try_join!(LoadedModel::load(model, &run, engines), input.open())?;
    loaded.start(audio, &run)
}
