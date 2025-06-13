use crate::{
    whisper::{load_model, StreamingChunk, TranscriptionResult, WhisperTranscriber},
    AudioStream, ModelManager, TranscriptionConfig,
};
use futures::Stream;
use std::{future::Future, pin::Pin};
use whisper_rs::WhisperContext;

pub struct StreamWhisperTranscriber {
    context: WhisperContext,
    config: TranscriptionConfig,
}

impl WhisperTranscriber for StreamWhisperTranscriber {
    type TranscriberResult = StreamingTranscriptionResult;
    type InputData = AudioStream;

    async fn from_config(config: crate::TranscriptionConfig) -> crate::Result<Self>
    where
        Self: Sized,
    {
        // Install logging hooks to redirect whisper.cpp output to Rust logging
        whisper_rs::install_logging_hooks();

        let config_clone = config.clone();

        let model_manager = ModelManager::new()?;

        // Load the model (which may involve async model discovery)
        let context = load_model(&config_clone, &model_manager).await?;

        Ok(Self { context, config })
    }

    fn transcribe(
        &mut self,
        input: AudioStream,
    ) -> impl Future<Output = crate::Result<StreamingTranscriptionResult>> + '_ {
        todo!()
    }
}

pub struct StreamingTranscriptionResult {
    stream: Pin<Box<dyn Stream<Item = Result<StreamingChunk, crate::WhisperError>> + Send>>,
}

impl TranscriptionResult for StreamingTranscriptionResult {}
