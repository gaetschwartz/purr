//! Whisper transcription functionality

use crate::{
    ModelManager,
    accel::report::AccelReport,
    audio::AudioData,
    config::TranscriptionConfig,
    error::{Result, TranscriptionError, WhisperError},
    whisper::{
        SyncTranscriptionResult, TranscriptionSegment, TranscriptionStats, WhisperTranscriber,
        loader::{SpecialTokens, load_model},
        segments,
    },
};
use whisper_rs::{FullParams, SamplingStrategy, WhisperState};

/// Whisper transcriber
pub struct SyncWhisperTranscriber {
    state: WhisperState,
    tokens: SpecialTokens,
    accel: AccelReport,
    config: TranscriptionConfig,
}

impl WhisperTranscriber for SyncWhisperTranscriber {
    type TranscriberResult = SyncTranscriptionResult;
    type InputData = AudioData;
    /// Create a new transcriber with the given configuration
    async fn from_config(config: TranscriptionConfig) -> Result<Self> {
        let config_clone = config.clone();

        let model_manager = ModelManager::new()?;

        // Load the model (which may involve async model discovery)
        let loaded = load_model(&config_clone, &model_manager).await?;

        Ok(Self {
            state: loaded.state,
            tokens: loaded.tokens,
            accel: loaded.accel,
            config,
        })
    }

    /// Transcribe audio data
    async fn transcribe(mut self, audio_data: AudioData) -> Result<SyncTranscriptionResult> {
        let config = self.config.clone();

        // Run transcription synchronously since we can't clone the context
        self.transcribe_sync_internal(audio_data, config)
    }
}

impl SyncWhisperTranscriber {
    /// Synchronous transcription implementation
    fn transcribe_sync_internal(
        &mut self,
        audio_data: AudioData,
        config: TranscriptionConfig,
    ) -> Result<SyncTranscriptionResult> {
        let start_time = std::time::Instant::now();

        // Setup transcription parameters with proper sampling strategy
        // Match whisper.cpp's "4 threads, 1 processors, 5 beams + best of 5" quality
        let beam_size = self.config.beam_size.unwrap_or(5) as i32;
        let mut params = if beam_size > 1 {
            FullParams::new(SamplingStrategy::BeamSearch {
                beam_size,
                patience: -1.0,
            })
        } else {
            FullParams::new(SamplingStrategy::Greedy { best_of: 5 })
        };

        // Configure parameters
        params.set_language(self.config.language.as_deref());
        params.set_translate(self.config.translate);

        let threads = config
            .num_threads
            .unwrap_or_else(crate::config::auto_threads);
        params.set_n_threads(threads as i32);

        params.set_temperature(config.temperature);

        params.set_print_timestamps(false); // Disable whisper.cpp's internal timestamp printing
        params.set_print_progress(false); // Disable progress output
        params.set_print_special(false); // Disable special token printing
        params.set_print_realtime(false); // Disable real-time printing
        params.set_token_timestamps(config.output_format.word_timestamps);

        let state = &mut self.state;

        // Run transcription using state.full()
        state
            .full(params, &audio_data.samples)
            .map_err(|e| WhisperError::from(TranscriptionError::Failed { source: e }))?;

        let processing_time = start_time.elapsed().as_secs_f64();

        let segments: Vec<TranscriptionSegment> = segments::read(state, self.tokens.eot)?
            .into_iter()
            .map(|segment| TranscriptionSegment {
                start: segment.start_cs as f64 / 100.0,
                end: segment.end_cs as f64 / 100.0,
                words: segment.words,
                text: segment.text,
            })
            .filter(|segment| !segment.text.is_empty())
            .collect();
        let full_text = segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // Calculate statistics
        let word_count = full_text.split_whitespace().count();
        let stats = {
            let audio_duration = audio_data.duration;
            let segment_count = segments.len();
            TranscriptionStats {
                processing_time,
                audio_duration,
                segment_count,
                word_count,
            }
        };

        Ok(SyncTranscriptionResult {
            text: full_text,
            segments,
            processing_time,
            audio_duration: audio_data.duration,
            stats,
            accel: self.accel.clone(),
        })
    }
}
