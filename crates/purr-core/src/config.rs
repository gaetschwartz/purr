//! What a transcription is asked to do: the model, how to run it, and how to read its input.

use std::{path::PathBuf, time::Duration};

use purr_common::{devices::DeviceOrder, language::tag::LanguageTag, model::engine::EngineId};
use serde::{Deserialize, Serialize};

/// The threads a transcription uses when none is asked for: one per physical core, because
/// ggml's threads wait on each other and a second hardware thread per core slows them down.
#[must_use]
pub fn auto_threads() -> usize {
    num_cpus::get_physical().max(1)
}

/// A model file and the engine that runs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRef {
    pub path: PathBuf,
    pub engine: EngineId,
}

impl ModelRef {
    pub fn new(path: impl Into<PathBuf>, engine: EngineId) -> Self {
        Self {
            path: path.into(),
            engine,
        }
    }
}

/// Whether the transcript arrives while the audio is decoded, or once all of it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Windowing {
    /// Window after window, each committed as soon as nothing after it can change it
    #[default]
    Stream,
    /// The whole source in one decode once it has all arrived, when the model takes audio of any
    /// length; otherwise as `Stream`
    Whole,
}

/// How a transcription runs.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// The spoken language; detected when `None`
    pub language: Option<LanguageTag>,

    /// The devices to try, first to last; the processor is the last resort
    pub devices: DeviceOrder,

    /// Number of threads to use
    pub threads: Option<usize>,

    /// Seconds of new audio a stream with a voice activity model gathers before it looks for a
    /// pause to decode up to
    pub chunk_size: f32,

    /// Silero voice activity model (ggml): streaming windows end at pauses in speech, and windows
    /// without speech are not decoded
    pub vad_model: Option<PathBuf>,

    /// Whether and when the stream shows text it has not committed yet
    pub tentative: TentativeText,

    /// Whether the words carry their times
    pub word_timestamps: bool,

    pub windowing: Windowing,

    /// How much audio to take from a source of unknown length before ending it as a normal end;
    /// a source of known length is always taken whole
    pub listen_for: Option<Duration>,
}

/// When a stream shows text it has not committed yet, as tentative chunks.
///
/// A tentative chunk carries the stream's whole uncommitted tail, with the index the next
/// committed chunk will take. A later tentative chunk replaces it, and so does any committed
/// chunk with the same or a higher index; the stream's last chunk is committed, so no
/// tentative text outlives a stream that ends normally.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TentativeText {
    /// Only committed text
    Off,
    /// Also the end of a full window that was decoded but not committed, which costs nothing
    Decoded,
    /// As `Decoded`, and also decode the pending audio while the stream waits for a slow (live)
    /// source, once at least `every_seconds` of new audio arrived and only when the next window
    /// cannot be ready before that decode ends; each such decode costs a whole encoder pass
    WhileWaiting { every_seconds: f32 },
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            language: None,
            devices: DeviceOrder::automatic(),
            threads: None,
            chunk_size: 4.0,
            vad_model: None,
            tentative: TentativeText::Off,
            word_timestamps: false,
            windowing: Windowing::Stream,
            listen_for: None,
        }
    }
}

impl RunConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_language(mut self, language: LanguageTag) -> Self {
        self.language = Some(language);
        self
    }

    #[must_use]
    pub fn with_opt_language(mut self, language: Option<LanguageTag>) -> Self {
        self.language = language;
        self
    }

    #[must_use]
    pub fn with_devices(mut self, devices: DeviceOrder) -> Self {
        self.devices = devices;
        self
    }

    /// `true` tries every device in the usual order, `false` the processor alone
    #[must_use]
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.devices = if use_gpu {
            DeviceOrder::automatic()
        } else {
            DeviceOrder::cpu_only()
        };
        self
    }

    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = Some(threads);
        self
    }

    /// The threads to run on: the ones asked for, or [`auto_threads`]
    #[must_use]
    pub fn thread_count(&self) -> usize {
        self.threads.unwrap_or_else(auto_threads)
    }

    #[must_use]
    pub fn with_chunk_size(mut self, chunk_size: f32) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    #[must_use]
    pub fn with_vad_model(mut self, path: Option<PathBuf>) -> Self {
        self.vad_model = path;
        self
    }

    #[must_use]
    pub fn with_tentative(mut self, tentative: TentativeText) -> Self {
        self.tentative = tentative;
        self
    }

    #[must_use]
    pub fn with_word_timestamps(mut self, word_timestamps: bool) -> Self {
        self.word_timestamps = word_timestamps;
        self
    }

    #[must_use]
    pub fn with_windowing(mut self, windowing: Windowing) -> Self {
        self.windowing = windowing;
        self
    }

    #[must_use]
    pub fn with_listen_for(mut self, listen_for: Option<Duration>) -> Self {
        self.listen_for = listen_for;
        self
    }
}

/// How audio is fetched from a URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputConfig {
    /// Seconds a URL transfer may go without data before it fails
    pub http_timeout: u64,
    /// HTTP connection timeout for URL requests, in seconds
    pub http_connect_timeout: u64,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            http_timeout: 30,
            http_connect_timeout: 10,
        }
    }
}

impl InputConfig {
    #[must_use]
    pub fn with_http_timeout(mut self, timeout: u64) -> Self {
        self.http_timeout = timeout;
        self
    }

    #[must_use]
    pub fn with_http_connect_timeout(mut self, timeout: u64) -> Self {
        self.http_connect_timeout = timeout;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_threads_never_exceed_the_logical_threads() {
        let logical = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        assert!((1..=logical).contains(&auto_threads()));
    }

    #[test]
    fn every_device_is_tried_by_default() {
        assert_eq!(RunConfig::new().devices, DeviceOrder::automatic());
    }

    #[test]
    fn disabling_gpu_leaves_the_processor_alone() {
        assert_eq!(
            RunConfig::new().with_gpu(false).devices,
            DeviceOrder::cpu_only()
        );
    }

    #[test]
    fn enabling_gpu_restores_the_usual_order() {
        let config = RunConfig::new()
            .with_devices(DeviceOrder::cpu_only())
            .with_gpu(true);
        assert_eq!(config.devices, DeviceOrder::automatic());
    }
}
