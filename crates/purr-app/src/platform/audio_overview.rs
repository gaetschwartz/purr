//! Decodes a stored upload for its length and waveform, the way the platform reads it to transcribe.

use crate::transcribe::waveform::{BARS, peaks};
use bytes::Bytes;
use purr_common::platform::FileId;

/// The largest file the platform accepts, when it has a limit.
#[cfg(purr_web)]
pub const UPLOAD_LIMIT_BYTES: Option<u64> = Some(purr_wasm::audio::MAX_AUDIO_BYTES as u64);

/// The largest file the platform accepts, when it has a limit.
#[cfg(not(purr_web))]
pub const UPLOAD_LIMIT_BYTES: Option<u64> = None;

/// The length of a recording and its waveform bars.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    pub duration_seconds: f64,
    pub peaks: Vec<f32>,
}

/// Decodes the copy the platform stored through the same chunk stream the transcription reads.
#[cfg(all(feature = "desktop", not(purr_web)))]
pub async fn decode(file_id: &FileId, _bytes: Bytes) -> miette::Result<Decoded> {
    use futures::StreamExt as _;

    let mut stream =
        purr_core::AudioProcessor::stream_file(super::platform_impl::stored_upload(file_id))
            .await
            .map_err(|error| miette::miette!("{error}"))?;
    let mut samples = Vec::new();
    let mut sample_rate = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| miette::miette!("{error}"))?;
        sample_rate = chunk.sample_rate;
        samples.extend_from_slice(&chunk.samples);
        if chunk.is_final {
            break;
        }
    }
    if sample_rate == 0 {
        return Err(miette::miette!("the file holds no audio"));
    }
    Ok(Decoded {
        duration_seconds: samples.len() as f64 / f64::from(sample_rate),
        peaks: peaks(&samples, BARS),
    })
}

/// Decodes the upload with the browser's Web Audio API, as the transcription does.
#[cfg(purr_web)]
pub async fn decode(_file_id: &FileId, bytes: Bytes) -> miette::Result<Decoded> {
    let audio = purr_wasm::audio::decode(bytes)
        .await
        .map_err(|error| miette::miette!("{error}"))?;
    Ok(Decoded {
        duration_seconds: audio.duration_seconds(),
        peaks: peaks(&audio.samples, BARS),
    })
}

/// Decoding needs a desktop or web platform.
#[cfg(not(any(purr_web, feature = "desktop")))]
pub async fn decode(_file_id: &FileId, _bytes: Bytes) -> miette::Result<Decoded> {
    Err(miette::miette!(
        "Decoding audio is not supported on this platform"
    ))
}
