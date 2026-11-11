//! Turns an uploaded file into the 16 kHz mono samples Whisper expects.
//!
//! Decoding is left to the browser's Web Audio API, which handles every container and codec the
//! browser supports and resamples to the context's rate.

use crate::error::{AudioError, WebError, WebResult, format_js_error};
use crate::local::run_local;
use bytes::Bytes;
use js_sys::Uint8Array;
use wasm_bindgen::{JsCast as _, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{AudioBuffer, OfflineAudioContext};

/// Whisper's input sample rate.
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// Largest upload accepted, in bytes.
pub const MAX_AUDIO_BYTES: usize = 100 * 1024 * 1024;

/// Decoded audio ready for the worker.
#[derive(Debug)]
pub struct DecodedAudio {
    pub samples: Vec<f32>,
}

impl DecodedAudio {
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.samples.len() as f64 / f64::from(SAMPLE_RATE_HZ)
    }
}

/// Rejects uploads that are empty or larger than [`MAX_AUDIO_BYTES`].
pub fn validate_size(len: usize) -> Result<(), AudioError> {
    if len == 0 {
        return Err(AudioError::Empty);
    }
    if len > MAX_AUDIO_BYTES {
        return Err(AudioError::TooLarge {
            file_size: len as u64,
            max_size: MAX_AUDIO_BYTES as u64,
        });
    }
    Ok(())
}

/// Averages the channels into one. Channels of different lengths are averaged over the
/// channels that have a sample at that position.
#[must_use]
pub fn downmix_to_mono(channels: &[Vec<f32>]) -> Vec<f32> {
    let longest = channels.iter().map(Vec::len).max().unwrap_or(0);
    (0..longest)
        .map(|index| {
            let (sum, count) = channels
                .iter()
                .filter_map(|channel| channel.get(index))
                .fold((0.0_f32, 0_u32), |(sum, count), sample| {
                    (sum + sample, count + 1)
                });
            if count == 0 { 0.0 } else { sum / count as f32 }
        })
        .collect()
}

async fn decode_in_browser(bytes: Bytes) -> WebResult<DecodedAudio> {
    let context = OfflineAudioContext::new_with_number_of_channels_and_length_and_sample_rate(
        1,
        1,
        SAMPLE_RATE_HZ as f32,
    )
    .map_err(|e| WebError::web_audio_api("OfflineAudioContext", e))?;
    let encoded = Uint8Array::from(bytes.as_ref()).buffer();
    let promise = context
        .decode_audio_data(&encoded)
        .map_err(|e| WebError::web_audio_api("decodeAudioData", e))?;
    let decoded = JsFuture::from(promise)
        .await
        .map_err(|e| AudioError::DecodeFailed {
            message: format_js_error(e),
        })?;
    let buffer = decoded
        .dyn_into::<AudioBuffer>()
        .map_err(|e: JsValue| WebError::web_audio_api("decodeAudioData result", e))?;

    let channels = (0..buffer.number_of_channels())
        .map(|channel| {
            buffer
                .get_channel_data(channel)
                .map_err(|e| WebError::web_audio_api("getChannelData", e))
        })
        .collect::<WebResult<Vec<Vec<f32>>>>()?;
    let samples = downmix_to_mono(&channels);
    if samples.is_empty() {
        return Err(AudioError::NoSamples.into());
    }
    Ok(DecodedAudio { samples })
}

/// Decodes an uploaded file to 16 kHz mono samples.
pub async fn decode(bytes: Bytes) -> WebResult<DecodedAudio> {
    validate_size(bytes.len())?;
    run_local(move || decode_in_browser(bytes)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_upload_is_rejected() {
        assert!(matches!(validate_size(0), Err(AudioError::Empty)));
    }

    #[test]
    fn upload_at_the_limit_is_accepted_and_beyond_it_rejected() {
        assert!(validate_size(MAX_AUDIO_BYTES).is_ok());
        assert!(matches!(
            validate_size(MAX_AUDIO_BYTES + 1),
            Err(AudioError::TooLarge { .. })
        ));
    }

    #[test]
    fn stereo_is_averaged() {
        let stereo = vec![vec![1.0, 0.0, -1.0], vec![0.0, 0.5, 1.0]];
        assert_eq!(downmix_to_mono(&stereo), vec![0.5, 0.25, 0.0]);
    }

    #[test]
    fn mono_is_unchanged() {
        assert_eq!(downmix_to_mono(&[vec![0.25, -0.5]]), vec![0.25, -0.5]);
    }

    #[test]
    fn no_channels_give_no_samples() {
        assert_eq!(downmix_to_mono(&[]), Vec::<f32>::new());
    }

    #[test]
    fn shorter_channel_does_not_dilute_the_tail() {
        let uneven = vec![vec![1.0, 1.0], vec![0.0]];
        assert_eq!(downmix_to_mono(&uneven), vec![0.5, 1.0]);
    }

    #[test]
    fn duration_follows_the_sample_count() {
        let audio = DecodedAudio {
            samples: vec![0.0; SAMPLE_RATE_HZ as usize * 3],
        };
        assert!((audio.duration_seconds() - 3.0).abs() < f64::EPSILON);
    }
}
