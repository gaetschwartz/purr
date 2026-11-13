//! Conversion of decoded audio frames to the 16 kHz mono f32 samples Whisper expects.

use crate::error::{AudioProcessingError, Result, WhisperError};
use ffmpeg::channel_layout::ChannelLayout;
use ffmpeg::format::{Sample, sample::Type};
use ffmpeg::frame;
use ffmpeg::software::resampling::Context;
use ffmpeg_next as ffmpeg;
use tracing::warn;

const TARGET_RATE: u32 = 16_000;
const TARGET_FORMAT: Sample = Sample::F32(Type::Planar);

#[derive(Clone, Copy, PartialEq)]
struct InputFormat {
    format: Sample,
    layout: ChannelLayout,
    rate: u32,
}

impl InputFormat {
    fn of(frame: &frame::Audio) -> Self {
        let layout = if frame.channel_layout().channels() == 0 {
            match frame.channels() {
                1 => ChannelLayout::MONO,
                2 => ChannelLayout::STEREO,
                _ => ChannelLayout::default(i32::from(frame.channels())),
            }
        } else {
            frame.channel_layout()
        };
        Self {
            format: frame.format(),
            layout,
            rate: frame.rate(),
        }
    }
}

/// Stateful converter: one resampler lives across all frames of a stream so no samples are
/// dropped or duplicated at frame boundaries.
#[derive(Default)]
pub struct FrameResampler {
    active: Option<(InputFormat, Context)>,
}

impl FrameResampler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resample one decoded frame and append the produced samples.
    pub fn push(&mut self, frame: &frame::Audio, samples: &mut Vec<f32>) -> Result<()> {
        let input = InputFormat::of(frame);

        if self
            .active
            .as_ref()
            .is_some_and(|(current, _)| *current != input)
        {
            self.finish(samples)?;
        }

        if self.active.is_none() {
            let context = Context::get(
                input.format,
                input.layout,
                input.rate,
                TARGET_FORMAT,
                ChannelLayout::MONO,
                TARGET_RATE,
            )
            .map_err(WhisperError::from)?;
            self.active = Some((input, context));
        }

        let Some((_, context)) = self.active.as_mut() else {
            return Ok(());
        };

        if let Err(e) = convert(context, frame, samples) {
            self.active = None;
            if fallback_convert(frame, samples, input.rate).is_err() {
                warn!("Skipping frame due to resampling error: {}", e);
            }
        }

        Ok(())
    }

    /// Drain the samples the resampler still holds; call once after the last frame.
    pub fn finish(&mut self, samples: &mut Vec<f32>) -> Result<()> {
        let Some((_, mut context)) = self.active.take() else {
            return Ok(());
        };

        while let Some(delay) = context.delay() {
            let capacity = usize::try_from(delay.output).unwrap_or(0).saturating_add(1);
            let mut out = frame::Audio::new(TARGET_FORMAT, capacity, ChannelLayout::MONO);
            context.flush(&mut out).map_err(WhisperError::from)?;
            if out.samples() == 0 {
                break;
            }
            append(&out, samples);
        }

        Ok(())
    }
}

fn convert(
    context: &mut Context,
    frame: &frame::Audio,
    samples: &mut Vec<f32>,
) -> std::result::Result<(), ffmpeg::Error> {
    let buffered = context
        .delay()
        .map_or(0, |d| usize::try_from(d.output).unwrap_or(0));
    let produced = frame
        .samples()
        .saturating_mul(TARGET_RATE as usize)
        .div_ceil(frame.rate().max(1) as usize);

    // swr_convert_frame reads a preallocated frame's nb_samples as capacity and overwrites it with the output count: never reuse the frame
    let mut out = frame::Audio::new(
        TARGET_FORMAT,
        produced.saturating_add(buffered).saturating_add(1),
        ChannelLayout::MONO,
    );
    context.run(frame, &mut out)?;
    append(&out, samples);
    Ok(())
}

fn append(out: &frame::Audio, samples: &mut Vec<f32>) {
    if out.planes() > 0 {
        samples.extend_from_slice(out.plane::<f32>(0));
    }
}

/// Fallback processing for frames that fail resampling
pub fn fallback_convert(
    frame: &ffmpeg::frame::Audio,
    samples: &mut Vec<f32>,
    current_rate: u32,
) -> Result<()> {
    let current_format = frame.format();
    let data = frame.data(0);
    let sample_count = frame.samples();
    let channels = frame.channels();

    if sample_count == 0 {
        return Ok(());
    }

    match current_format {
        ffmpeg::format::Sample::I16(sample_type) => {
            unsafe {
                let ptr = data.as_ptr().cast::<i16>();
                let slice = std::slice::from_raw_parts(ptr, sample_count * channels as usize);

                let step = current_rate as f32 / 16000.0;
                let mut pos = 0.0;

                match sample_type {
                    ffmpeg::format::sample::Type::Packed => {
                        // Interleaved format (LRLRLR...)
                        while (pos as usize * channels as usize) < slice.len() {
                            let idx = (pos as usize) * channels as usize;
                            // Convert to mono by averaging channels or taking left channel
                            let mono_sample = if channels == 1 {
                                slice[idx]
                            } else {
                                // Average all channels for mono conversion
                                let sum: i32 = (0..channels)
                                    .map(|c| i32::from(slice[idx + c as usize]))
                                    .sum();
                                (sum / i32::from(channels)) as i16
                            };
                            samples.push(f32::from(mono_sample) / 32768.0);
                            pos += step;
                        }
                    }
                    ffmpeg::format::sample::Type::Planar => {
                        // Planar format - take first channel only
                        while (pos as usize) < sample_count {
                            let idx = pos as usize;
                            samples.push(f32::from(slice[idx]) / 32768.0);
                            pos += step;
                        }
                    }
                }
            }
        }
        ffmpeg::format::Sample::F32(sample_type) => {
            unsafe {
                let ptr = data.as_ptr().cast::<f32>();
                let slice = std::slice::from_raw_parts(ptr, sample_count * channels as usize);

                let step = current_rate as f32 / 16000.0;
                let mut pos = 0.0;

                match sample_type {
                    ffmpeg::format::sample::Type::Packed => {
                        // Interleaved format (LRLRLR...)
                        while (pos as usize * channels as usize) < slice.len() {
                            let idx = (pos as usize) * channels as usize;
                            // Convert to mono by averaging channels or taking left channel
                            let mono_sample = if channels == 1 {
                                slice[idx]
                            } else {
                                // Average all channels for mono conversion
                                let sum: f32 = (0..channels).map(|c| slice[idx + c as usize]).sum();
                                sum / f32::from(channels)
                            };
                            samples.push(mono_sample);
                            pos += step;
                        }
                    }
                    ffmpeg::format::sample::Type::Planar => {
                        // Planar format - take first channel only
                        while (pos as usize) < sample_count {
                            let idx = pos as usize;
                            samples.push(slice[idx]);
                            pos += step;
                        }
                    }
                }
            }
        }
        _ => {
            return Err(WhisperError::from(AudioProcessingError::FormatConversion));
        }
    }

    Ok(())
}
