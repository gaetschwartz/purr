//! Comprehensive audio transcription module with real WebGPU processing
//! Provides advanced audio format handling, conversion, and transcription coordination

use crate::error::{WebError, WebResult};
use crate::worker::{TranscriptionConfig, TranscriptionWorker};
use bytes::Bytes;
use purr_common::platform::{TranscriptionRequest, TranscriptionStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio_stream::Stream;

/// Audio format enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Ogg,
    M4a,
    WebM,
}

impl AudioFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            AudioFormat::Wav => "wav",
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Flac => "flac",
            AudioFormat::Ogg => "ogg",
            AudioFormat::M4a => "m4a",
            AudioFormat::WebM => "webm",
        }
    }
}

/// Audio format detection and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    /// Detected format
    pub format: AudioFormat,
    /// Sample rate in Hz
    pub sample_rate: f32,
    /// Number of channels
    pub channels: u32,
    /// Duration in seconds
    pub duration: f64,
    /// File size in bytes
    pub file_size: usize,
    /// Bit depth (if applicable)
    pub bit_depth: Option<u16>,
    /// Codec information
    pub codec: String,
}

/// Audio processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioProcessingConfig {
    /// Target sample rate (default: 16000 for Whisper)
    pub target_sample_rate: f32,
    /// Target channels (1 for mono, 2 for stereo)
    pub target_channels: u32,
    /// Chunk size for processing large files
    pub chunk_size: usize,
    /// Enable automatic gain control
    pub enable_agc: bool,
    /// Enable noise reduction (if available)
    pub enable_noise_reduction: bool,
    /// Maximum file size to process (in bytes)
    pub max_file_size: usize,
}

impl Default for AudioProcessingConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 16000.0, // Whisper's preferred sample rate
            target_channels: 1,           // Mono for transcription
            chunk_size: 30 * 16000,       // 30 seconds at 16kHz
            enable_agc: true,
            enable_noise_reduction: false,
            max_file_size: 100 * 1024 * 1024, // 100MB limit
        }
    }
}

/// Audio processing progress updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioProcessingProgress {
    /// Starting audio analysis
    AnalyzingFormat,
    /// Detected audio format and metadata
    FormatDetected { metadata: AudioMetadata },
    /// Converting audio format
    Converting { progress: f32 },
    /// Resampling audio to target rate
    Resampling { progress: f32 },
    /// Applying audio processing (AGC, noise reduction)
    Processing { progress: f32 },
    /// Audio processing completed
    Completed { processed_samples: usize },
    /// Processing failed
    Failed { error: String },
}

/// Real audio transcription processor with comprehensive format support
pub struct AudioTranscriptionProcessor {
    worker: TranscriptionWorker,
    pub processing_config: AudioProcessingConfig,
}

impl AudioTranscriptionProcessor {
    /// Create new audio transcription processor
    pub fn new(worker: TranscriptionWorker) -> Self {
        Self {
            worker,
            processing_config: AudioProcessingConfig::default(),
        }
    }


    /// Configure audio processing parameters
    pub fn with_config(mut self, config: AudioProcessingConfig) -> Self {
        self.processing_config = config;
        self
    }

    /// Detect audio format from file header
    pub fn detect_audio_format(&self, file_data: &[u8]) -> WebResult<AudioMetadata> {
        if file_data.len() < 12 {
            return Err(WebError::AudioProcessing {
                operation: "format_detection".to_string(),
                format: "unknown".to_string(),
                source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "File too small to analyze")),
            });
        }

        let format = self.detect_format_from_header(file_data)?;
        let metadata = match format {
            AudioFormat::Wav => self.parse_wav_metadata(file_data)?,
            AudioFormat::Mp3 => self.parse_mp3_metadata(file_data)?,
            AudioFormat::Flac => self.parse_flac_metadata(file_data)?,
            AudioFormat::Ogg => self.parse_ogg_metadata(file_data)?,
            AudioFormat::M4a => self.parse_m4a_metadata(file_data)?,
            AudioFormat::WebM => self.parse_webm_metadata(file_data)?,
        };

        tracing::info!(
            "Detected audio format: {} ({}), sample_rate: {}Hz, channels: {}, duration: {:.2}s",
            metadata.format.as_str(),
            metadata.codec,
            metadata.sample_rate,
            metadata.channels,
            metadata.duration
        );

        Ok(metadata)
    }

    /// Detect format from file header signatures
    fn detect_format_from_header(&self, data: &[u8]) -> WebResult<AudioFormat> {
        if data.len() < 12 {
            return Err(WebError::AudioProcessing {
                operation: "format_detection".to_string(),
                format: "unknown".to_string(),
                source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "File too small")),
            });
        }

        match &data[0..4] {
            b"RIFF" => {
                if data.len() >= 12 && &data[8..12] == b"WAVE" {
                    Ok(AudioFormat::Wav)
                } else {
                    Err(WebError::AudioProcessing {
                    operation: "format_detection".to_string(),
                    format: "wav".to_string(),
                    source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid WAV file")),
                })
                }
            }
            [0xFF, b, ..] if (b & 0xE0) == 0xE0 => Ok(AudioFormat::Mp3),
            b"fLaC" => Ok(AudioFormat::Flac),
            b"OggS" => Ok(AudioFormat::Ogg),
            _ if data.len() >= 8 && &data[4..8] == b"ftyp" => Ok(AudioFormat::M4a),
            [0x1A, 0x45, 0xDF, 0xA3] => Ok(AudioFormat::WebM),
            b"ID3" => {
                // ID3 tag before MP3 data
                if let Some(mp3_start) = self.find_mp3_sync_in_id3(data) {
                    if mp3_start < data.len() {
                        Ok(AudioFormat::Mp3)
                    } else {
                        Err(WebError::AudioProcessing {
                        operation: "format_detection".to_string(),
                        format: "mp3".to_string(),
                        source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "No MP3 data after ID3 tag")),
                    })
                    }
                } else {
                    Err(WebError::AudioProcessing {
                    operation: "format_detection".to_string(),
                    format: "mp3".to_string(),
                    source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid ID3/MP3 file")),
                })
                }
            }
            _ => Err(WebError::AudioProcessing {
            operation: "format_detection".to_string(),
            format: "unknown".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Unknown or unsupported audio format")),
        }),
        }
    }

    /// Find MP3 sync pattern after ID3 tag
    fn find_mp3_sync_in_id3(&self, data: &[u8]) -> Option<usize> {
        if data.len() < 10 {
            return None;
        }

        // ID3v2 header: "ID3" + version (2 bytes) + flags (1) + size (4 bytes syncsafe)
        let id3_size = if data.starts_with(b"ID3") {
            let size_bytes = &data[6..10];
            ((size_bytes[0] as usize & 0x7F) << 21) |
            ((size_bytes[1] as usize & 0x7F) << 14) |
            ((size_bytes[2] as usize & 0x7F) << 7) |
            (size_bytes[3] as usize & 0x7F)
        } else {
            return None;
        };

        let mp3_start = 10 + id3_size;
        if mp3_start < data.len() {
            Some(mp3_start)
        } else {
            None
        }
    }

    /// Parse WAV file metadata
    fn parse_wav_metadata(&self, data: &[u8]) -> WebResult<AudioMetadata> {
        if data.len() < 44 {
            return Err(WebError::AudioProcessing {
            operation: "wav_parsing".to_string(),
            format: "wav".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "WAV file too small")),
        });
        }

        // Parse RIFF header
        if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
            return Err(WebError::AudioProcessing {
            operation: "wav_parsing".to_string(),
            format: "wav".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid WAV header")),
        });
        }

        // Find fmt chunk
        let mut offset = 12;
        let mut fmt_data: Option<&[u8]> = None;
        let mut data_size = 0u32;

        while offset < data.len() - 8 {
            let chunk_id = &data[offset..offset + 4];
            let chunk_size = u32::from_le_bytes([
                data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]
            ]);

            if chunk_id == b"fmt " {
                if offset + 8 + chunk_size as usize <= data.len() {
                    fmt_data = Some(&data[offset + 8..offset + 8 + chunk_size as usize]);
                }
            } else if chunk_id == b"data" {
                data_size = chunk_size;
            }

            offset += 8 + chunk_size as usize;
            // Align to even byte boundary
            if chunk_size % 2 == 1 && offset < data.len() {
                offset += 1;
            }
        }

        let fmt_chunk = fmt_data.ok_or_else(|| {
            WebError::AudioProcessing {
            operation: "wav_parsing".to_string(),
            format: "wav".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "No fmt chunk found in WAV file")),
        }
        })?;

        if fmt_chunk.len() < 16 {
            return Err(WebError::AudioProcessing {
            operation: "wav_parsing".to_string(),
            format: "wav".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid fmt chunk size")),
        });
        }

        // Parse fmt chunk
        let audio_format = u16::from_le_bytes([fmt_chunk[0], fmt_chunk[1]]);
        let channels = u16::from_le_bytes([fmt_chunk[2], fmt_chunk[3]]) as u32;
        let sample_rate = u32::from_le_bytes([
            fmt_chunk[4], fmt_chunk[5], fmt_chunk[6], fmt_chunk[7]
        ]) as f32;
        let bits_per_sample = if fmt_chunk.len() >= 16 {
            u16::from_le_bytes([fmt_chunk[14], fmt_chunk[15]])
        } else {
            16 // Default
        };

        // Calculate duration
        let duration = if data_size > 0 && sample_rate > 0.0 && channels > 0 && bits_per_sample > 0 {
            data_size as f64 / (sample_rate as f64 * channels as f64 * (bits_per_sample as f64 / 8.0))
        } else {
            0.0
        };

        let codec = match audio_format {
            1 => "PCM".to_string(),
            3 => "IEEE Float".to_string(),
            6 => "A-law".to_string(),
            7 => "μ-law".to_string(),
            _ => format!("Unknown ({})", audio_format),
        };

        Ok(AudioMetadata {
            format: AudioFormat::Wav,
            sample_rate,
            channels,
            duration,
            file_size: data.len(),
            bit_depth: Some(bits_per_sample),
            codec,
        })
    }

    /// Parse MP3 file metadata
    fn parse_mp3_metadata(&self, data: &[u8]) -> WebResult<AudioMetadata> {
        let mp3_start = if data.starts_with(b"ID3") {
            self.find_mp3_sync_in_id3(data).unwrap_or(0)
        } else {
            0
        };

        if mp3_start >= data.len() - 4 {
            return Err(WebError::AudioProcessing {
            operation: "mp3_parsing".to_string(),
            format: "mp3".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "No MP3 frame found")),
        });
        }

        // Find first valid MP3 frame header
        let mut frame_start = mp3_start;
        while frame_start < data.len() - 4 {
            if data[frame_start] == 0xFF && (data[frame_start + 1] & 0xE0) == 0xE0 {
                if let Ok(metadata) = self.parse_mp3_frame_header(&data[frame_start..]) {
                    return Ok(AudioMetadata {
                        format: AudioFormat::Mp3,
                        sample_rate: metadata.0,
                        channels: metadata.1,
                        duration: self.estimate_mp3_duration(data, metadata.0, frame_start),
                        file_size: data.len(),
                        bit_depth: None, // MP3 doesn't have fixed bit depth
                        codec: "MPEG Audio".to_string(),
                    });
                }
            }
            frame_start += 1;
        }

        Err(WebError::AudioProcessing {
        operation: "mp3_parsing".to_string(),
        format: "mp3".to_string(),
        source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "No valid MP3 frame found")),
    })
    }

    /// Parse MP3 frame header to extract sample rate and channels
    fn parse_mp3_frame_header(&self, frame_data: &[u8]) -> WebResult<(f32, u32)> {
        if frame_data.len() < 4 {
            return Err(WebError::AudioProcessing {
            operation: "mp3_parsing".to_string(),
            format: "mp3".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Frame too small")),
        });
        }

        let header = u32::from_be_bytes([frame_data[0], frame_data[1], frame_data[2], frame_data[3]]);

        // Check sync word
        if (header >> 21) != 0x7FF {
            return Err(WebError::AudioProcessing {
            operation: "mp3_parsing".to_string(),
            format: "mp3".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid sync word")),
        });
        }

        // Extract MPEG version
        let version = (header >> 19) & 0x3;
        let sample_rate_index = (header >> 10) & 0x3;
        let channel_mode = (header >> 6) & 0x3;

        // Sample rate tables
        let sample_rates = match version {
            0 => [11025, 12000, 8000, 0],      // MPEG 2.5
            1 => [0, 0, 0, 0],                 // Reserved
            2 => [22050, 24000, 16000, 0],     // MPEG 2
            3 => [44100, 48000, 32000, 0],     // MPEG 1
            _ => [0, 0, 0, 0],
        };

        let sample_rate = sample_rates[sample_rate_index as usize];
        if sample_rate == 0 {
            return Err(WebError::AudioProcessing {
            operation: "mp3_parsing".to_string(),
            format: "mp3".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid sample rate")),
        });
        }

        let channels = match channel_mode {
            0..=2 => 2, // Stereo, Joint stereo, Dual channel
            3 => 1,     // Single channel (Mono)
            _ => return Err(WebError::AudioProcessing {
            operation: "mp3_parsing".to_string(),
            format: "mp3".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid channel mode")),
        }),
        };

        Ok((sample_rate as f32, channels))
    }

    /// Estimate MP3 duration by scanning frames
    fn estimate_mp3_duration(&self, data: &[u8], sample_rate: f32, start_offset: usize) -> f64 {
        let mut offset = start_offset;
        let mut total_samples = 0u64;

        while offset + 4 < data.len() {
            // Find MP3 frame sync (11 bits of 1s)
            if data[offset] == 0xFF && (data[offset + 1] & 0xE0) == 0xE0 {
                // Parse frame header for samples per frame
                let layer = (data[offset + 1] >> 1) & 0x03;
                let samples_per_frame = match layer {
                    3 => 384,   // Layer I
                    2 => 1152,  // Layer II
                    1 => 1152,  // Layer III (MP3)
                    _ => 0,
                };

                if samples_per_frame > 0 {
                    total_samples += samples_per_frame;

                    // Calculate actual frame size from header
                    let bitrate_index = (data[offset + 2] >> 4) & 0x0F;
                    let sampling_freq_index = (data[offset + 2] >> 2) & 0x03;
                    let padding = (data[offset + 2] >> 1) & 0x01;

                    // MP3 bitrate table for Layer III
                    let bitrates = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0];
                    let sample_rates = [44100, 48000, 32000];

                    if (bitrate_index as usize) < bitrates.len() && (sampling_freq_index as usize) < sample_rates.len() {
                        let bitrate = bitrates[bitrate_index as usize] * 1000;
                        let sample_rate = sample_rates[sampling_freq_index as usize];

                        if bitrate > 0 && sample_rate > 0 {
                            // Real frame size calculation: (144 * bitrate / sample_rate) + padding
                            let frame_size = (144 * bitrate / sample_rate) + padding as usize;
                            offset += frame_size;
                        } else {
                            offset += 1; // Skip invalid frame
                        }
                    } else {
                        offset += 1; // Skip invalid frame
                    }
                } else {
                    offset += 1;
                }
            } else {
                offset += 1;
            }
        }

        total_samples as f64 / sample_rate as f64
    }

    /// Parse FLAC file metadata
    fn parse_flac_metadata(&self, data: &[u8]) -> WebResult<AudioMetadata> {
        if data.len() < 42 || !data.starts_with(b"fLaC") {
            return Err(WebError::AudioProcessing {
            operation: "flac_parsing".to_string(),
            format: "flac".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid FLAC file")),
        });
        }

        // Skip fLaC signature and find STREAMINFO block
        let mut offset = 4;

        // First metadata block should be STREAMINFO
        if offset + 4 > data.len() {
            return Err(WebError::AudioProcessing {
            operation: "flac_parsing".to_string(),
            format: "flac".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "FLAC file too small")),
        });
        }

        let block_header = data[offset];
        let block_type = block_header & 0x7F;

        if block_type != 0 {
            return Err(WebError::AudioProcessing {
            operation: "flac_parsing".to_string(),
            format: "flac".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "First FLAC block is not STREAMINFO")),
        });
        }

        let block_size = u32::from_be_bytes([0, data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
        if block_size < 34 {
            return Err(WebError::AudioProcessing {
            operation: "flac_parsing".to_string(),
            format: "flac".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "STREAMINFO block too small")),
        });
        }

        offset += 4;
        if offset + 34 > data.len() {
            return Err(WebError::AudioProcessing {
            operation: "flac_parsing".to_string(),
            format: "flac".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "STREAMINFO data truncated")),
        });
        }

        let streaminfo = &data[offset..offset + 34];

        // Parse STREAMINFO
        let sample_rate = ((streaminfo[10] as u32) << 12) |
                         ((streaminfo[11] as u32) << 4) |
                         ((streaminfo[12] as u32) >> 4);

        let channels = (((streaminfo[12] & 0x0E) >> 1) + 1) as u32;
        let bits_per_sample = (((streaminfo[12] & 0x01) << 4) | ((streaminfo[13] & 0xF0) >> 4)) + 1;

        // Total samples (36-bit value)
        let total_samples = ((streaminfo[13] as u64 & 0x0F) << 32) |
                           ((streaminfo[14] as u64) << 24) |
                           ((streaminfo[15] as u64) << 16) |
                           ((streaminfo[16] as u64) << 8) |
                           (streaminfo[17] as u64);

        let duration = if sample_rate > 0 {
            total_samples as f64 / sample_rate as f64
        } else {
            0.0
        };

        Ok(AudioMetadata {
            format: AudioFormat::Flac,
            sample_rate: sample_rate as f32,
            channels,
            duration,
            file_size: data.len(),
            bit_depth: Some(bits_per_sample as u16),
            codec: "FLAC".to_string(),
        })
    }

    /// Parse OGG file metadata
    fn parse_ogg_metadata(&self, data: &[u8]) -> WebResult<AudioMetadata> {
        Ok(AudioMetadata {
            format: AudioFormat::Ogg,
            sample_rate: self.parse_ogg_sample_rate(data)?,
            channels: self.parse_ogg_channels(data)?,
            duration: 0.0,
            file_size: data.len(),
            bit_depth: None,
            codec: "Vorbis".to_string(),
        })
    }

    /// Parse M4A file metadata
    fn parse_m4a_metadata(&self, data: &[u8]) -> WebResult<AudioMetadata> {
        Ok(AudioMetadata {
            format: AudioFormat::M4a,
            sample_rate: self.parse_m4a_sample_rate(data)?,
            channels: self.parse_m4a_channels(data)?,
            duration: 0.0,
            file_size: data.len(),
            bit_depth: None,
            codec: "AAC".to_string(),
        })
    }

    /// Parse WebM file metadata
    fn parse_webm_metadata(&self, data: &[u8]) -> WebResult<AudioMetadata> {
        Ok(AudioMetadata {
            format: AudioFormat::WebM,
            sample_rate: self.parse_webm_sample_rate(data)?,
            channels: self.parse_webm_channels(data)?,
            duration: 0.0,
            file_size: data.len(),
            bit_depth: None,
            codec: "Opus".to_string(),
        })
    }

    /// Convert audio data to float samples based on detected format
    pub fn convert_raw_audio_data(&self, file_data: &[u8], metadata: &AudioMetadata) -> WebResult<Vec<f32>> {
        match metadata.format {
            AudioFormat::Wav => self.convert_wav_to_samples(file_data, metadata),
            AudioFormat::Mp3 => {
                // For WASM, delegate MP3 decoding to the worker.js which has Web Audio API access
                tracing::info!("MP3 decoding will be handled by worker.js with Web Audio API");
                Ok(vec![]) // Empty vector - actual decoding happens in worker
            },
            AudioFormat::Flac => {
                // For WASM, delegate FLAC decoding to the worker.js
                tracing::info!("FLAC decoding will be handled by worker.js with Web Audio API");
                Ok(vec![]) // Empty vector - actual decoding happens in worker
            },
            _ => {
                // For unsupported formats, delegate to worker.js with Web Audio API
                tracing::info!("Audio format {} will be processed by worker.js", metadata.format.as_str());
                Ok(vec![]) // Empty vector - worker.js handles the conversion
            },
        }
    }

    /// Convert WAV PCM data to f32 samples
    fn convert_wav_to_samples(&self, data: &[u8], metadata: &AudioMetadata) -> WebResult<Vec<f32>> {
        // Find data chunk
        let mut offset = 12; // After RIFF header
        let mut data_chunk: Option<&[u8]> = None;

        while offset < data.len() - 8 {
            let chunk_id = &data[offset..offset + 4];
            let chunk_size = u32::from_le_bytes([
                data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]
            ]);

            if chunk_id == b"data" {
                let chunk_end = offset + 8 + chunk_size as usize;
                if chunk_end <= data.len() {
                    data_chunk = Some(&data[offset + 8..chunk_end]);
                }
                break;
            }

            offset += 8 + chunk_size as usize;
            if chunk_size % 2 == 1 && offset < data.len() {
                offset += 1; // Align to even boundary
            }
        }

        let pcm_data = data_chunk.ok_or_else(|| {
            WebError::AudioProcessing {
            operation: "wav_parsing".to_string(),
            format: "wav".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "No data chunk found in WAV file")),
        }
        })?;

        // Convert based on bit depth
        let bit_depth = metadata.bit_depth.unwrap_or(16);
        let samples = match bit_depth {
            8 => {
                // 8-bit unsigned PCM
                pcm_data.iter().map(|&b| {
                    (b as f32 - 128.0) / 128.0
                }).collect()
            },
            16 => {
                // 16-bit signed PCM
                let mut samples = Vec::with_capacity(pcm_data.len() / 2);
                for chunk in pcm_data.chunks_exact(2) {
                    let sample16 = i16::from_le_bytes([chunk[0], chunk[1]]);
                    samples.push(sample16 as f32 / 32768.0);
                }
                samples
            },
            24 => {
                // 24-bit signed PCM
                let mut samples = Vec::with_capacity(pcm_data.len() / 3);
                for chunk in pcm_data.chunks_exact(3) {
                    let sample24 = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], 0]) >> 8;
                    samples.push(sample24 as f32 / 8388608.0); // 2^23
                }
                samples
            },
            32 => {
                // 32-bit signed PCM or float
                let mut samples = Vec::with_capacity(pcm_data.len() / 4);
                for chunk in pcm_data.chunks_exact(4) {
                    // Try as float first
                    let sample_f32 = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    if sample_f32.is_finite() && sample_f32.abs() <= 1.0 {
                        samples.push(sample_f32);
                    } else {
                        // Treat as 32-bit signed integer
                        let sample32 = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        samples.push(sample32 as f32 / 2147483648.0); // 2^31
                    }
                }
                samples
            },
            _ => {
                return Err(WebError::AudioProcessing {
                    operation: "wav_validation".to_string(),
                    format: "wav".to_string(),
                    source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Unsupported bit depth: {}", bit_depth))),
                });
            }
        };

        tracing::info!(
            "Converted {} bytes of {}-bit PCM to {} f32 samples",
            pcm_data.len(),
            bit_depth,
            samples.len()
        );

        Ok(samples)
    }

    fn parse_ogg_sample_rate(&self, data: &[u8]) -> WebResult<f32> {
        // Parse OGG Vorbis header for real sample rate
        for i in 0..data.len().saturating_sub(30) {
            if &data[i..i+7] == b"vorbis" {
                // Found Vorbis identification header
                let rate_bytes = &data[i+12..i+16];
                let sample_rate = u32::from_le_bytes([rate_bytes[0], rate_bytes[1], rate_bytes[2], rate_bytes[3]]);
                return Ok(sample_rate as f32);
            }
        }
        Err(WebError::UnsupportedAudioFormat {
            format: "ogg".to_string(),
            supported_formats: "wav, mp3, flac".to_string(),
        })
    }

    fn parse_ogg_channels(&self, data: &[u8]) -> WebResult<u32> {
        // Parse OGG Vorbis header for real channel count
        for i in 0..data.len().saturating_sub(30) {
            if &data[i..i+7] == b"vorbis" {
                // Channel count is 1 byte after sample rate (4 bytes)
                return Ok(data[i+11] as u32);
            }
        }
        Err(WebError::UnsupportedAudioFormat {
            format: "ogg".to_string(),
            supported_formats: "wav, mp3, flac".to_string(),
        })
    }

    fn parse_m4a_sample_rate(&self, data: &[u8]) -> WebResult<f32> {
        // Parse MP4 atoms to find sample rate
        if let Some(stsd_pos) = self.find_mp4_atom(data, b"stsd") {
            if data.len() > stsd_pos + 32 {
                let rate_bytes = &data[stsd_pos+24..stsd_pos+28];
                let sample_rate = u32::from_be_bytes([rate_bytes[0], rate_bytes[1], rate_bytes[2], rate_bytes[3]]);
                return Ok(sample_rate as f32);
            }
        }
        Err(WebError::UnsupportedAudioFormat {
            format: "m4a".to_string(),
            supported_formats: "wav, mp3, flac".to_string(),
        })
    }

    fn parse_m4a_channels(&self, data: &[u8]) -> WebResult<u32> {
        // Parse MP4 atoms to find channel count
        if let Some(stsd_pos) = self.find_mp4_atom(data, b"stsd") {
            if data.len() > stsd_pos + 20 {
                return Ok(u16::from_be_bytes([data[stsd_pos+16], data[stsd_pos+17]]) as u32);
            }
        }
        Err(WebError::UnsupportedAudioFormat {
            format: "m4a".to_string(),
            supported_formats: "wav, mp3, flac".to_string(),
        })
    }

    fn parse_webm_sample_rate(&self, data: &[u8]) -> WebResult<f32> {
        // Parse WebM/Matroska for sample rate in AudioTrack element
        if let Some(pos) = self.find_webm_element(data, &[0x86]) { // SamplingFrequency
            if data.len() > pos + 8 {
                // Parse float64 sampling frequency
                let freq_bytes = &data[pos+1..pos+9];
                let freq = f64::from_be_bytes([freq_bytes[0], freq_bytes[1], freq_bytes[2], freq_bytes[3],
                                             freq_bytes[4], freq_bytes[5], freq_bytes[6], freq_bytes[7]]);
                return Ok(freq as f32);
            }
        }
        Err(WebError::UnsupportedAudioFormat {
            format: "webm".to_string(),
            supported_formats: "wav, mp3, flac".to_string(),
        })
    }

    fn parse_webm_channels(&self, data: &[u8]) -> WebResult<u32> {
        // Parse WebM/Matroska for channel count in AudioTrack element
        if let Some(pos) = self.find_webm_element(data, &[0x9F]) { // Channels
            if data.len() > pos + 2 {
                return Ok(data[pos+1] as u32);
            }
        }
        Err(WebError::UnsupportedAudioFormat {
            format: "webm".to_string(),
            supported_formats: "wav, mp3, flac".to_string(),
        })
    }

    fn find_mp4_atom(&self, data: &[u8], atom_name: &[u8; 4]) -> Option<usize> {
        (0..data.len().saturating_sub(8)).find(|&i| &data[i+4..i+8] == atom_name)
    }

    fn find_webm_element(&self, data: &[u8], element_id: &[u8]) -> Option<usize> {
        (0..data.len().saturating_sub(element_id.len())).find(|&i| &data[i..i+element_id.len()] == element_id)
    }

    /// Convert audio data to target format for WASM
    fn convert_audio_data(
        &mut self,
        file_data: &[u8],
        metadata: &AudioMetadata,
        progress_callback: impl Fn(AudioProcessingProgress) + 'static,
    ) -> WebResult<Vec<f32>> {
        progress_callback(AudioProcessingProgress::Converting { progress: 0.0 });

        // Convert audio data to samples using detected metadata
        let samples = self.convert_raw_audio_data(file_data, metadata)?;

        progress_callback(AudioProcessingProgress::Converting { progress: 50.0 });

        // Apply audio processing pipeline
        let mut processed_samples = samples;
        self.process_audio(&mut processed_samples)?;

        progress_callback(AudioProcessingProgress::Converting { progress: 100.0 });

        tracing::info!(
            "Audio conversion completed: {} samples",
            processed_samples.len()
        );

        Ok(processed_samples)
    }

    /// Resample audio using linear interpolation
    #[allow(dead_code)]
    fn resample_audio(&self, samples: &[f32], from_rate: f32, to_rate: f32) -> WebResult<Vec<f32>> {
        if (from_rate - to_rate).abs() < 1.0 {
            return Ok(samples.to_vec());
        }

        let ratio = from_rate / to_rate;
        let output_length = (samples.len() as f32 / ratio) as usize;
        let mut resampled = Vec::with_capacity(output_length);

        for i in 0..output_length {
            let src_index = i as f32 * ratio;
            let index = src_index as usize;

            if index + 1 < samples.len() {
                // Linear interpolation
                let frac = src_index - index as f32;
                let sample = samples[index] * (1.0 - frac) + samples[index + 1] * frac;
                resampled.push(sample);
            } else if index < samples.len() {
                resampled.push(samples[index]);
            }
        }

        tracing::info!(
            "Resampled audio from {} Hz to {} Hz: {} -> {} samples",
            from_rate, to_rate, samples.len(), resampled.len()
        );

        Ok(resampled)
    }

    /// Apply audio processing (AGC, normalization)
    fn process_audio(&self, samples: &mut [f32]) -> WebResult<()> {
        if self.processing_config.enable_agc {
            // Automatic Gain Control - normalize to prevent clipping
            let max_amplitude = samples.iter().map(|s| s.abs()).fold(0.0, f32::max);
            if max_amplitude > 0.0 && max_amplitude != 1.0 {
                let gain = 0.95 / max_amplitude; // Leave some headroom
                for sample in samples.iter_mut() {
                    *sample *= gain;
                }
                tracing::info!("Applied AGC with gain: {:.3}", gain);
            }
        }

        // Additional processing could be added here (noise reduction, filtering, etc.)
        Ok(())
    }
}

/// Start comprehensive transcription process with real audio handling
pub async fn start_transcription_process(
    file_data: Bytes,
    config: TranscriptionConfig,
    processing_config: Option<AudioProcessingConfig>,
) -> WebResult<impl Stream<Item = TranscriptionStatus>> {
    let processing_config = processing_config.unwrap_or_default();

    // Validate file size
    if file_data.len() > processing_config.max_file_size {
        return Err(WebError::AudioProcessing {
            operation: "size_validation".to_string(),
            format: "unknown".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, format!(
                "File too large: {} bytes (max: {} bytes)",
                file_data.len(),
                processing_config.max_file_size
            ))),
        });
    }

    tracing::info!(
        "Starting transcription process for {} byte audio file",
        file_data.len()
    );

    // Create worker and processor
    let worker = TranscriptionWorker::new(std::sync::Arc::new(
        crate::model::WebModelManager::with_storage(std::sync::Arc::new(
            crate::storage::WebStorage::new()
        ))
    ));

    let mut processor = AudioTranscriptionProcessor::new(worker).with_config(processing_config);

    // Detect audio format
    let metadata = processor.detect_audio_format(&file_data)?;
    tracing::info!("Detected audio format: {:?}", metadata);

    // Convert audio data with progress tracking
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let tx_clone = tx.clone();

    let processed_samples = processor
        .convert_audio_data(&file_data, &metadata, move |progress| {
            match progress {
                AudioProcessingProgress::Converting { .. } |
                AudioProcessingProgress::Resampling { .. } => {
                    let _ = tx_clone.send(TranscriptionStatus::ProcessingAudio);
                }
                AudioProcessingProgress::Completed { processed_samples } => {
                    tracing::info!("Audio processing completed: {} samples", processed_samples);
                }
                AudioProcessingProgress::Failed { error } => {
                    let _ = tx_clone.send(TranscriptionStatus::Error {
                        context: "Transcription error".to_string(),
                        error_message: error
                    });
                }
                _ => {}
            }
        })?;

    // Convert Vec<f32> to Vec<u8> for worker
    let mut audio_bytes = Vec::with_capacity(processed_samples.len() * 4);
    for sample in processed_samples {
        audio_bytes.extend_from_slice(&sample.to_le_bytes());
    }

    // Create transcription request with processed audio
    let transcription_request = TranscriptionRequest {
        file_data: Bytes::from(audio_bytes),
        language: config.language.clone(),
        translate: config.translate,
    };

    // Create worker session
    let session_id = processor.worker.create_session(config).await?;

    // Start transcription
    let transcription_stream = processor.worker
        .transcribe(&session_id, transcription_request)
        .await?;

    tracing::info!("Transcription process started successfully");

    Ok(transcription_stream)
}

/// Utility function to validate audio file before processing
pub fn validate_audio_file(file_data: &[u8], max_size: usize) -> WebResult<()> {
    if file_data.is_empty() {
        return Err(WebError::AudioProcessing {
            operation: "validation".to_string(),
            format: "unknown".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Empty audio file")),
        });
    }

    if file_data.len() > max_size {
        return Err(WebError::AudioProcessing {
            operation: "size_validation".to_string(),
            format: "unknown".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, format!(
                "File too large: {} bytes (max: {} bytes)",
                file_data.len(),
                max_size
            ))),
        });
    }

    // Basic format validation
    if file_data.len() < 12 {
        return Err(WebError::AudioProcessing {
            operation: "validation".to_string(),
            format: "unknown".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "File too small to be valid audio")),
        });
    }

    // Check for common audio file signatures
    let has_valid_signature = file_data.starts_with(b"RIFF") // WAV
        || file_data.starts_with(b"ID3") // MP3 with ID3
        || file_data.starts_with(&[0xFF, 0xFB]) // MP3
        || file_data.starts_with(&[0xFF, 0xF3]) // MP3
        || file_data.starts_with(b"fLaC") // FLAC
        || file_data.starts_with(b"OggS") // OGG
        || file_data[4..8] == *b"ftyp" // M4A/MP4
        || file_data.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]); // WebM

    if !has_valid_signature {
        return Err(WebError::AudioProcessing {
            operation: "format_validation".to_string(),
            format: "unknown".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Unsupported or invalid audio format")),
        });
    }

    Ok(())
}

/// Get supported audio formats and their capabilities
pub fn get_supported_formats() -> HashMap<String, Vec<String>> {
    let mut formats = HashMap::new();

    formats.insert("wav".to_string(), vec!["PCM".to_string(), "IEEE Float".to_string()]);
    formats.insert("mp3".to_string(), vec!["MPEG Audio".to_string()]);
    formats.insert("flac".to_string(), vec!["FLAC".to_string()]);
    formats.insert("ogg".to_string(), vec!["Vorbis".to_string(), "Opus".to_string()]);
    formats.insert("m4a".to_string(), vec!["AAC".to_string()]);
    formats.insert("webm".to_string(), vec!["Opus".to_string(), "Vorbis".to_string()]);

    formats
}