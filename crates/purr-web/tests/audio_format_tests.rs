//! Tests for audio format detection and parsing (native-compatible)

use purr_web::{validate_audio_file, get_supported_formats};

/// Create test WAV file data (44.1kHz, 16-bit, stereo, 1 second)
fn create_test_wav_data() -> Vec<u8> {
    let mut wav_data = Vec::new();

    // RIFF header
    wav_data.extend_from_slice(b"RIFF");
    wav_data.extend_from_slice(&(36u32 + 176400).to_le_bytes()); // File size - 8
    wav_data.extend_from_slice(b"WAVE");

    // fmt chunk
    wav_data.extend_from_slice(b"fmt ");
    wav_data.extend_from_slice(&16u32.to_le_bytes()); // Chunk size
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // Audio format (PCM)
    wav_data.extend_from_slice(&2u16.to_le_bytes()); // Channels (stereo)
    wav_data.extend_from_slice(&44100u32.to_le_bytes()); // Sample rate
    wav_data.extend_from_slice(&176400u32.to_le_bytes()); // Byte rate
    wav_data.extend_from_slice(&4u16.to_le_bytes()); // Block align
    wav_data.extend_from_slice(&16u16.to_le_bytes()); // Bits per sample

    // data chunk
    wav_data.extend_from_slice(b"data");
    wav_data.extend_from_slice(&176400u32.to_le_bytes()); // Data size

    // Generate 1 second of sine wave (440 Hz)
    for i in 0..88200 { // 44100 samples per channel
        let t = i as f32 / 44100.0;
        let sample = (440.0 * 2.0 * std::f32::consts::PI * t).sin();
        let sample16 = (sample * 32767.0) as i16;

        // Stereo: same sample for both channels
        wav_data.extend_from_slice(&sample16.to_le_bytes());
        wav_data.extend_from_slice(&sample16.to_le_bytes());
    }

    wav_data
}

/// Create test MP3 header (simplified)
fn create_test_mp3_data() -> Vec<u8> {
    let mut mp3_data = Vec::new();

    // ID3v2 header
    mp3_data.extend_from_slice(b"ID3");
    mp3_data.extend_from_slice(&[3, 0]); // Version 2.3
    mp3_data.extend_from_slice(&[0]); // Flags
    mp3_data.extend_from_slice(&[0, 0, 0, 10]); // Size (syncsafe)

    // Padding
    mp3_data.extend_from_slice(&[0; 10]);

    // MP3 frame header (44.1kHz, stereo, 128kbps)
    mp3_data.extend_from_slice(&[0xFF, 0xFB, 0x90, 0x64]); // MPEG-1, Layer 3, 44.1kHz, stereo

    // Add some dummy frame data
    mp3_data.extend_from_slice(&[0x42; 400]); // Placeholder frame data

    mp3_data
}

/// Create test FLAC data
fn create_test_flac_data() -> Vec<u8> {
    let mut flac_data = Vec::new();

    // fLaC signature
    flac_data.extend_from_slice(b"fLaC");

    // STREAMINFO metadata block
    flac_data.push(0x00); // Block type 0 (STREAMINFO), last block = false
    flac_data.extend_from_slice(&[0, 0, 34]); // Block size = 34 bytes

    // STREAMINFO data
    flac_data.extend_from_slice(&[0x00, 0x10]); // Min block size
    flac_data.extend_from_slice(&[0x10, 0x00]); // Max block size
    flac_data.extend_from_slice(&[0x00, 0x00, 0x00]); // Min frame size
    flac_data.extend_from_slice(&[0x00, 0x00, 0x00]); // Max frame size

    // Sample rate (44100), channels (2), bits per sample (16)
    flac_data.extend_from_slice(&[0xAC, 0x44, 0x11]); // 44100 Hz = 0xAC44, shifted

    // Total samples in stream (44100 for 1 second)
    flac_data.extend_from_slice(&[0x01, 0x00, 0xAC, 0x44, 0x00]); // 44100 samples

    // MD5 signature (dummy)
    flac_data.extend_from_slice(&[0; 16]);

    flac_data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_file_validation() {
        let wav_data = create_test_wav_data();

        // Test file validation
        assert!(validate_audio_file(&wav_data, 1024 * 1024).is_ok());

        // File should be recognized as valid audio
        assert!(wav_data.starts_with(b"RIFF"));
        assert_eq!(&wav_data[8..12], b"WAVE");
    }

    #[test]
    fn test_mp3_file_validation() {
        let mp3_data = create_test_mp3_data();

        // Test file validation
        assert!(validate_audio_file(&mp3_data, 1024 * 1024).is_ok());

        // File should be recognized as valid MP3
        assert!(mp3_data.starts_with(b"ID3"));
    }

    #[test]
    fn test_flac_file_validation() {
        let flac_data = create_test_flac_data();

        // Test file validation
        assert!(validate_audio_file(&flac_data, 1024 * 1024).is_ok());

        // File should be recognized as valid FLAC
        assert!(flac_data.starts_with(b"fLaC"));
    }

    #[test]
    fn test_invalid_audio_file() {
        let invalid_data = vec![0x00; 100]; // Random bytes

        // Should fail validation
        assert!(validate_audio_file(&invalid_data, 1024 * 1024).is_err());
    }

    #[test]
    fn test_file_size_limit() {
        let large_data = vec![0x00; 200 * 1024 * 1024]; // 200MB

        // Should fail size validation
        assert!(validate_audio_file(&large_data, 100 * 1024 * 1024).is_err());
    }

    #[test]
    fn test_empty_file() {
        let empty_data = vec![];

        // Should fail validation
        assert!(validate_audio_file(&empty_data, 1024 * 1024).is_err());
    }

    #[test]
    fn test_too_small_file() {
        let small_data = vec![0x00; 5]; // Too small to be valid audio

        // Should fail validation
        assert!(validate_audio_file(&small_data, 1024 * 1024).is_err());
    }

    #[test]
    fn test_supported_formats() {
        let formats = get_supported_formats();

        // Should support common audio formats
        assert!(formats.contains_key("wav"));
        assert!(formats.contains_key("mp3"));
        assert!(formats.contains_key("flac"));
        assert!(formats.contains_key("ogg"));
        assert!(formats.contains_key("m4a"));
        assert!(formats.contains_key("webm"));

        // WAV should support PCM
        let wav_codecs = formats.get("wav").unwrap();
        assert!(wav_codecs.contains(&"PCM".to_string()));
    }

    #[test]
    fn test_wav_header_parsing() {
        let wav_data = create_test_wav_data();

        // Verify WAV structure
        assert_eq!(&wav_data[0..4], b"RIFF");
        assert_eq!(&wav_data[8..12], b"WAVE");

        // Find and verify fmt chunk
        let fmt_start = 12;
        assert_eq!(&wav_data[fmt_start..fmt_start + 4], b"fmt ");

        let fmt_size = u32::from_le_bytes([
            wav_data[fmt_start + 4], wav_data[fmt_start + 5],
            wav_data[fmt_start + 6], wav_data[fmt_start + 7]
        ]);
        assert_eq!(fmt_size, 16);

        // Verify audio format (PCM = 1)
        let audio_format = u16::from_le_bytes([
            wav_data[fmt_start + 8], wav_data[fmt_start + 9]
        ]);
        assert_eq!(audio_format, 1);

        // Verify channels (stereo = 2)
        let channels = u16::from_le_bytes([
            wav_data[fmt_start + 10], wav_data[fmt_start + 11]
        ]);
        assert_eq!(channels, 2);

        // Verify sample rate (44100)
        let sample_rate = u32::from_le_bytes([
            wav_data[fmt_start + 12], wav_data[fmt_start + 13],
            wav_data[fmt_start + 14], wav_data[fmt_start + 15]
        ]);
        assert_eq!(sample_rate, 44100);
    }

    #[test]
    fn test_mp3_header_parsing() {
        let mp3_data = create_test_mp3_data();

        // Verify ID3 header
        assert_eq!(&mp3_data[0..3], b"ID3");

        // Verify version
        assert_eq!(mp3_data[3], 3); // ID3v2.3

        // Find MP3 frame (after ID3 tag)
        let mp3_frame_start = 20; // After 10-byte ID3 header + 10 bytes padding
        assert_eq!(mp3_data[mp3_frame_start], 0xFF);
        assert_eq!(mp3_data[mp3_frame_start + 1] & 0xE0, 0xE0);
    }

    #[test]
    fn test_flac_header_parsing() {
        let flac_data = create_test_flac_data();

        // Verify fLaC signature
        assert_eq!(&flac_data[0..4], b"fLaC");

        // Verify STREAMINFO block
        assert_eq!(flac_data[4] & 0x7F, 0); // Block type 0 (STREAMINFO)

        let block_size = u32::from_be_bytes([
            0, flac_data[5], flac_data[6], flac_data[7]
        ]);
        assert_eq!(block_size, 34);
    }

    #[test]
    fn test_corrupted_headers() {
        // Test corrupted WAV
        let mut corrupted_wav = create_test_wav_data();
        corrupted_wav[0] = 0x00; // Corrupt RIFF signature
        assert!(validate_audio_file(&corrupted_wav, 1024 * 1024).is_err());

        // Test corrupted MP3
        let mut corrupted_mp3 = create_test_mp3_data();
        corrupted_mp3[0] = 0x00; // Corrupt ID3 signature
        assert!(validate_audio_file(&corrupted_mp3, 1024 * 1024).is_err());

        // Test corrupted FLAC
        let mut corrupted_flac = create_test_flac_data();
        corrupted_flac[0] = 0x00; // Corrupt fLaC signature
        assert!(validate_audio_file(&corrupted_flac, 1024 * 1024).is_err());
    }
}