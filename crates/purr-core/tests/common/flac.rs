use super::wav::SAMPLE_RATE;

const BLOCK_SIZE: u16 = 4096;
const STREAM_INFO_BYTES: u32 = 34;

/// The start of a mono 16-bit FLAC stream that says it holds `seconds` of audio, with no frames.
pub fn stream_info_only(seconds: u32) -> Vec<u8> {
    let total_samples = u64::from(seconds) * u64::from(SAMPLE_RATE);
    let packed = (u64::from(SAMPLE_RATE) << 44) | (15 << 36) | total_samples;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"fLaC");
    bytes.push(0x80);
    bytes.extend_from_slice(&STREAM_INFO_BYTES.to_be_bytes()[1..]);
    bytes.extend_from_slice(&BLOCK_SIZE.to_be_bytes());
    bytes.extend_from_slice(&BLOCK_SIZE.to_be_bytes());
    bytes.extend_from_slice(&[0; 6]);
    bytes.extend_from_slice(&packed.to_be_bytes());
    bytes.extend_from_slice(&[0; 16]);
    bytes
}
