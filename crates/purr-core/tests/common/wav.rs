pub const SAMPLE_RATE: u32 = 16_000;

const HEADER_BYTES: u32 = 44;
const UNKNOWN_SIZE: u32 = 0xFFFF_FFFF;

fn header(riff_size: u32, data_size: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_BYTES as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    bytes.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());
    bytes
}

pub fn silence_bytes(seconds: u32) -> Vec<u8> {
    vec![0; (seconds * SAMPLE_RATE * 2) as usize]
}

/// A complete 16 kHz mono 16-bit WAV of silence whose header states its size.
pub fn wav(seconds: u32) -> Vec<u8> {
    let data = silence_bytes(seconds);
    let mut bytes = header(36 + data.len() as u32, data.len() as u32);
    bytes.extend_from_slice(&data);
    bytes
}

/// The header of a WAV that does not state how long it is, as a live encoder writes it.
pub fn open_ended_header() -> Vec<u8> {
    header(UNKNOWN_SIZE, UNKNOWN_SIZE)
}

/// Silence behind a header that does not state how long it is.
pub fn open_ended_wav(seconds: u32) -> Vec<u8> {
    let mut bytes = open_ended_header();
    bytes.extend_from_slice(&silence_bytes(seconds));
    bytes
}

/// The header of a WAV of `seconds` of silence, without the audio.
pub fn sized_header(seconds: u32) -> Vec<u8> {
    let size = seconds * SAMPLE_RATE * 2;
    header(36 + size, size)
}
