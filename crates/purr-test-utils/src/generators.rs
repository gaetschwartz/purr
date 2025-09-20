//! Property-based test data generators for Purr components

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use fake::{Dummy, Fake, Faker};
use rand::seq::SliceRandom;
use std::time::Duration;

/// Generate valid audio sample rates
pub fn audio_sample_rate() -> impl Strategy<Value = u32> {
    prop_oneof![
        Just(8000),
        Just(16000),
        Just(22050),
        Just(44100),
        Just(48000),
        Just(96000),
    ]
}

/// Generate valid channel counts
pub fn audio_channels() -> impl Strategy<Value = u16> {
    1u16..=8u16
}

/// Generate audio duration in seconds
pub fn audio_duration() -> impl Strategy<Value = f32> {
    0.1f32..=3600.0f32 // 0.1 seconds to 1 hour
}

/// Generate confidence scores
pub fn confidence_score() -> impl Strategy<Value = f32> {
    0.0f32..=1.0f32
}

/// Generate valid model names
pub fn model_name() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("tiny".to_string()),
        Just("tiny.en".to_string()),
        Just("base".to_string()),
        Just("base.en".to_string()),
        Just("small".to_string()),
        Just("small.en".to_string()),
        Just("medium".to_string()),
        Just("medium.en".to_string()),
        Just("large".to_string()),
        Just("large-v1".to_string()),
        Just("large-v2".to_string()),
        Just("large-v3".to_string()),
    ]
}

/// Generate language codes
pub fn language_code() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("auto".to_string()),
        Just("en".to_string()),
        Just("es".to_string()),
        Just("fr".to_string()),
        Just("de".to_string()),
        Just("it".to_string()),
        Just("pt".to_string()),
        Just("ru".to_string()),
        Just("ja".to_string()),
        Just("ko".to_string()),
        Just("zh".to_string()),
    ]
}

/// Generate file sizes in bytes
pub fn file_size() -> impl Strategy<Value = u64> {
    1u64..=(1024 * 1024 * 1024) // 1 byte to 1GB
}

/// Generate buffer sizes
pub fn buffer_size() -> impl Strategy<Value = usize> {
    1usize..=(1024 * 1024) // 1 byte to 1MB
}

/// Generate timestamps
pub fn timestamp() -> impl Strategy<Value = Duration> {
    (0u64..=3600000u64).prop_map(Duration::from_millis)
}

/// Generate transcription text
pub fn transcription_text() -> impl Strategy<Value = String> {
    prop_oneof![
        // Common English phrases
        Just("Hello, world!".to_string()),
        Just("The quick brown fox jumps over the lazy dog.".to_string()),
        Just("This is a test transcription.".to_string()),
        // Multi-language samples
        Just("Bonjour le monde!".to_string()),
        Just("Hola mundo!".to_string()),
        Just("Hallo Welt!".to_string()),
        // Technical terms
        Just("WebGPU compute shader optimization".to_string()),
        Just("Neural network inference pipeline".to_string()),
        // Random text generation
        "[a-zA-Z0-9 .,!?'-]{10,200}".prop_map(|s| s),
    ]
}

/// Generate WebGPU workgroup sizes
pub fn webgpu_workgroup_size() -> impl Strategy<Value = (u32, u32, u32)> {
    (1u32..=256, 1u32..=256, 1u32..=64)
}

/// Generate GPU buffer usage flags
#[cfg(feature = "web")]
pub fn gpu_buffer_usage() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(
        prop_oneof![
            Just("COPY_SRC".to_string()),
            Just("COPY_DST".to_string()),
            Just("STORAGE".to_string()),
            Just("UNIFORM".to_string()),
            Just("VERTEX".to_string()),
            Just("INDEX".to_string()),
        ],
        1..=4,
    )
}

/// Custom generator for audio data
#[derive(Debug, Clone)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl Dummy<Faker> for AudioData {
    fn dummy_with_rng<R: fake::Rng + ?Sized>(_: &Faker, rng: &mut R) -> Self {
        let sample_rate = *[8000, 16000, 44100, 48000].choose(rng).unwrap();
        let channels = rng.gen_range(1..=2);
        let duration_secs = rng.gen_range(0.1..=10.0);
        let num_samples = (sample_rate as f32 * duration_secs * channels as f32) as usize;

        let samples: Vec<f32> = (0..num_samples)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        AudioData {
            samples,
            sample_rate,
            channels,
        }
    }
}

/// Custom generator for transcription results
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub confidence: f32,
    pub duration: Duration,
    pub timestamps: Vec<(Duration, Duration, String)>, // start, end, text
}

impl Dummy<Faker> for TranscriptionResult {
    fn dummy_with_rng<R: fake::Rng + ?Sized>(_: &Faker, rng: &mut R) -> Self {
        let text: String = fake::faker::lorem::en::Sentence(3..10).fake_with_rng(rng);
        let confidence = rng.gen_range(0.0..=1.0);
        let duration_ms = rng.gen_range(100..=10000);
        let duration = Duration::from_millis(duration_ms);

        // Generate word-level timestamps
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut timestamps = Vec::new();
        let mut current_time = Duration::ZERO;

        for word in words {
            let word_duration = Duration::from_millis(rng.gen_range(100..=500));
            let start = current_time;
            let end = current_time + word_duration;
            timestamps.push((start, end, word.to_string()));
            current_time = end + Duration::from_millis(rng.gen_range(10..=100)); // pause
        }

        TranscriptionResult {
            text,
            confidence,
            duration,
            timestamps,
        }
    }
}

/// Generator for WebGPU compute operations
#[cfg(feature = "web")]
#[derive(Debug, Clone)]
pub struct ComputeOperation {
    pub shader_source: String,
    pub workgroup_size: (u32, u32, u32),
    pub dispatch_size: (u32, u32, u32),
    pub buffer_size: usize,
}

#[cfg(feature = "web")]
impl Dummy<Faker> for ComputeOperation {
    fn dummy_with_rng<R: fake::Rng + ?Sized>(_: &Faker, rng: &mut R) -> Self {
        let operations = [
            "data[index] = data[index] * 2.0;",
            "data[index] = sqrt(data[index]);",
            "data[index] = data[index] + 1.0;",
            "data[index] = sin(data[index]);",
            "data[index] = data[index] * data[index];",
        ];

        let operation = operations.choose(rng).unwrap();
        let workgroup_x = 2u32.pow(rng.gen_range(1..=8)); // Power of 2, 2-256
        let workgroup_size = (workgroup_x, 1, 1);

        let buffer_size = rng.gen_range(64..=4096) * 4; // Multiple of 4 bytes
        let dispatch_size = ((buffer_size / 4 + workgroup_x as usize - 1) / workgroup_x as usize, 1, 1);

        let shader_source = format!(
            r#"
            @group(0) @binding(0)
            var<storage, read_write> data: array<f32>;

            @compute @workgroup_size({}, {}, {})
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let index = global_id.x;
                if (index >= arrayLength(&data)) {{
                    return;
                }}
                {}
            }}
            "#,
            workgroup_size.0, workgroup_size.1, workgroup_size.2, operation
        );

        ComputeOperation {
            shader_source,
            workgroup_size,
            dispatch_size: (dispatch_size.0 as u32, dispatch_size.1 as u32, dispatch_size.2 as u32),
            buffer_size,
        }
    }
}

/// Property test strategy for generating valid file paths
pub fn file_path() -> impl Strategy<Value = String> {
    prop::collection::vec("[a-zA-Z0-9_-]{1,20}", 1..=5)
        .prop_map(|segments| {
            let mut path = segments.join("/");
            path.push_str(".wav");
            path
        })
}

/// Property test strategy for generating URL-like strings
pub fn url_string() -> impl Strategy<Value = String> {
    (
        prop_oneof!["http", "https"],
        "[a-zA-Z0-9-]{3,20}",
        "[a-zA-Z]{2,4}",
        prop::option::of("[a-zA-Z0-9/_-]{0,50}"),
    ).prop_map(|(scheme, domain, tld, path)| {
        match path {
            Some(p) if !p.is_empty() => format!("{}://{}.{}/{}", scheme, domain, tld, p),
            _ => format!("{}://{}.{}", scheme, domain, tld),
        }
    })
}

/// Generate error messages for testing error handling
pub fn error_message() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("Network connection failed".to_string()),
        Just("Invalid audio format".to_string()),
        Just("Model not found".to_string()),
        Just("Insufficient memory".to_string()),
        Just("WebGPU device lost".to_string()),
        Just("Transcription timeout".to_string()),
        "[A-Z][a-z ]{10,50}".prop_map(|s| s),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::test_runner::TestRunner;

    #[test]
    fn test_audio_sample_rate_generator() {
        let mut runner = TestRunner::default();
        for _ in 0..100 {
            let rate = audio_sample_rate().new_tree(&mut runner).unwrap().current();
            assert!([8000, 16000, 22050, 44100, 48000, 96000].contains(&rate));
        }
    }

    #[test]
    fn test_confidence_score_generator() {
        let mut runner = TestRunner::default();
        for _ in 0..100 {
            let score = confidence_score().new_tree(&mut runner).unwrap().current();
            assert!(score >= 0.0 && score <= 1.0);
        }
    }

    #[test]
    fn test_fake_audio_data() {
        let audio: AudioData = Faker.fake();
        assert!(!audio.samples.is_empty());
        assert!([8000, 16000, 44100, 48000].contains(&audio.sample_rate));
        assert!(audio.channels >= 1 && audio.channels <= 2);
        assert!(audio.samples.iter().all(|&s| s >= -1.0 && s <= 1.0));
    }

    #[test]
    fn test_fake_transcription_result() {
        let transcription: TranscriptionResult = Faker.fake();
        assert!(!transcription.text.is_empty());
        assert!(transcription.confidence >= 0.0 && transcription.confidence <= 1.0);
        assert!(transcription.duration > Duration::ZERO);
        assert!(!transcription.timestamps.is_empty());
    }
}