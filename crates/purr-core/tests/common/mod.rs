use purr_core::model::digest::{Sha256Digest, Sha256Hasher};
use purr_core::model::registry::Registry;
use std::error::Error;
use std::sync::Arc;

const REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";

pub const VAD_FILENAME: &str = "ggml-silero-fixture.bin";
pub const FIXTURE_FILENAME: &str = "ggml-fixture.bin";
pub const GGUF_FILENAME: &str = "fixture.gguf";

pub fn fixture_bytes() -> Vec<u8> {
    (0..300_000u32).map(|i| (i % 251) as u8).collect()
}

pub fn vad_fixture_bytes() -> Vec<u8> {
    (0..50_000u32).map(|i| (i % 241) as u8).collect()
}

pub fn digest_of(bytes: &[u8]) -> Sha256Digest {
    let mut hasher = Sha256Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

pub fn fixture_registry(bytes: &[u8]) -> Result<Arc<Registry>, Box<dyn Error>> {
    let vad = vad_fixture_bytes();
    let json = serde_json::json!({
        "schema": 3,
        "preferred": ["fixture"],
        "sources": [{"id": "main", "repo": "owner/repo", "revision": REVISION}],
        "profiles": {
            "plain": {
                "languages": "any", "language_hint": "steers", "detection": "reported",
                "timing": "word", "word_scores": true, "speakers": "none", "input": "unbounded"
            },
        },
        "models": [
            {
                "id": "fixture", "engine": "whisper-cpp", "format": "ggml-bin", "source": "main",
                "filename": FIXTURE_FILENAME, "family": "tiny", "profile": "plain",
                "quantization": "f16", "size_bytes": bytes.len(),
                "sha256": digest_of(bytes).to_string(), "description": "Fixture"
            },
            {
                "id": "fixture-gguf", "engine": "transcribe-cpp", "format": "gguf", "source": "main",
                "filename": GGUF_FILENAME, "family": "parakeet-tdt-0.6b-v3", "profile": "plain",
                "quantization": "f16", "size_bytes": bytes.len(),
                "sha256": digest_of(bytes).to_string(), "description": "Fixture GGUF"
            },
        ],
        "vad": {
            "id": "vad-fixture", "source": "main", "filename": VAD_FILENAME,
            "size_bytes": vad.len(), "sha256": digest_of(&vad).to_string()
        },
    });
    Ok(Arc::new(Registry::from_json(&json.to_string())?))
}
