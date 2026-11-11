use purr_core::model::digest::{Sha256Digest, Sha256Hasher};
use purr_core::model::registry::Registry;
use std::error::Error;
use std::sync::Arc;

const REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";

pub fn fixture_bytes() -> Vec<u8> {
    (0..300_000u32).map(|i| (i % 251) as u8).collect()
}

pub fn digest_of(bytes: &[u8]) -> Sha256Digest {
    let mut hasher = Sha256Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

pub fn fixture_registry(bytes: &[u8]) -> Result<Arc<Registry>, Box<dyn Error>> {
    let json = serde_json::json!({
        "schema": 1,
        "preferred": ["fixture"],
        "sources": [{"id": "main", "repo": "owner/repo", "revision": REVISION}],
        "models": [{
            "id": "fixture", "source": "main", "filename": "ggml-fixture.bin",
            "family": "tiny", "language": "multilingual", "quantization": "fp16",
            "features": [], "size_bytes": bytes.len(), "sha256": digest_of(bytes).to_string(),
            "description": "Fixture"
        }],
    });
    Ok(Arc::new(Registry::from_json(&json.to_string())?))
}
