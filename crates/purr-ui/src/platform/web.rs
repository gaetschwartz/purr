/// WASM platform implementation for browser builds
/// This implementation uses in-memory storage and will integrate with WebGPU-based whisper
use super::{Platform, PlatformError, TranscriptionRequest, TranscriptionStatus};
use bytes::Bytes;
use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tracing::{error, info};
use wasm_bindgen_futures::spawn_local;

pub(super) struct PlatformImpl {
    // In-memory storage for file data in browser
    storage: Arc<Mutex<HashMap<String, Bytes>>>,
}

impl PlatformImpl {
    pub fn new() -> Self {
        Self {
            storage: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl Platform for PlatformImpl {
    async fn process_file(
        &self,
        file_data: Bytes,
        file_name: String,
    ) -> Result<String, PlatformError> {
        todo!()
    }

    async fn transcribe(
        &self,
        file_id: String,
        request: TranscriptionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<TranscriptionStatus, PlatformError>> + Send>>,
        PlatformError,
    > {
        todo!()
    }

    async fn cleanup(&self, file_id: &str) -> Result<(), PlatformError> {
        todo!()
    }
}
