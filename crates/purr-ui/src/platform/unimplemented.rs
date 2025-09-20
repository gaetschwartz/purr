use super::{Platform, PlatformError, TranscriptionRequest, TranscriptionStatus};
use bytes::Bytes;
use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tracing::{error, info};
use wasm_bindgen_futures::spawn_local;
pub(super) struct PlatformImpl {}

impl PlatformImpl {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait::async_trait]
impl Platform for PlatformImpl {
    async fn process_file(
        &self,
        file_data: Bytes,
        file_name: String,
    ) -> Result<String, PlatformError> {
        Err(PlatformError::Unsupported {
            operation: "process_file in WASM".into(),
        })
    }

    async fn transcribe(
        &self,
        file_id: String,
        request: TranscriptionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<TranscriptionStatus, PlatformError>> + Send>>,
        PlatformError,
    > {
        Err(PlatformError::Unsupported {
            operation: "transcribe in WASM".into(),
        })
    }

    async fn cleanup(&self, file_id: &str) -> Result<(), PlatformError> {
        Err(PlatformError::Unsupported {
            operation: "cleanup in WASM".into(),
        })
    }
}
