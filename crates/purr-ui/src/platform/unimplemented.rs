use super::{Platform, PlatformError, TranscriptionRequest, TranscriptionStatus};
use bytes::Bytes;
use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tracing::{error, info};
use wasm_bindgen_futures::spawn_local;

pub(super) type PlatformImpl = UnimplementedPlatformImpl;

pub(super) struct UnimplementedPlatformImpl {}

#[async_trait::async_trait]
impl Platform for UnimplementedPlatformImpl {
    async fn process_file(
        &self,
        file_data: Bytes,
        file_name: String,
    ) -> Result<String, PlatformError> {
        Err(PlatformError::Unsupported {
            operation: "process_file in WASM".into(),
        })
    }

    async fn new() -> Result<Self, PlatformError> {
        Ok(Self {})
    }

    async fn list_installed_models(&self) -> Result<Vec<String>, PlatformError> {
        Err(PlatformError::Unsupported {
            operation: "list_installed_models in WASM".into(),
        })
    }

    async fn list_available_devices(&self) -> Result<Vec<String>, PlatformError> {
        Err(PlatformError::Unsupported {
            operation: "list_available_devices in WASM".into(),
        })
    }

    async fn list_available_models(&self) -> Result<HashMap<String, String>, PlatformError> {
        Err(PlatformError::Unsupported {
            operation: "list_available_models in WASM".into(),
        })
    }

    async fn fetch_model(&self, model: &str) -> Result<(), PlatformError> {
        Err(PlatformError::Unsupported {
            operation: "download_model in WASM".into(),
        })
    }

    async fn get_model_info(&self, model: &str) -> Result<String, PlatformError> {
        Err(PlatformError::Unsupported {
            operation: "get_model_info in WASM".into(),
        })
    }

    async fn remove_model(&self, model: &str) -> Result<(), PlatformError> {
        Err(PlatformError::Unsupported {
            operation: "remove_model in WASM".into(),
        })
    }

    async fn is_model_installed(&self, model: &str) -> Result<bool, PlatformError> {
        Err(PlatformError::Unsupported {
            operation: "is_model_installed in WASM".into(),
        })
    }

    async fn get_model_path(&self, model: &str) -> Result<String, PlatformError> {
        Err(PlatformError::Unsupported {
            operation: "get_model_path in WASM".into(),
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
