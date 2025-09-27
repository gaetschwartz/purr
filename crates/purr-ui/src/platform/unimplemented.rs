#![allow(dead_code)]

use super::{Platform, PlatformError, TranscriptionRequest};
use bytes::Bytes;
use purr_common::platform::TranscriptionStream;
use std::path::Path;

pub(super) type PlatformImpl = UnimplementedPlatformImpl;

pub(super) struct UnimplementedPlatformImpl {}

macro_rules! unsupported {
    () => {
        return Err(purr_common::platform::UnsupportedPlatformError.into())
    };
}

impl UnimplementedPlatformImpl {
    pub fn new() -> Result<Self, PlatformError> {
        unsupported!()
    }
}

#[allow(unused)]
impl Platform for UnimplementedPlatformImpl {
    async fn process_file(
        &self,
        file_data: Bytes,
        file_path: &Path,
    ) -> Result<purr_common::platform::FileId, PlatformError> {
        unsupported!()
    }

    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionStream, PlatformError> {
        unsupported!()
    }

    async fn cleanup(&self, file_id: &str) -> Result<(), PlatformError> {
        unsupported!()
    }

    async fn list_installed_models(
        &self,
    ) -> Result<Vec<purr_common::platform::ModelInfo>, PlatformError> {
        unsupported!()
    }

    async fn list_available_models(
        &self,
    ) -> Result<Vec<purr_common::platform::ModelInfo>, PlatformError> {
        unsupported!()
    }

    async fn fetch_model(
        &self,
        model_id: &str,
    ) -> Result<purr_common::platform::ModelProgressStream, PlatformError> {
        unsupported!()
    }

    async fn get_model_info(
        &self,
        model_id: &str,
    ) -> Result<purr_common::platform::ModelInfo, PlatformError> {
        unsupported!()
    }

    async fn remove_model(&self, model_id: &str) -> Result<(), PlatformError> {
        unsupported!()
    }

    async fn is_model_installed(&self, model_id: &str) -> Result<bool, PlatformError> {
        unsupported!()
    }

    async fn get_model_path(&self, model_id: &str) -> Result<String, PlatformError> {
        unsupported!()
    }

    async fn list_available_devices(
        &self,
    ) -> Result<Vec<purr_common::platform::DeviceInfo>, PlatformError> {
        unsupported!()
    }
}
