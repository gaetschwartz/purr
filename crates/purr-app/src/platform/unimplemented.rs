#![allow(dead_code)]

use super::{Platform, PlatformError, TranscriptionRequest};
use bytes::Bytes;
use purr_common::settings::Settings;
use purr_common::{
    audio_overview::AudioOverview,
    history::{
        entry::{HistoryEntry, HistoryId, HistoryRecord},
        listing::{HistoryListing, Saved},
    },
    platform::{FileId, TranscriptionStream},
};
use std::path::Path;

pub type PlatformImpl = UnimplementedPlatformImpl;

pub struct UnimplementedPlatformImpl {}

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
#[allow(
    clippy::unused_async_trait_impl,
    reason = "the trait methods are async"
)]
impl Platform for UnimplementedPlatformImpl {
    fn install_logging_hooks(&self) {}

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

    fn max_upload_bytes(&self) -> Option<u64> {
        None
    }

    async fn audio_overview(
        &self,
        file_id: &FileId,
        buckets: usize,
    ) -> Result<AudioOverview, PlatformError> {
        unsupported!()
    }

    async fn list_history(&self) -> Result<HistoryListing, PlatformError> {
        unsupported!()
    }

    async fn get_history_entry(&self, id: &HistoryId) -> Result<HistoryRecord, PlatformError> {
        unsupported!()
    }

    async fn save_history_entry(&self, entry: &HistoryEntry) -> Result<Saved, PlatformError> {
        unsupported!()
    }

    async fn rename_history_entry(&self, id: &HistoryId, title: &str) -> Result<(), PlatformError> {
        unsupported!()
    }

    async fn delete_history_entry(&self, id: &HistoryId) -> Result<(), PlatformError> {
        unsupported!()
    }

    async fn clear_history(&self) -> Result<usize, PlatformError> {
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

    async fn verify_model(
        &self,
        model_id: &str,
    ) -> Result<purr_common::platform::ModelIntegrity, PlatformError> {
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

    async fn load_settings(&self) -> Result<Settings, PlatformError> {
        unsupported!()
    }

    async fn save_settings(&self, _settings: &Settings) -> Result<(), PlatformError> {
        unsupported!()
    }

    async fn get_settings_location(&self) -> Result<Option<String>, PlatformError> {
        unsupported!()
    }
}
