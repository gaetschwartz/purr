use super::use_models::ModelsHandle;
use crate::{
    components::toast::{Toast, ToastKind},
    model_catalog::{
        activity::{Damage, DownloadFailure, DownloadProgress, ModelActivity, SpeedMeter},
        entry::ModelEntry,
    },
    platform,
    utils::{byte_size::ByteSize, clock::now_ms},
};
use dioxus::prelude::ReadableExt as _;
use futures::StreamExt as _;
use purr_common::platform::{
    ModelFailureKind, ModelIntegrity, ModelOperationProgress, Platform as _,
};

const STOPPED_EARLY: &str = "The download stopped before it finished.";

enum Outcome {
    Done,
    Failed(DownloadFailure),
}

fn describe(models: ModelsHandle, id: &str) -> (String, Option<u64>) {
    models.find(id).map_or_else(
        || (id.to_owned(), None),
        |info| {
            let entry = ModelEntry::from_info(&info);
            (entry.label, Some(entry.size_bytes))
        },
    )
}

fn failure(models: ModelsHandle, kind: ModelFailureKind, message: String) -> DownloadFailure {
    let offline = !*models.online.peek();
    let kind = match kind {
        ModelFailureKind::Other if offline => ModelFailureKind::Network,
        ModelFailureKind::Network | ModelFailureKind::Integrity | ModelFailureKind::Other => kind,
    };
    DownloadFailure { kind, message }
}

async fn receive(models: ModelsHandle, id: &str) -> Outcome {
    let mut stream = match platform::get_platform().fetch_model(id).await {
        Ok(stream) => stream,
        Err(error) => {
            return Outcome::Failed(failure(models, ModelFailureKind::Other, error.to_string()));
        }
    };
    let mut meter = SpeedMeter::default();
    while let Some(item) = stream.next().await {
        match item {
            Ok(ModelOperationProgress::Starting { .. }) => {}
            Ok(ModelOperationProgress::Downloading {
                bytes_downloaded,
                total_bytes,
                ..
            }) => {
                let bytes_per_second = meter.record(now_ms(), bytes_downloaded);
                models.set_activity(
                    id,
                    ModelActivity::Downloading(DownloadProgress {
                        loaded_bytes: bytes_downloaded,
                        total_bytes,
                        bytes_per_second,
                    }),
                );
            }
            Ok(ModelOperationProgress::Installing { .. }) => {
                models.set_activity(id, ModelActivity::Verifying);
            }
            Ok(ModelOperationProgress::Completed { .. }) => return Outcome::Done,
            Ok(ModelOperationProgress::Failed {
                error_message,
                kind,
                ..
            }) => return Outcome::Failed(failure(models, kind, error_message)),
            Err(error) => {
                return Outcome::Failed(failure(
                    models,
                    ModelFailureKind::Other,
                    error.to_string(),
                ));
            }
        }
    }
    Outcome::Failed(failure(
        models,
        ModelFailureKind::Other,
        STOPPED_EARLY.to_owned(),
    ))
}

/// Downloads `id` and reports the end with a toast; with `replace` the stored files go first.
pub(super) async fn download(models: ModelsHandle, id: String, replace: bool) {
    let (label, _) = describe(models, &id);
    if replace && let Err(error) = platform::get_platform().remove_model(&id).await {
        let failed = failure(models, ModelFailureKind::Other, error.to_string());
        models.fail(&id, failed);
        models.toasts.error(
            format!("Couldn't replace {label}"),
            "The stored files could not be removed.",
        );
        return;
    }

    match receive(models, &id).await {
        Outcome::Done => {
            models.clear_activity(&id);
            models.refresh();
            models
                .toasts
                .success(format!("{label} is ready"), "Downloaded and checked.");
        }
        Outcome::Failed(failed) => {
            let title = match failed.kind {
                ModelFailureKind::Network => format!("{label} didn't finish downloading"),
                ModelFailureKind::Integrity => format!("{label} failed its checksum"),
                ModelFailureKind::Other => format!("Couldn't download {label}"),
            };
            let retry_id = id.clone();
            let toast = Toast::new(ToastKind::Error, title)
                .with_detail(failed.message.clone())
                .with_action("Retry", move || models.download(&retry_id));
            models.fail(&id, failed);
            models.refresh();
            models.toasts.show(toast);
        }
    }
}

/// Removes whatever a cancelled download left stored.
pub(super) async fn discard(models: ModelsHandle, id: String) {
    let (label, _) = describe(models, &id);
    match platform::get_platform().remove_model(&id).await {
        Ok(()) => {
            models.refresh();
            models
                .toasts
                .info("Download cancelled", "Partial files were removed.");
        }
        Err(error) => {
            models
                .toasts
                .error(format!("Couldn't clean up {label}"), error.to_string());
        }
    }
}

/// Checks the stored files of `id` and reports what the check found.
pub(super) async fn verify(models: ModelsHandle, id: String) {
    let (label, _) = describe(models, &id);
    match platform::get_platform().verify_model(&id).await {
        Ok(ModelIntegrity::Verified) => {
            models.clear_activity(&id);
            models.toasts.success(
                "Files check out",
                format!("{label} matches its published checksums."),
            );
        }
        Ok(ModelIntegrity::Damaged { problem, file }) => {
            models.set_activity(&id, ModelActivity::Damaged(Damage { problem, file }));
            models.refresh();
            models.toasts.error(
                format!("{label} failed its check"),
                "Download it again to repair it.",
            );
        }
        Err(error) => {
            models.clear_activity(&id);
            models
                .toasts
                .error(format!("Couldn't check {label}"), error.to_string());
        }
    }
}

/// Deletes the stored files of `id`.
pub(super) async fn remove(models: ModelsHandle, id: String) {
    let (label, size) = describe(models, &id);
    match platform::get_platform().remove_model(&id).await {
        Ok(()) => {
            models.clear_activity(&id);
            models.refresh();
            let freed = size.map_or_else(String::new, |size| format!("{} freed.", ByteSize(size)));
            models.toasts.info(format!("{label} deleted"), freed);
        }
        Err(error) => {
            models
                .toasts
                .error(format!("Couldn't delete {label}"), error.to_string());
        }
    }
}
