use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonSize, ButtonVariant},
        icon::IconName,
        page::Page,
        transcription_display::TranscriptionDisplay,
    },
    main_app::{Route, UploadNames},
    platform::{self, TranscriptionRequest, TranscriptionStatus},
    transcript::Transcript,
};
use dioxus::prelude::*;
use futures::StreamExt as _;
use purr_common::platform::{FileId, FileSource, Platform as _};
use tracing::{error, info};

/// The Transcription page component that displays the transcription functionality
/// for an uploaded audio file
#[component]
pub fn Transcription(file: FileId) -> Element {
    let navigator = use_navigator();
    let names = use_context::<UploadNames>();
    let mut status = use_signal(|| None::<TranscriptionStatus>);
    let mut transcript = use_signal(Transcript::default);
    let mut language = use_signal(|| None::<String>);
    let mut attempt = use_signal(|| 0u32);
    let mut cancelled = use_signal(|| false);

    let file_for_run = file.clone();
    use_resource(move || {
        let file_id = file_for_run.clone();
        async move {
            attempt();
            if cancelled() {
                return;
            }
            status.set(None);
            transcript.set(Transcript::default());
            language.set(None);
            run_transcription(file_id, status, transcript, language).await;
        }
    });

    let file_for_cleanup = file.clone();
    use_drop(move || {
        let file_id = file_for_cleanup;
        dioxus::core::spawn_forever(async move {
            if let Err(e) = platform::get_platform().cleanup(&file_id).await {
                error!("Could not remove the uploaded file: {e}");
            }
        });
    });

    let file_name = names.0.read().get(&file).cloned();
    let title = file_name
        .clone()
        .unwrap_or_else(|| "Transcription".to_owned());

    rsx! {
        Page {
            title,
            actions: rsx! {
                Button {
                    variant: ButtonVariant::Plain,
                    size: ButtonSize::Small,
                    icon: IconName::Back,
                    onclick: move |_| {
                        navigator.push(Route::Home {});
                    },
                    "New transcription"
                }
            },
            if cancelled() {
                Banner {
                    tone: BannerTone::Info,
                    title: "Transcription cancelled",
                    role: "status",
                    actions: rsx! {
                        Button {
                            variant: ButtonVariant::Primary,
                            size: ButtonSize::Small,
                            icon: IconName::Retry,
                            onclick: move |_| {
                                cancelled.set(false);
                                attempt += 1;
                            },
                            "Start again"
                        }
                        Button {
                            size: ButtonSize::Small,
                            onclick: move |_| {
                                navigator.push(Route::Home {});
                            },
                            "Choose another file"
                        }
                    },
                    "Nothing was saved."
                }
            } else {
                section { class: "card glass", aria_label: "Transcription",
                    TranscriptionDisplay {
                        status: status(),
                        transcript: transcript(),
                        language: language(),
                        file_name,
                        on_cancel: move |()| {
                            cancelled.set(true);
                            status.set(None);
                        },
                        on_retry: move |()| attempt += 1,
                        on_new: move |()| {
                            navigator.push(Route::Home {});
                        },
                    }
                }
            }
        }
    }
}

/// Streams the transcription of `file_id` into the signals until it ends or the caller drops it.
async fn run_transcription(
    file_id: FileId,
    mut status: Signal<Option<TranscriptionStatus>>,
    mut transcript: Signal<Transcript>,
    mut language: Signal<Option<String>>,
) {
    info!("Starting transcription for file ID: {file_id}");
    let request = TranscriptionRequest {
        language: None,
        translate: false,
        file: FileSource::Uploaded(file_id.clone()),
    };

    let mut stream = match platform::get_platform().transcribe(request).await {
        Ok(stream) => stream,
        Err(e) => {
            error!("Failed to start transcription: {e}");
            let error_message = match &e {
                platform::PlatformError::ModelNotFound { model_id } => format!(
                    "The selected model \"{model_id}\" is not available in this app. Choose another one in Settings."
                ),
                platform::PlatformError::FileProcessing { .. }
                | platform::PlatformError::Transcription { .. }
                | platform::PlatformError::AudioProcessing { .. }
                | platform::PlatformError::Initialization { .. }
                | platform::PlatformError::Io { .. }
                | platform::PlatformError::Unsupported { .. }
                | platform::PlatformError::ModelManagement { .. }
                | platform::PlatformError::ModelDownloadFailed { .. }
                | platform::PlatformError::ModelInstallationFailed { .. }
                | platform::PlatformError::InvalidModelMetadata { .. }
                | platform::PlatformError::ModelVerificationFailed { .. }
                | platform::PlatformError::StorageQuotaExceeded { .. }
                | platform::PlatformError::DeviceDetection { .. }
                | platform::PlatformError::SettingsPersistence { .. }
                | platform::PlatformError::UnsupportedPlatform(_) => e.to_string(),
            };
            status.set(Some(TranscriptionStatus::Error {
                context: "Transcription".to_owned(),
                error_message,
            }));
            return;
        }
    };

    while let Some(update) = stream.next().await {
        match update {
            Ok(next) => {
                transcript.write().apply(&next);
                if let TranscriptionStatus::LanguageDetected { language: code } = &next {
                    language.set(Some(code.clone()));
                    continue;
                }
                let finished = matches!(
                    next,
                    TranscriptionStatus::Completed { .. }
                        | TranscriptionStatus::Error { .. }
                        | TranscriptionStatus::InitFailed { .. }
                );
                status.set(Some(next));
                if finished {
                    info!("Transcription of {file_id} ended");
                    return;
                }
            }
            Err(e) => {
                error!("Transcription stream failed: {e}");
                status.set(Some(TranscriptionStatus::Error {
                    context: "Transcription".to_owned(),
                    error_message: e.to_string(),
                }));
                return;
            }
        }
    }
}
