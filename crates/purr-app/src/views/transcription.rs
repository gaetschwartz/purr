use crate::{
    components::{Card, TranscriptionDisplay},
    main_app::{Route, UploadNames},
    platform::{self, TranscriptionRequest, TranscriptionStatus},
    transcript::Transcript,
    views::NavBarState,
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
    let mut navbar_state = use_context::<NavBarState>();
    let names = use_context::<UploadNames>();
    let mut status = use_signal(|| None::<TranscriptionStatus>);
    let mut transcript = use_signal(Transcript::default);
    let mut language = use_signal(|| None::<String>);
    let mut attempt = use_signal(|| 0u32);
    let mut cancelled = use_signal(|| false);

    use_effect(move || {
        navbar_state.title.set("Transcription".to_owned());
        navbar_state.show_back.set(true);
        navbar_state.top_right.set(None);
    });

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

    rsx! {
        document::Title { "Purr - Transcription" }
        div { class: "main-content",
            div { class: "content-container",
                if cancelled() {
                    div { class: "text-center py-8 space-y-4",
                        h2 { class: "text-lg font-semibold text-gray-900 dark:text-gray-100", "Transcription cancelled" }
                        div { class: "flex justify-center gap-3",
                            button {
                                class: "px-4 py-2 rounded-lg bg-teal-500 text-white hover:bg-teal-600 focus:outline-none focus:ring-2 focus:ring-teal-500",
                                onclick: move |_| {
                                    cancelled.set(false);
                                    attempt += 1;
                                },
                                "Start again"
                            }
                            button {
                                class: "px-4 py-2 rounded-lg bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-100 hover:bg-gray-200 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-teal-500",
                                onclick: move |_| {
                                    navigator.push(Route::Home {});
                                },
                                "Choose another file"
                            }
                        }
                    }
                } else {
                    Card {
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
            status.set(Some(TranscriptionStatus::Error {
                context: "Transcription".to_owned(),
                error_message: e.to_string(),
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
