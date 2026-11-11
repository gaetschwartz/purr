use super::{
    Button, ButtonVariant, Card, CardVariant, Icon, IconType, model_loading::ModelLoading,
    transcript_panel::TranscriptPanel,
};
use crate::{
    transcript::Transcript,
    utils::{file_kind::file_stem, language_label::language_label},
};
use dioxus::prelude::*;
use purr_common::platform::TranscriptionStatus;

/// Props for the `TranscriptionDisplay` component
#[derive(Props, Clone, PartialEq)]
pub struct TranscriptionDisplayProps {
    /// Current transcription status
    pub status: Option<TranscriptionStatus>,
    /// Segments reported so far
    pub transcript: Transcript,
    /// Language Whisper recognised, when it was asked to detect it
    pub language: Option<String>,
    /// Name of the audio file being transcribed
    pub file_name: Option<String>,
    /// Stops the transcription
    pub on_cancel: EventHandler<()>,
    /// Runs the transcription again
    pub on_retry: EventHandler<()>,
    /// Leaves for another file
    pub on_new: EventHandler<()>,
}

fn display_name(file_name: Option<&str>) -> String {
    file_name.unwrap_or("your audio file").to_owned()
}

fn export_stem(file_name: Option<&str>) -> String {
    file_name.map_or_else(|| "transcript".to_owned(), file_stem)
}

/// Component for displaying transcription status and results
#[component]
pub fn TranscriptionDisplay(props: TranscriptionDisplayProps) -> Element {
    let cancel = move |_| props.on_cancel.call(());
    let new_file = move |_| props.on_new.call(());

    match &props.status {
        None | Some(TranscriptionStatus::Starting) => rsx! {
            Waiting {
                title: "Starting transcription...",
                detail: format!("Preparing to transcribe {}", display_name(props.file_name.as_deref())),
                on_cancel: props.on_cancel,
            }
        },

        Some(TranscriptionStatus::ProcessingAudio) => rsx! {
            Waiting {
                title: "Decoding the audio...",
                detail: "Your browser is converting the file for the model",
                on_cancel: props.on_cancel,
            }
        },

        Some(TranscriptionStatus::LoadingModel {
            loaded_bytes,
            total_bytes,
        }) => rsx! {
            div { class: "text-center py-8",
                LoadingSpinner {}
                ModelLoading { loaded_bytes: *loaded_bytes, total_bytes: *total_bytes }
                div { class: "mt-6",
                    Button { variant: ButtonVariant::Secondary, onclick: cancel, "Cancel" }
                }
            }
        },

        Some(
            TranscriptionStatus::Transcribing
            | TranscriptionStatus::LanguageDetected { .. }
            | TranscriptionStatus::InProgress { .. },
        ) => rsx! {
            div { class: "space-y-6",
                div { class: "text-center",
                    PulsingIndicator {}
                    h3 { class: "text-lg font-semibold text-gray-900 dark:text-gray-100 mb-1", role: "status",
                        "Transcribing audio..."
                    }
                    p { class: "text-sm text-gray-600 dark:text-gray-400", {display_name(props.file_name.as_deref())} }
                    if let Some(code) = &props.language {
                        p { class: "text-sm text-gray-600 dark:text-gray-400", "Detected language: {language_label(code)}" }
                    }
                }
                Card { variant: CardVariant::Info, class: "text-left shadow-sm", padding: false,
                    div { class: "p-6",
                        h4 { class: "text-base font-semibold text-blue-900 dark:text-blue-200 mb-3 flex items-center",
                            Icon {
                                icon_type: IconType::LiveTranscription,
                                class: "w-4 h-4 mr-2 text-blue-600",
                            }
                            "Live Transcription"
                        }
                        TranscriptPanel {
                            transcript: props.transcript.clone(),
                            complete: false,
                            file_stem: export_stem(props.file_name.as_deref()),
                        }
                    }
                }
                div { class: "text-center",
                    Button { variant: ButtonVariant::Secondary, onclick: cancel, "Cancel" }
                }
            }
        },

        Some(TranscriptionStatus::Segments { .. }) => rsx! {
            Waiting {
                title: "Finishing...",
                detail: "Putting the transcript together",
                on_cancel: props.on_cancel,
            }
        },

        Some(TranscriptionStatus::Completed {
            processing_time,
            audio_duration,
            word_count,
        }) => rsx! {
            div { class: "space-y-6",
                div { class: "text-center",
                    SuccessIcon {}
                    h3 { class: "text-xl font-bold text-green-700 dark:text-green-400 mb-4", role: "status",
                        "Transcription Complete!"
                    }
                    if let Some(code) = &props.language {
                        p { class: "text-sm text-gray-600 dark:text-gray-400 mb-4", "Detected language: {language_label(code)}" }
                    }
                    TranscriptionStats {
                        processing_time: *processing_time,
                        audio_duration: *audio_duration,
                        word_count: *word_count,
                    }
                }
                Card { variant: CardVariant::Success, class: "text-left shadow-lg", padding: false,
                    div { class: "p-6",
                        h4 { class: "text-lg font-bold text-green-900 dark:text-green-200 mb-3 flex items-center",
                            Icon {
                                icon_type: IconType::Check,
                                class: "w-5 h-5 mr-2 text-green-600",
                            }
                            "Final Transcription"
                        }
                        TranscriptPanel {
                            transcript: props.transcript.clone(),
                            complete: true,
                            file_stem: export_stem(props.file_name.as_deref()),
                        }
                    }
                }
                div { class: "text-center",
                    Button { onclick: new_file, "Transcribe Another File" }
                }
            }
        },

        Some(TranscriptionStatus::Error { error_message, .. }) => rsx! {
            Failure {
                title: "Transcription Failed",
                message: error_message.clone(),
                hint: String::new(),
                on_retry: props.on_retry,
                on_new: props.on_new,
            }
        },

        Some(TranscriptionStatus::InitFailed {
            reason,
            error_details,
            ..
        }) => rsx! {
            Failure {
                title: "Could not start the transcription",
                message: reason.clone(),
                hint: error_details.clone().unwrap_or_default(),
                on_retry: props.on_retry,
                on_new: props.on_new,
            }
        },
    }
}

#[component]
fn Waiting(title: String, detail: String, on_cancel: EventHandler<()>) -> Element {
    rsx! {
        div { class: "text-center py-8",
            LoadingSpinner {}
            h3 { class: "text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2", role: "status", "{title}" }
            p { class: "text-gray-600 dark:text-gray-400 mb-6", "{detail}" }
            Button {
                variant: ButtonVariant::Secondary,
                onclick: move |_| on_cancel.call(()),
                "Cancel"
            }
        }
    }
}

#[component]
fn Failure(
    title: String,
    message: String,
    hint: String,
    on_retry: EventHandler<()>,
    on_new: EventHandler<()>,
) -> Element {
    rsx! {
        div { class: "text-center py-8", role: "alert",
            ErrorIcon {}
            h3 { class: "text-lg font-semibold text-red-800 dark:text-red-300 mb-3", "{title}" }
            Card { variant: CardVariant::Error, class: "mb-4 max-w-xl mx-auto",
                p { class: "text-red-700 dark:text-red-300 text-sm break-words", "{message}" }
                if !hint.is_empty() {
                    p { class: "text-red-700 dark:text-red-300 text-sm mt-3", "{hint}" }
                }
            }
            div { class: "flex justify-center gap-3",
                Button { onclick: move |_| on_retry.call(()), "Try Again" }
                Button {
                    variant: ButtonVariant::Secondary,
                    onclick: move |_| on_new.call(()),
                    "Try Another File"
                }
            }
        }
    }
}

/// Loading spinner component
#[component]
fn LoadingSpinner() -> Element {
    rsx! {
        div { class: "flex justify-center mb-4", aria_hidden: "true",
            div { class: "animate-spin rounded-full h-10 w-10 border-4 border-teal-200 border-t-teal-500" }
        }
    }
}

/// Pulsing indicator for in-progress state
#[component]
fn PulsingIndicator() -> Element {
    rsx! {
        div { class: "flex justify-center mb-4", aria_hidden: "true",
            div { class: "animate-pulse rounded-full h-10 w-10 bg-gradient-to-r from-teal-400 to-teal-600 shadow-lg" }
        }
    }
}

/// Success icon component
#[component]
fn SuccessIcon() -> Element {
    rsx! {
        div { class: "flex justify-center mb-4",
            div { class: "w-14 h-14 bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center",
                Icon { icon_type: IconType::Check, class: "w-7 h-7 text-green-600" }
            }
        }
    }
}

/// Error icon component
#[component]
fn ErrorIcon() -> Element {
    rsx! {
        div { class: "flex justify-center mb-4",
            div { class: "w-14 h-14 bg-red-100 dark:bg-red-950 rounded-full flex items-center justify-center",
                Icon { icon_type: IconType::Error, class: "w-7 h-7 text-red-600" }
            }
        }
    }
}

/// Props for transcription statistics
#[derive(Props, Clone, PartialEq)]
struct TranscriptionStatsProps {
    processing_time: f64,
    audio_duration: f32,
    word_count: usize,
}

/// Statistics display component
#[component]
fn TranscriptionStats(props: TranscriptionStatsProps) -> Element {
    let speed = if props.processing_time > 0.0 {
        format!(
            "{:.1}x",
            f64::from(props.audio_duration) / props.processing_time
        )
    } else {
        "n/a".to_owned()
    };
    rsx! {
        div { class: "grid grid-cols-2 md:grid-cols-4 gap-4 max-w-2xl mx-auto mb-6",
            StatCard { label: "Processing Time", value: format!("{:.1}s", props.processing_time) }
            StatCard { label: "Audio Duration", value: format!("{:.1}s", props.audio_duration) }
            StatCard { label: "Word Count", value: format!("{}", props.word_count) }
            StatCard { label: "Speed", value: speed }
        }
    }
}

/// Props for stat cards
#[derive(Props, Clone, PartialEq)]
struct StatCardProps {
    label: &'static str,
    value: String,
}

/// Individual stat card component
#[component]
fn StatCard(props: StatCardProps) -> Element {
    rsx! {
        div { class: "bg-white dark:bg-gray-800 p-3 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 text-center",
            p { class: "text-xs font-medium text-gray-600 dark:text-gray-400 mb-1", "{props.label}" }
            p { class: "text-lg font-bold text-teal-600 dark:text-teal-400", "{props.value}" }
        }
    }
}
