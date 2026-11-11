use super::{
    banner::{Banner, BannerTone},
    button::{Button, ButtonVariant},
    chip::{Chip, ChipTone},
    icon::IconName,
    model_loading::ModelLoading,
    spinner::Spinner,
    transcript_panel::TranscriptPanel,
};
use crate::{
    transcribe::transcript::Transcript,
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
                title: "Starting transcription…",
                detail: format!("Preparing to transcribe {}", display_name(props.file_name.as_deref())),
                on_cancel: props.on_cancel,
            }
        },

        Some(TranscriptionStatus::ProcessingAudio) => rsx! {
            Waiting {
                title: "Decoding the audio…",
                detail: "Converting the file for the model",
                on_cancel: props.on_cancel,
            }
        },

        Some(TranscriptionStatus::LoadingModel {
            loaded_bytes,
            total_bytes,
        }) => rsx! {
            div { class: "stack",
                ModelLoading { loaded_bytes: *loaded_bytes, total_bytes: *total_bytes }
                div { class: "row",
                    Button { icon: IconName::Stop, onclick: cancel, "Cancel" }
                }
            }
        },

        Some(
            TranscriptionStatus::Transcribing
            | TranscriptionStatus::LanguageDetected { .. }
            | TranscriptionStatus::InProgress { .. },
        ) => rsx! {
            div { class: "stack",
                div { class: "row",
                    Spinner {}
                    h2 { class: "t3 grow", role: "status", "Transcribing…" }
                    Chip { tone: ChipTone::Accent, icon: IconName::Wave, "Streaming" }
                }
                if let Some(code) = &props.language {
                    p { class: "muted", "Detected language: {language_label(code)}" }
                }
                TranscriptPanel {
                    transcript: props.transcript.clone(),
                    complete: false,
                    file_stem: export_stem(props.file_name.as_deref()),
                }
                div { class: "row",
                    Button { icon: IconName::Stop, onclick: cancel, "Cancel" }
                }
            }
        },

        Some(TranscriptionStatus::Segments { .. }) => rsx! {
            Waiting {
                title: "Finishing…",
                detail: "Putting the transcript together",
                on_cancel: props.on_cancel,
            }
        },

        Some(TranscriptionStatus::Completed {
            processing_time,
            audio_duration,
            word_count,
        }) => rsx! {
            div { class: "stack",
                Banner {
                    tone: BannerTone::Ok,
                    title: "Transcription finished",
                    role: "status",
                    if let Some(code) = &props.language {
                        "Detected language: {language_label(code)}"
                    }
                }
                TranscriptionStats {
                    processing_time: *processing_time,
                    audio_duration: *audio_duration,
                    word_count: *word_count,
                }
                TranscriptPanel {
                    transcript: props.transcript.clone(),
                    complete: true,
                    file_stem: export_stem(props.file_name.as_deref()),
                }
                div { class: "row",
                    Button {
                        variant: ButtonVariant::Primary,
                        icon: IconName::Plus,
                        onclick: new_file,
                        "Transcribe another file"
                    }
                }
            }
        },

        Some(TranscriptionStatus::Error { error_message, .. }) => rsx! {
            Failure {
                title: "Transcription failed",
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
        div { class: "stack",
            div { class: "row",
                Spinner {}
                h2 { class: "t3", role: "status", "{title}" }
            }
            p { class: "muted", style: "margin:0", "{detail}" }
            div { class: "row",
                Button { icon: IconName::Stop, onclick: move |_| on_cancel.call(()), "Cancel" }
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
        Banner {
            tone: BannerTone::Bad,
            title,
            role: "alert",
            actions: rsx! {
                Button {
                    variant: ButtonVariant::Primary,
                    icon: IconName::Retry,
                    onclick: move |_| on_retry.call(()),
                    "Try again"
                }
                Button { onclick: move |_| on_new.call(()), "Try another file" }
            },
            "{message}"
            if !hint.is_empty() {
                " {hint}"
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
            "{:.1}×",
            f64::from(props.audio_duration) / props.processing_time
        )
    } else {
        "n/a".to_owned()
    };
    rsx! {
        div { class: "stats",
            Stat { label: "Took", value: format!("{:.1} s", props.processing_time) }
            Stat { label: "Audio", value: format!("{:.1} s", props.audio_duration) }
            Stat { label: "Words", value: format!("{}", props.word_count) }
            Stat { label: "Speed", value: speed }
        }
    }
}

#[component]
fn Stat(label: &'static str, value: String) -> Element {
    rsx! {
        div { class: "stat",
            div { class: "k", "{label}" }
            div { class: "v", "{value}" }
        }
    }
}
