use super::{Card, CardVariant, Icon, IconType, StatusBadge, StatusType};
use dioxus::prelude::*;
use purr_common::platform::TranscriptionStatus;

/// Props for the TranscriptionDisplay component
#[derive(Props, Clone, PartialEq)]
pub struct TranscriptionDisplayProps {
    /// Current transcription status
    pub status: Option<TranscriptionStatus>,
    /// Current transcription text
    pub text: String,
    /// File name being transcribed
    pub file_name: String,
}

/// Component for displaying transcription status and results
#[component]
pub fn TranscriptionDisplay(props: TranscriptionDisplayProps) -> Element {
    match &props.status {
        Some(TranscriptionStatus::Starting) => rsx! {
            div { class: "text-center py-16",
                LoadingSpinner {}
                h3 { class: "text-xl font-semibold text-gray-900 mb-3",
                    "Initializing transcription..."
                }
                p { class: "text-gray-600 text-lg", "Setting up transcription for: {props.file_name}" }
            }
        },

        Some(TranscriptionStatus::ProcessingAudio) => rsx! {
            div { class: "text-center py-16",
                LoadingSpinner {}
                h3 { class: "text-xl font-semibold text-gray-900 mb-3", "Processing audio file..." }
                p { class: "text-gray-600 text-lg", "Preparing audio for transcription" }
            }
        },

        Some(TranscriptionStatus::InProgress { chunk_index, .. }) => rsx! {
            div { class: "text-center py-16",
                PulsingIndicator {}
                h3 { class: "text-xl font-semibold text-gray-900 mb-3", "Transcribing audio..." }

                StatusBadge {
                    text: format!("Processing chunk {}", chunk_index + 1),
                    status: StatusType::Processing,
                    class: "mb-8",
                }

                // Show live transcription results
                if !props.text.is_empty() {
                    Card {
                        variant: CardVariant::Info,
                        class: "mt-8 text-left shadow-sm",

                        h4 { class: "text-lg font-semibold text-blue-900 mb-4 flex items-center",
                            Icon {
                                icon_type: IconType::LiveTranscription,
                                class: "w-5 h-5 mr-2 text-blue-600",
                            }
                            "Live Transcription"
                        }
                        p { class: "text-gray-800 whitespace-pre-wrap leading-relaxed text-lg",
                            "{props.text}"
                        }
                    }
                }
            }
        },

        Some(TranscriptionStatus::Completed {
            processing_time,
            audio_duration,
            word_count,
        }) => rsx! {
            div { class: "text-center py-16",
                SuccessIcon {}
                h3 { class: "text-2xl font-bold text-green-700 mb-8", "Transcription Complete!" }

                // Statistics
                TranscriptionStats {
                    processing_time: *processing_time,
                    audio_duration: *audio_duration,
                    word_count: *word_count,
                }

                // Final transcription results
                Card {
                    variant: CardVariant::Success,
                    class: "mt-8 text-left shadow-lg",

                    h4 { class: "text-xl font-bold text-green-900 mb-6 flex items-center",
                        Icon {
                            icon_type: IconType::Check,
                            class: "w-6 h-6 mr-3 text-green-600",
                        }
                        "Final Transcription"
                    }
                    p { class: "text-gray-800 whitespace-pre-wrap leading-relaxed text-lg",
                        "{props.text}"
                    }
                }
            }
        },

        Some(TranscriptionStatus::Error { message }) => rsx! {
            div { class: "text-center py-16",
                ErrorIcon {}
                h3 { class: "text-xl font-semibold text-red-800 mb-4", "Transcription Failed" }

                Card {
                    variant: CardVariant::Error,
                    class: "mb-8 max-w-md mx-auto",

                    p { class: "text-red-700 text-sm", "{message}" }
                }
            }
        },

        Some(TranscriptionStatus::InitFailed { message }) => rsx! {
            div { class: "text-center py-16",
                ErrorIcon {}
                h3 { class: "text-xl font-semibold text-red-800 mb-4", "Initialization Failed" }

                Card {
                    variant: CardVariant::Error,
                    class: "mb-8 max-w-md mx-auto",

                    p { class: "text-red-700 text-sm", "{message}" }
                }
            }
        },

        None => rsx! {
            div { class: "text-center py-16",
                LoadingSpinner {}
                h3 { class: "text-xl font-semibold text-gray-900 mb-3", "Starting transcription..." }
                p { class: "text-gray-600 text-lg", "Preparing to transcribe: {props.file_name}" }
            }
        },
    }
}

/// Loading spinner component
#[component]
fn LoadingSpinner() -> Element {
    rsx! {
        div { class: "flex justify-center mb-6",
            div { class: "animate-spin rounded-full h-16 w-16 border-4 border-teal-200 border-t-teal-500" }
        }
    }
}

/// Pulsing indicator for in-progress state
#[component]
fn PulsingIndicator() -> Element {
    rsx! {
        div { class: "flex justify-center mb-6",
            div { class: "animate-pulse rounded-full h-16 w-16 bg-gradient-to-r from-teal-400 to-teal-600 shadow-lg" }
        }
    }
}

/// Success icon component
#[component]
fn SuccessIcon() -> Element {
    rsx! {
        div { class: "flex justify-center mb-6",
            div { class: "w-20 h-20 bg-green-100 rounded-full flex items-center justify-center",
                Icon {
                    icon_type: IconType::Check,
                    class: "w-10 h-10 text-green-600",
                }
            }
        }
    }
}

/// Error icon component
#[component]
fn ErrorIcon() -> Element {
    rsx! {
        div { class: "flex justify-center mb-6",
            div { class: "w-20 h-20 bg-red-100 rounded-full flex items-center justify-center",
                Icon {
                    icon_type: IconType::Error,
                    class: "w-10 h-10 text-red-600",
                }
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
    rsx! {
        div { class: "grid grid-cols-2 md:grid-cols-4 gap-6 max-w-2xl mx-auto mb-10",
            StatCard {
                label: "Processing Time",
                value: format!("{:.1}s", props.processing_time),
            }
            StatCard {
                label: "Audio Duration",
                value: format!("{:.1}s", props.audio_duration),
            }
            StatCard { label: "Word Count", value: format!("{}", props.word_count) }
            StatCard {
                label: "Speed",
                value: format!("{:.1}x", props.audio_duration / props.processing_time as f32),
            }
        }
    }
}

/// Props for stat cards
#[derive(Props, Clone, PartialEq)]
struct StatCardProps {
    label: String,
    value: String,
}

/// Individual stat card component
#[component]
fn StatCard(props: StatCardProps) -> Element {
    rsx! {
        div { class: "bg-white p-4 rounded-lg shadow-sm border border-gray-200 text-center",
            p { class: "text-sm font-medium text-gray-600 mb-1", "{props.label}" }
            p { class: "text-2xl font-bold text-teal-600", "{props.value}" }
        }
    }
}
