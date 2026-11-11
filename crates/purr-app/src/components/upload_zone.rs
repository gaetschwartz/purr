use super::{Icon, IconType};
use dioxus::prelude::*;

/// Upload state for the upload zone
#[derive(Clone, PartialEq, Eq)]
pub enum UploadState {
    Idle,
    Dragging,
    Reading { file_name: String },
    Failed { file_name: String, message: String },
}

/// Props for the `UploadZone` component
#[derive(Props, Clone, PartialEq)]
pub struct UploadZoneProps {
    /// Current upload state
    pub state: UploadState,
    /// Handler for drag over events
    pub ondragover: EventHandler<DragEvent>,
    /// Handler for drag leave events
    pub ondragleave: EventHandler<DragEvent>,
    /// Handler for drop events
    pub ondrop: EventHandler<DragEvent>,
    /// Handler for file input change
    pub onchange: EventHandler<FormEvent>,
}

/// Reusable upload zone component
#[component]
pub fn UploadZone(props: UploadZoneProps) -> Element {
    let border_class = match props.state {
        UploadState::Dragging => "border-teal-500 bg-teal-50/50 dark:bg-teal-950/40",
        UploadState::Failed { .. } => "border-red-400",
        UploadState::Idle | UploadState::Reading { .. } => {
            "border-gray-300 dark:border-gray-600 hover:border-teal-400"
        }
    };
    let reading = matches!(props.state, UploadState::Reading { .. });

    rsx! {
        div {
            class: "centered-content w-full px-4",
            ondragover: move |evt| props.ondragover.call(evt),
            ondragleave: move |evt| props.ondragleave.call(evt),
            ondrop: move |evt| props.ondrop.call(evt),

            label {
                class: "flex flex-col items-center justify-center w-full max-w-lg min-h-80 p-8 border-2 border-dashed {border_class} rounded-xl cursor-pointer transition-colors duration-300 ease-in-out bg-white dark:bg-gray-800 shadow-lg focus-within:ring-2 focus-within:ring-teal-500 focus-within:ring-offset-2",

                input {
                    r#type: "file",
                    class: "sr-only",
                    id: "file-input",
                    name: "audio-file",
                    multiple: false,
                    disabled: reading,
                    accept: "audio/*, .mp3, .wav, .aac, .flac, .ogg, .opus, .webm, .m4a, .mp4",
                    onchange: move |evt| props.onchange.call(evt),
                }

                match &props.state {
                    UploadState::Reading { file_name } => rsx! {
                        div { class: "flex flex-col items-center text-center", role: "status",
                            div { class: "animate-spin rounded-full h-12 w-12 border-b-2 border-teal-500 mb-6" }
                            h2 { class: "text-lg font-medium text-gray-900 dark:text-gray-100 mb-2", "Reading file..." }
                            p { class: "text-gray-600 dark:text-gray-300 text-sm break-all max-w-full", "{file_name}" }
                        }
                    },
                    UploadState::Failed { file_name, message } => rsx! {
                        div { class: "flex flex-col items-center text-center", role: "alert",
                            div { class: "w-16 h-16 bg-red-100 dark:bg-red-950 rounded-full flex items-center justify-center mb-4",
                                Icon { icon_type: IconType::Error, class: "w-8 h-8 text-red-600" }
                            }
                            h2 { class: "text-lg font-medium text-red-700 dark:text-red-400 mb-2", "This file cannot be used" }
                            p { class: "text-gray-600 dark:text-gray-300 text-sm break-all max-w-full mb-1", "{file_name}" }
                            p { class: "text-red-700 dark:text-red-400 text-sm mb-4", "{message}" }
                            p { class: "text-gray-600 dark:text-gray-300", "Click to choose another file or drag and drop it here" }
                        }
                    },
                    UploadState::Idle | UploadState::Dragging => rsx! {
                        div { class: "flex flex-col items-center text-center",
                            div { class: "w-16 h-16 bg-teal-100 dark:bg-teal-900 rounded-full flex items-center justify-center mb-6",
                                Icon { icon_type: IconType::Upload, class: "w-8 h-8 text-teal-600" }
                            }
                            h2 { class: "text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2", "Upload Audio File" }
                            p { class: "text-gray-600 dark:text-gray-300 mb-1",
                                span { class: "font-medium", "Click to upload" }
                                " or drag and drop"
                            }
                            p { class: "text-sm text-gray-500 dark:text-gray-400",
                                "MP3, WAV, FLAC, OGG, M4A, AAC and WebM, up to 100 MB. Decoded by your browser."
                            }
                        }
                    },
                }
            }
        }
    }
}
