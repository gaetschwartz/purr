use dioxus::prelude::*;
use super::{Icon, IconType, Button, ButtonVariant, ProgressBar};

/// Upload state for the upload zone
#[derive(Clone, PartialEq)]
pub enum UploadState {
    Idle,
    Dragging,
    Uploading { progress: f32, file_name: String },
    Success { file_name: String },
}

/// Props for the UploadZone component
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
    /// Handler for upload another file button
    pub on_upload_another: EventHandler<MouseEvent>,
}

/// Reusable upload zone component
#[component]
pub fn UploadZone(props: UploadZoneProps) -> Element {
    let border_class = match props.state {
        UploadState::Dragging => "border-2 border-dashed border-teal-500 bg-teal-50/50",
        _ => "border-2 border-dashed border-gray-300 hover:border-teal-400 hover:bg-gray-50/50",
    };

    rsx! {
        div {
            class: "flex items-center justify-center w-full h-full min-h-screen bg-gradient-to-br from-gray-50 to-gray-100",
            
            div {
                class: "flex flex-col items-center justify-center w-full max-w-lg mx-auto h-96 {border_class} rounded-xl cursor-pointer transition-all duration-300 ease-in-out transform hover:scale-105 bg-white shadow-lg",
                ondragover: move |evt| props.ondragover.call(evt),
                ondragleave: move |evt| props.ondragleave.call(evt),
                ondrop: move |evt| props.ondrop.call(evt),

                // Hidden file input
                input {
                    r#type: "file",
                    class: "hidden",
                    id: "file-input",
                    name: "audio-file",
                    multiple: false,
                    accept: "audio/mpeg, audio/wav, audio/x-wav, audio/wave, audio/aac, audio/flac, audio/ogg, audio/webm, audio/mp4, audio/x-m4a, .mp3, .wav, .aac, .flac, .ogg, .webm, .m4a, .mp4",
                    onchange: move |evt| props.onchange.call(evt),
                }

                // Click handler that triggers the file input
                label {
                    r#for: "file-input",
                    class: "flex flex-col items-center justify-center w-full h-full cursor-pointer p-8",

                    match &props.state {
                        UploadState::Uploading { progress, file_name: _ } => rsx! {
                            div { class: "flex flex-col items-center text-center",
                                div { class: "animate-spin rounded-full h-12 w-12 border-b-2 border-teal-500 mb-6" }
                                h3 { class: "text-lg font-medium text-gray-900 mb-2", "Uploading file..." }

                                if *progress > 0.0 {
                                    ProgressBar {
                                        value: *progress,
                                        label: Some(format!("{} KB uploaded", (*progress * 10.0) as i32)),
                                        show_percentage: false,
                                    }
                                }
                            }
                        },

                        UploadState::Success { file_name } => rsx! {
                            div { class: "flex flex-col items-center text-center",
                                div { class: "w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-4",
                                    Icon {
                                        icon_type: IconType::Check,
                                        class: "w-8 h-8 text-green-600"
                                    }
                                }
                                h3 { class: "text-lg font-medium text-green-700 mb-2",
                                    "File uploaded successfully!"
                                }
                                p { class: "text-gray-600 text-sm break-all break-words max-w-full mb-4",
                                    "{file_name}"
                                }

                                Button {
                                    variant: ButtonVariant::Primary,
                                    onclick: move |evt: MouseEvent| {
                                        evt.prevent_default();
                                        evt.stop_propagation();
                                        props.on_upload_another.call(evt);
                                    },
                                    "Upload Another File"
                                }
                            }
                        },

                        _ => rsx! {
                            div { class: "flex flex-col items-center text-center",
                                div { class: "w-16 h-16 bg-teal-100 rounded-full flex items-center justify-center mb-6",
                                    Icon {
                                        icon_type: IconType::Upload,
                                        class: "w-8 h-8 text-teal-600"
                                    }
                                }
                                h2 { class: "text-xl font-semibold text-gray-900 mb-2", "Upload Audio File" }
                                p { class: "text-gray-600 mb-1",
                                    span { class: "font-medium", "Click to upload" }
                                    " or drag and drop"
                                }
                                p { class: "text-sm text-gray-500", "Audio files supported (MP3, WAV, AAC, FLAC, OGG, M4A)" }
                            }
                        }
                    }
                }
            }
        }
    }
}

