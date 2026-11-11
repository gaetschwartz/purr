use super::{
    banner::{Banner, BannerTone},
    icon::{Icon, IconName, IconSize},
    spinner::Spinner,
};
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

/// The drop zone of the Transcribe view: a glass well that also opens the file chooser.
#[component]
pub fn UploadZone(props: UploadZoneProps) -> Element {
    let class = match props.state {
        UploadState::Dragging => "drop glass over",
        UploadState::Failed { .. } => "drop glass bad",
        UploadState::Idle | UploadState::Reading { .. } => "drop glass",
    };
    let reading = matches!(props.state, UploadState::Reading { .. });

    rsx! {
        if let UploadState::Failed { file_name, message } = &props.state {
            Banner {
                tone: BannerTone::Bad,
                title: "This file cannot be used",
                role: "alert",
                "{file_name}: {message} Choose another file or drop it below."
            }
        }
        label {
            class,
            ondragover: move |evt| props.ondragover.call(evt),
            ondragleave: move |evt| props.ondragleave.call(evt),
            ondrop: move |evt| props.ondrop.call(evt),
            span { class: "well" }
            input {
                r#type: "file",
                class: "sr",
                id: "file-input",
                name: "audio-file",
                multiple: false,
                disabled: reading,
                aria_describedby: "drop-formats",
                accept: "audio/*, .mp3, .wav, .aac, .flac, .ogg, .opus, .webm, .m4a, .mp4",
                onchange: move |evt| props.onchange.call(evt),
            }
            if let UploadState::Reading { file_name } = &props.state {
                span { class: "halo",
                    Spinner { large: true }
                }
                h2 { role: "status", "Reading file…" }
                p { "{file_name}" }
            } else {
                span { class: "halo",
                    Icon { name: IconName::Upload, size: IconSize::Large }
                }
                h2 { "Drop an audio file" }
                p {
                    "or "
                    span { class: "link", "choose a file" }
                }
                span { class: "formats", id: "drop-formats",
                    "MP3, WAV, FLAC, OGG, M4A, AAC or WebM, up to 100 MB"
                }
            }
        }
    }
}
