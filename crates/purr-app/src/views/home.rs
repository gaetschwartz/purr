use crate::{
    components::{
        icon::{Icon, IconName, IconSize},
        page::Page,
    },
    platform::IS_WEB,
    views::drag_drop::DragDropZone,
};
use dioxus::prelude::*;

/// The Transcribe view: drop or choose an audio file to transcribe it.
#[component]
pub fn Home() -> Element {
    let (subtitle, privacy) = if IS_WEB {
        (
            "Turn a recording into text. Nothing leaves this browser.",
            "Processed on this device with WebGPU. Audio is never uploaded.",
        )
    } else {
        (
            "Turn a recording into text. Nothing leaves this computer.",
            "Processed on this device. Audio is never uploaded.",
        )
    };
    rsx! {
        Page { title: "Transcribe", subtitle,
            DragDropZone {}
            p { class: "privacy-line",
                Icon { name: IconName::Lock, size: IconSize::Small }
                "{privacy}"
            }
        }
    }
}
