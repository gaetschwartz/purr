use dioxus::prelude::*;
use ui::TranscriptionView;

#[component]
pub fn Transcription() -> Element {
    rsx! {
        TranscriptionView {}
    }
}