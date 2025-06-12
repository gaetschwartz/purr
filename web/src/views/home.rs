use dioxus::prelude::*;
use ui::{TranscriptionView, Hero};

#[component]
pub fn Home() -> Element {
    rsx! {
        Hero {}
        TranscriptionView {}
    }
}
