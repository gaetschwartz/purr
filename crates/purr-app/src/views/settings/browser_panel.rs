use super::{system_list::SystemList, system_rows::browser_rows};
use crate::{hooks::use_models::use_models, utils::dom_id::use_dom_id};
use dioxus::prelude::*;

/// What this browser offers for transcription: the WebGPU adapter it found.
#[component]
pub fn BrowserPanel() -> Element {
    let models = use_models();
    let heading = use_dom_id("browser");
    let rows = browser_rows(&models.gpu.read());

    rsx! {
        section { aria_labelledby: heading.clone(),
            h2 { class: "group-h", id: heading, "This browser" }
            SystemList { label: "WebGPU in this browser", rows }
        }
    }
}
