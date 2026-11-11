use super::icon::{Icon, IconName};
use dioxus::prelude::*;

/// The Purr logo: the cat-head waveform on an indigo gradient tile, decorative.
#[component]
pub fn BrandMark(#[props(default)] large: bool) -> Element {
    let class = if large { "brand-mark lg" } else { "brand-mark" };
    rsx! {
        span { class, aria_hidden: "true",
            Icon { name: IconName::Cat }
        }
    }
}
