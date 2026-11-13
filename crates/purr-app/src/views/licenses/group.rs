use super::catalog::{Licence, Notice};
use dioxus::prelude::*;

const FONT_LICENCE: &str = include_str!("../../../assets/fonts/OFL.txt");

#[component]
pub fn LicenceGroup(licence: Licence, query: String) -> Element {
    let count = licence.crate_count();
    let notices = licence
        .texts
        .iter()
        .filter(|notice| !notice.matching(&query).is_empty())
        .cloned()
        .collect::<Vec<_>>();
    rsx! {
        section {
            class: "card surface",
            style: "display:flex;flex-direction:column;gap:14px",
            aria_label: "{licence.name}",
            h2 { "{licence.name}" }
            p { class: "foot", "{count} crates" }
            for (index , notice) in notices.into_iter().enumerate() {
                NoticeEntry { key: "{index}", notice, query: query.clone() }
            }
        }
    }
}

#[component]
fn NoticeEntry(notice: Notice, query: String) -> Element {
    let crates = notice
        .matching(&query)
        .iter()
        .map(|package| format!("{} {}", package.name, package.version))
        .collect::<Vec<_>>()
        .join(", ");
    rsx! {
        div {
            p { class: "foot", "{crates}" }
            details { class: "foot",
                summary { "Licence text" }
                pre { class: "preview", "{notice.text}" }
            }
        }
    }
}

#[component]
pub fn FontGroup() -> Element {
    rsx! {
        section {
            class: "card surface",
            style: "display:flex;flex-direction:column;gap:14px",
            aria_label: "Inter font",
            h2 { "SIL Open Font License 1.1" }
            div {
                p { class: "foot", "Inter, the typeface Purr is set in." }
                details { class: "foot",
                    summary { "Licence text" }
                    pre { class: "preview", "{FONT_LICENCE}" }
                }
            }
        }
    }
}
