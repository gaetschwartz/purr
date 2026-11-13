use crate::{
    components::button::{Button, ButtonSize, ButtonVariant},
    history::{
        store::use_history,
        wording::{card_meta, preview_text},
    },
    main_app::Route,
    utils::{clock::unix_ms, local_time::when_text},
};
use dioxus::prelude::*;

/// How many cards the Recent row shows.
const CARDS: usize = 3;

/// The latest saved transcriptions as cards under the run bar; nothing while there are none.
#[component]
pub fn Recent() -> Element {
    let store = use_history();
    let navigator = use_navigator();
    let Some(listing) = store.listing() else {
        return rsx! {};
    };
    if listing.summaries.is_empty() {
        return rsx! {};
    }
    let now = unix_ms();
    rsx! {
        section { aria_labelledby: "recent-title",
            div { class: "row", style: "justify-content:space-between;margin-top:8px",
                h2 { class: "t2", id: "recent-title", "Recent" }
                Button {
                    variant: ButtonVariant::Plain,
                    size: ButtonSize::Small,
                    id: "see-all",
                    onclick: move |_| {
                        navigator.push(Route::History {});
                    },
                    "See all"
                }
            }
            div { class: "recent",
                for summary in listing.summaries.iter().take(CARDS) {
                    {
                        let id = summary.id.clone();
                        rsx! {
                            button {
                                key: "{summary.id}",
                                class: "surface",
                                r#type: "button",
                                id: "rc-{summary.id}",
                                onclick: move |_| {
                                    navigator.push(Route::Transcription { id: id.clone() });
                                },
                                span { class: "nm", dir: "auto", title: "{summary.title}", "{summary.title}" }
                                span { class: "pv", dir: "auto", "{preview_text(summary)}" }
                                span { class: "mt tnum", "{card_meta(&when_text(summary.created_at_ms, now), summary)}" }
                            }
                        }
                    }
                }
            }
        }
    }
}
