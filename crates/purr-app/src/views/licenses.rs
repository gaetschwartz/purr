//! The Licenses view: the third-party crates and the font Purr ships with, grouped by licence.

mod catalog;
mod group;

use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonSize, ButtonVariant},
        icon::IconName,
        page::Page,
        search_field::SearchField,
    },
    main_app::Route,
    views::settings::{dialog_handle::use_settings_dialog, settings_page::SettingsPage},
};
use catalog::Licence;
use dioxus::prelude::*;
use group::{FontGroup, LicenceGroup};
use std::rc::Rc;

/// The page that lists every licence, with the crates released under it and the full text.
#[component]
pub fn Licenses() -> Element {
    let navigator = use_navigator();
    let dialog = use_settings_dialog();
    let licences = use_hook(|| {
        catalog::generated()
            .map(Rc::new)
            .map_err(|error| error.to_string())
    });
    let mut query = use_signal(String::new);

    let eyebrow = rsx! {
        Button {
            variant: ButtonVariant::Plain,
            size: ButtonSize::Small,
            icon: IconName::Back,
            onclick: move |_| {
                navigator.push(Route::Home {});
                dialog.open(SettingsPage::Advanced);
            },
            "Settings"
        }
    };

    rsx! {
        Page {
            title: "Licenses",
            subtitle: "Purr is built on open source software.",
            eyebrow,
            match &licences {
                Ok(licences) => rsx! {
                    SearchField {
                        value: query(),
                        oninput: move |value| query.set(value),
                        placeholder: "Search crates",
                        label: "Search crates",
                    }
                    LicenceList { licences: Rc::clone(licences), query: query() }
                    if query().trim().is_empty() {
                        FontGroup {}
                    }
                },
                Err(error) => rsx! {
                    Banner {
                        tone: BannerTone::Bad,
                        title: "The licence list could not be read",
                        role: "alert",
                        "{error}"
                    }
                    FontGroup {}
                },
            }
        }
    }
}

#[component]
fn LicenceList(licences: Rc<Vec<Licence>>, query: String) -> Element {
    let groups = licences
        .iter()
        .filter(|licence| {
            licence
                .texts
                .iter()
                .any(|notice| !notice.matching(&query).is_empty())
        })
        .collect::<Vec<_>>();
    if groups.is_empty() {
        return rsx! {
            p { class: "foot", "No crate matches “{query}”." }
        };
    }
    rsx! {
        for licence in groups {
            LicenceGroup {
                key: "{licence.id}",
                licence: licence.clone(),
                query: query.clone(),
            }
        }
    }
}
