use super::{changes::report, sections::SettingsSection};
use crate::{
    components::{
        group::{Group, Item},
        segmented::{Segment, Segmented},
        switch::Switch,
        toast::use_toasts,
    },
    settings_store::use_settings,
};
use dioxus::prelude::*;
use purr_common::settings::Theme;

/// Theme and transparency.
#[component]
pub fn AppearanceSection() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let (theme, reduce_transparency) =
        settings.with(|settings| (settings.ui.theme, settings.ui.reduce_transparency));

    rsx! {
        Group { id: SettingsSection::Appearance.anchor(), header: SettingsSection::Appearance.label(),
            Item {
                title: "Theme",
                stackable: true,
                control: rsx! {
                    Segmented {
                        label: "Theme",
                        value: theme,
                        segments: vec![
                            Segment::new(Theme::System, "Auto"),
                            Segment::new(Theme::Light, "Light"),
                            Segment::new(Theme::Dark, "Dark"),
                        ],
                        onchange: move |theme| report(toasts, settings.set_theme(theme)),
                    }
                },
            }
            Item {
                title: "Reduce transparency",
                control: rsx! {
                    Switch {
                        label: "Reduce transparency",
                        checked: reduce_transparency,
                        onchange: move |reduce| report(toasts, settings.set_reduce_transparency(reduce)),
                    }
                },
            }
        }
    }
}
