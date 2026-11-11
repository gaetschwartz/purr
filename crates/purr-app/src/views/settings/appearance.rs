use super::{changes::report, sections::SettingsSection};
use crate::{
    components::{
        group::{Group, Item, TileColor},
        icon::IconName,
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
                icon: IconName::Sun,
                tile: TileColor::Blue,
                title: "Theme",
                stackable: true,
                control: rsx! {
                    Segmented {
                        label: "Theme",
                        value: theme,
                        segments: vec![
                            Segment::new(Theme::System, "Auto").with_icon(IconName::Auto),
                            Segment::new(Theme::Light, "Light").with_icon(IconName::Sun),
                            Segment::new(Theme::Dark, "Dark").with_icon(IconName::Moon),
                        ],
                        onchange: move |theme| report(toasts, settings.set_theme(theme)),
                    }
                },
            }
            Item {
                icon: IconName::Stack,
                tile: TileColor::Gray,
                title: "Reduce transparency",
                subtitle: "Solid backgrounds instead of glass. Also follows the system setting.",
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
