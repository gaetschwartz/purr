use super::changes::report;
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
use purr_common::{export_format::ExportFormat, settings::Theme};

/// Theme, transparency and the export format.
#[component]
pub fn GeneralPage() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let (theme, reduce_transparency, export_format) = settings.with(|settings| {
        (
            settings.ui.theme,
            settings.ui.reduce_transparency,
            settings.ui.export_format,
        )
    });

    rsx! {
        Group {
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
            Item {
                title: "Default export format",
                stackable: true,
                control: rsx! {
                    Segmented {
                        label: "Export format",
                        value: export_format,
                        segments: ExportFormat::ALL
                            .into_iter()
                            .map(|format| Segment::new(format, format.short_name()))
                            .collect::<Vec<_>>(),
                        onchange: move |format| report(toasts, settings.update(|s| s.ui.export_format = format)),
                    }
                },
            }
        }
    }
}
