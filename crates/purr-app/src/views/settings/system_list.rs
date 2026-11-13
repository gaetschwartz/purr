use super::system_rows::{Badge, SystemRow};
use crate::components::{
    chip::{Chip, ChipTone},
    icon::IconName,
};
use dioxus::prelude::*;

const fn badge_tone(badge: Badge) -> ChipTone {
    match badge {
        Badge::Available => ChipTone::Ok,
        Badge::Unavailable => ChipTone::Neutral,
    }
}

const fn badge_icon(badge: Badge) -> Option<IconName> {
    match badge {
        Badge::Available => Some(IconName::CheckCircle),
        Badge::Unavailable => None,
    }
}

/// Read-only rows of backend or device, what it is and a status badge.
#[component]
pub fn SystemList(#[props(into)] label: String, rows: Vec<SystemRow>) -> Element {
    rsx! {
        div { class: "group surface sys", role: "list", aria_label: label,
            for (index , row) in rows.into_iter().enumerate() {
                div { key: "{index}", class: "dev", role: "listitem",
                    span { class: "be", "{row.backend}" }
                    span { class: "dn",
                        "{row.name}"
                        if let Some(detail) = &row.detail {
                            small { "{detail}" }
                        }
                    }
                    Chip { tone: badge_tone(row.badge), icon: badge_icon(row.badge), "{row.badge.label()}" }
                }
            }
        }
    }
}
