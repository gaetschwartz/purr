use super::engine_notes::only_used_by;
use crate::{
    components::{
        chip::{Chip, ChipTone},
        icon::{IconName, IconSize},
        icon_button::IconButton,
        switch::Switch,
    },
    transcribe::device_rows::{DeviceRow, DeviceState, Move, reachable_rows},
};
use dioxus::prelude::*;
use purr_common::{
    accel::{BackendUnavailable, EngineSummary},
    devices::DeviceKey,
    model::engine::EngineId,
};

/// Why a device cannot run, as a short phrase.
#[must_use]
pub fn unavailable_text(why: &BackendUnavailable) -> String {
    match why {
        BackendUnavailable::NotBuilt | BackendUnavailable::NotBuiltOrNoDevice => {
            "Not in this build".to_owned()
        }
        BackendUnavailable::NoDevice => "No device or driver".to_owned(),
        BackendUnavailable::RuntimeLibraryMissing { library, .. } => format!("{library} missing"),
    }
}

/// The cause the badge of a device does not already say, when there is one.
#[must_use]
pub const fn state_cause(state: &DeviceState) -> Option<&BackendUnavailable> {
    match state {
        DeviceState::Available | DeviceState::NotFound(None) => None,
        DeviceState::Unavailable(why) | DeviceState::NotFound(Some(why)) => Some(why),
    }
}

fn engines_text(engines: &[EngineId]) -> String {
    let names: Vec<&str> = engines.iter().map(|engine| engine.label()).collect();
    format!("{} only", names.join(", "))
}

const fn badge(state: &DeviceState) -> (ChipTone, Option<IconName>, &'static str) {
    match state {
        DeviceState::Available => (ChipTone::Ok, Some(IconName::CheckCircle), "Available"),
        DeviceState::Unavailable(_) => (ChipTone::Warn, Some(IconName::Warning), "Unavailable"),
        DeviceState::NotFound(_) => (ChipTone::Neutral, None, "Not found"),
    }
}

fn row_class(row: &DeviceRow) -> &'static str {
    match (&row.state, row.enabled) {
        (DeviceState::NotFound(_), _) => "devrow gone",
        (DeviceState::Available | DeviceState::Unavailable(_), false) => "devrow off",
        (DeviceState::Available | DeviceState::Unavailable(_), true) => "devrow",
    }
}

fn move_id(index: usize, direction: Move) -> String {
    match direction {
        Move::Up => format!("dev-up-{index}"),
        Move::Down => format!("dev-down-{index}"),
    }
}

/// The button to focus after the device at `index` of `count` moved to `to`: the one it was
/// pressed with, or the other when that one has nowhere left to go.
fn focus_target(direction: Move, to: usize, count: usize) -> String {
    let can_continue = match direction {
        Move::Up => to > 0,
        Move::Down => to + 1 < count,
    };
    match (can_continue, direction) {
        (true, direction) => move_id(to, direction),
        (false, Move::Up) => move_id(to, Move::Down),
        (false, Move::Down) => move_id(to, Move::Up),
    }
}

/// The devices in the order they are tried, each with a switch and buttons to move it.
#[component]
pub fn DeviceList(
    rows: Vec<DeviceRow>,
    engines: Vec<EngineSummary>,
    on_move: EventHandler<(DeviceKey, Move)>,
    on_toggle: EventHandler<(DeviceKey, bool)>,
) -> Element {
    let mut announcement = use_signal(String::new);
    let mut focus = use_signal(|| (0_u32, String::new()));
    use_effect(move || {
        let (requests, id) = focus();
        if requests > 0 {
            document::eval(&format!("document.getElementById({id:?})?.focus()"));
        }
    });
    let count = rows.len();
    let reachable = reachable_rows(&rows);

    rsx! {
        div {
            class: "group surface devlist",
            role: "list",
            aria_label: "Devices, in the order they are tried",
            for (index , row) in rows.iter().cloned().enumerate() {
                div {
                    key: "{row.backend.label()}|{row.key.name}|{row.key.occurrence}",
                    class: if index >= reachable && row.enabled { "devrow unreached" } else { row_class(&row) },
                    role: "listitem",
                    span { class: "devrank", aria_hidden: "true", "{index + 1}" }
                    span { class: "devbe", "{row.backend.label()}" }
                    span { class: "devnm",
                        "{row.name}"
                        if let Some(detail) = &row.detail {
                            small { "{detail}" }
                        }
                        if let Some(only) = only_used_by(&row.seen_by, &engines) {
                            small { "{engines_text(&only)}" }
                        }
                        if let Some(why) = state_cause(&row.state) {
                            small { class: "note", "{unavailable_text(why)}" }
                        }
                        if index >= reachable && row.enabled {
                            small { class: "note", "Not reached: the CPU runs first." }
                        }
                    }
                    {
                        let (tone, icon, words) = badge(&row.state);
                        rsx! {
                            Chip { tone, icon, "{words}" }
                        }
                    }
                    span { class: "devsw",
                        if row.key.is_cpu() {
                            span { class: "devlast", "Always on" }
                        } else {
                            Switch {
                                label: format!("Use {} {}", row.backend.label(), row.name),
                                checked: row.enabled,
                                onchange: {
                                    let key = row.key.clone();
                                    move |value| on_toggle.call((key.clone(), value))
                                },
                            }
                        }
                    }
                    span { class: "devmove",
                        for direction in [Move::Up, Move::Down] {
                            IconButton {
                                key: "{move_id(index, direction)}",
                                id: move_id(index, direction),
                                icon: match direction {
                                    Move::Up => IconName::ChevronUp,
                                    Move::Down => IconName::ChevronDown,
                                },
                                size: IconSize::Small,
                                label: match direction {
                                    Move::Up => format!("Move {} {} up", row.backend.label(), row.name),
                                    Move::Down => format!("Move {} {} down", row.backend.label(), row.name),
                                },
                                disabled: match direction {
                                    Move::Up => index == 0,
                                    Move::Down => index + 1 == count,
                                },
                                onclick: {
                                    let key = row.key.clone();
                                    let name = format!("{} {}", row.backend.label(), row.name);
                                    move |_| {
                                        let to = match direction {
                                            Move::Up => index.saturating_sub(1),
                                            Move::Down => (index + 1).min(count.saturating_sub(1)),
                                        };
                                        announcement
                                            .set(format!("{name} moved to position {} of {count}", to + 1));
                                        let requests = focus.peek().0 + 1;
                                        focus.set((requests, focus_target(direction, to, count)));
                                        on_move.call((key.clone(), direction));
                                    }
                                },
                            }
                        }
                    }
                }
            }
        }
        p { class: "sr", role: "status", aria_live: "polite", "{announcement}" }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_cause_the_badge_does_not_say_is_noted() {
        let missing = BackendUnavailable::RuntimeLibraryMissing {
            library: "vulkan-1.dll".to_owned(),
            searched: Vec::new(),
        };
        assert_eq!(state_cause(&DeviceState::Available), None);
        assert_eq!(state_cause(&DeviceState::NotFound(None)), None);
        assert_eq!(
            state_cause(&DeviceState::Unavailable(missing.clone())),
            Some(&missing)
        );
        assert_eq!(
            state_cause(&DeviceState::NotFound(Some(BackendUnavailable::NotBuilt))),
            Some(&BackendUnavailable::NotBuilt)
        );
    }

    #[test]
    fn focus_stays_on_the_pressed_button_until_it_has_nowhere_to_go() {
        assert_eq!(focus_target(Move::Up, 1, 4), "dev-up-1");
        assert_eq!(focus_target(Move::Up, 0, 4), "dev-down-0");
        assert_eq!(focus_target(Move::Down, 2, 4), "dev-down-2");
        assert_eq!(focus_target(Move::Down, 3, 4), "dev-up-3");
    }
}
