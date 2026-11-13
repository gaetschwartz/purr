use super::{
    icon::{Icon, IconName, IconSize},
    pill::Pill,
    popover::{OptionRole, Popover, PopoverOption},
    presence::Presence,
};
use crate::{
    hooks::use_system_probe::{SystemProbe, use_system_probe},
    settings_store::use_settings,
    transcribe::{
        device_rows::{
            DeviceChoice, DeviceRow, choices, device_rows, override_for, row_of, row_phrase,
        },
        session::use_transcribe_session,
    },
};
use dioxus::prelude::*;
use purr_common::devices::DeviceKey;

const AUTOMATIC: &str = "Automatic (settings order)";

fn title(choice: &DeviceChoice, rows: &[DeviceRow]) -> String {
    match choice {
        DeviceChoice::Automatic => AUTOMATIC.to_owned(),
        DeviceChoice::Device(key) => row_of(rows, key).map_or_else(String::new, row_phrase),
    }
}

/// The pill showing which device the next runs start on; it opens a [`DevicePicker`].
///
/// The choice lives in the Transcribe session, so it applies until the app restarts and never
/// reaches the settings.
#[component]
pub fn DevicePill(
    /// The pill's DOM id, which the popover hangs from.
    #[props(into)]
    id: String,
) -> Element {
    let session = use_transcribe_session();
    let settings = use_settings();
    let probe = use_system_probe();
    let mut open = use_signal(|| false);
    let rows = match &*probe.read() {
        SystemProbe::Ready(report) => Some(device_rows(
            &settings.with(|settings| settings.performance.clone()),
            report,
        )),
        SystemProbe::Checking | SystemProbe::Unavailable => None,
    };
    let picked = rows
        .as_ref()
        .and_then(|rows| override_for(session.device_override().as_ref(), rows));
    let value = match (&picked, &rows) {
        (Some(key), Some(rows)) => {
            row_of(rows, key).map_or_else(|| AUTOMATIC.to_owned(), row_phrase)
        }
        (None, _) | (Some(_), None) => AUTOMATIC.to_owned(),
    };
    rsx! {
        Pill {
            id: id.clone(),
            icon: IconName::Cpu,
            label: "Device",
            value,
            hint: "Where this session's transcriptions start. The order in Settings still applies if it fails.",
            expanded: open(),
            disabled: rows.is_none(),
            onclick: move |_| open.toggle(),
        }
        if let Some(rows) = rows {
            Presence { present: open(),
                DevicePicker {
                    anchor: id,
                    rows,
                    picked,
                    on_close: move |()| open.set(false),
                }
            }
        }
    }
}

/// The devices a run can start on, anchored on `anchor`; render it inside a [`Presence`].
#[component]
pub fn DevicePicker(
    #[props(into)] anchor: String,
    rows: Vec<DeviceRow>,
    picked: Option<DeviceKey>,
    on_close: EventHandler<()>,
) -> Element {
    let session = use_transcribe_session();
    let options = choices(&rows);
    rsx! {
        Popover { anchor, label: "Device", on_close, wide: true,
            for (index , choice) in options.into_iter().enumerate() {
                {
                    let selected = match &choice {
                        DeviceChoice::Automatic => picked.is_none(),
                        DeviceChoice::Device(key) => picked.as_ref() == Some(key),
                    };
                    let subtitle = match &choice {
                        DeviceChoice::Automatic => {
                            Some("Tries the devices in the order set in Settings".to_owned())
                        }
                        DeviceChoice::Device(key) => row_of(&rows, key).and_then(|row| row.detail.clone()),
                    };
                    let name = title(&choice, &rows);
                    rsx! {
                        PopoverOption {
                            key: "{index}",
                            title: name,
                            subtitle,
                            role: OptionRole::Option,
                            selected,
                            leading: rsx! {
                                if selected {
                                    Icon { name: IconName::Check, size: IconSize::Small }
                                }
                            },
                            onselect: move |()| {
                                session
                                    .set_device_override(
                                        match &choice {
                                            DeviceChoice::Automatic => None,
                                            DeviceChoice::Device(key) => Some(key.clone()),
                                        },
                                    );
                                on_close.call(());
                            },
                        }
                    }
                }
            }
        }
    }
}
