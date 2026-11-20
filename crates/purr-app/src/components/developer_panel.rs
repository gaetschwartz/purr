use super::{
    button::{Button, ButtonSize},
    icon::IconName,
    icon_button::IconButton,
    toast::use_toasts,
};
use crate::{
    developer_report::{Section, as_text, sections},
    hooks::use_developer_panel::use_developer_panel,
    platform::{self, export::copy_to_clipboard},
    settings_store::use_settings,
    utils::delay::sleep,
};
use dioxus::{core::spawn_forever, prelude::*};
use purr_common::{developer::snapshot::DeveloperSnapshot, platform::Platform as _};
use std::time::Duration;

const REFRESH: Duration = Duration::from_secs(1);

/// The button that opens the developer panel; present while developer mode is on.
#[component]
pub fn DeveloperLauncher() -> Element {
    let settings = use_settings();
    let panel = use_developer_panel();
    let enabled = settings.with(|settings| settings.developer.enabled);
    if !enabled || panel.is_open() {
        return rsx! {};
    }
    rsx! {
        button {
            class: "dev-launch",
            r#type: "button",
            aria_haspopup: "dialog",
            title: "Open the developer panel",
            onclick: move |_| panel.open(),
            "Dev"
        }
    }
}

/// A floating panel with the live developer readings; it stays out of the way of the run.
#[component]
pub fn DeveloperPanel() -> Element {
    let settings = use_settings();
    let panel = use_developer_panel();
    let enabled = settings.with(|settings| settings.developer.enabled);
    if !enabled || !panel.is_open() {
        return rsx! {};
    }
    rsx! {
        PanelBody {}
    }
}

#[component]
fn PanelBody() -> Element {
    let panel = use_developer_panel();
    let toasts = use_toasts();
    let mut reading = use_signal(|| None::<Result<DeveloperSnapshot, String>>);
    let mut reset = use_signal(|| false);

    use_future(move || async move {
        loop {
            let reset_peaks = reset.replace(false);
            let next = platform::get_platform()
                .developer_snapshot(reset_peaks)
                .await
                .map_err(|error| error.to_string());
            reading.set(Some(next));
            sleep(REFRESH).await;
        }
    });

    let shown: Option<Result<Vec<Section>, String>> = reading
        .read()
        .as_ref()
        .map(|result| result.as_ref().map(sections).map_err(Clone::clone));
    let text = shown
        .as_ref()
        .and_then(|result| result.as_ref().ok())
        .map(|sections| as_text(sections));

    let copy = move |_| {
        let Some(text) = text.clone() else {
            return;
        };
        let copying = copy_to_clipboard(&text);
        spawn_forever(async move {
            match copying.await {
                Ok(()) => {
                    toasts.success(
                        "Developer readings copied",
                        "They are on the clipboard as text.",
                    );
                }
                Err(error) => {
                    toasts.error("Couldn't copy the readings", error.to_string());
                }
            }
        });
    };

    rsx! {
        aside {
            class: "dev-panel glass g3",
            role: "dialog",
            aria_label: "Developer panel",
            div { class: "dev-head",
                h2 { "Developer" }
                Button { size: ButtonSize::Small, onclick: move |_| reset.set(true), "Reset peaks" }
                Button { size: ButtonSize::Small, icon: IconName::Copy, onclick: copy, "Copy" }
                IconButton {
                    icon: IconName::Close,
                    label: "Close the developer panel",
                    onclick: move |_| panel.close(),
                }
            }
            div { class: "dev-body",
                match shown {
                    None => rsx! {
                        p { class: "group-f", role: "status", "Reading…" }
                    },
                    Some(Err(error)) => rsx! {
                        p { class: "group-f", role: "alert", "{error}" }
                    },
                    Some(Ok(sections)) => rsx! {
                        for section in sections {
                            section { key: "{section.title}",
                                h3 { "{section.title}" }
                                dl {
                                    for row in section.rows {
                                        div { key: "{row.label}",
                                            dt { "{row.label}" }
                                            dd { "{row.value}" }
                                        }
                                    }
                                }
                            }
                        }
                    },
                }
            }
        }
    }
}
