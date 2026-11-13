use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonSize},
        icon::IconName,
        progress_bar::{Progress, ProgressBar},
        spinner::Spinner,
    },
    hooks::use_models::use_models,
    model_catalog::{
        copy,
        format::{rate_and_eta, transferred},
        selection::Selection,
        state::ModelState,
    },
    transcribe::start_plan::{Blocker, StartPlan, blocked_text},
};
use dioxus::prelude::*;

/// What stands between the selected model and a run, when it is worth saying; nothing otherwise.
#[component]
pub fn ModelNotice(selection: Option<Selection>, plan: StartPlan) -> Element {
    let models = use_models();

    if let Some(Err(error)) = models.rows() {
        return rsx! {
            Banner {
                tone: BannerTone::Bad,
                title: "Couldn't list the models",
                role: "alert",
                actions: rsx! {
                    Button {
                        size: ButtonSize::Small,
                        icon: IconName::Retry,
                        onclick: move |_| models.refresh(),
                        "Try again"
                    }
                },
                "{error}"
            }
        };
    }
    let Some(selection) = selection else {
        return rsx! {};
    };
    let Some(row) = selection.row().cloned() else {
        return rsx! {
            Banner { tone: BannerTone::Warn, title: "No model is available", role: "alert" }
        };
    };
    let label = row.entry.label.clone();
    let id = row.entry.id.clone();
    let state_notice = match row.state {
        ModelState::Downloaded | ModelState::NotDownloaded => rsx! {},
        ModelState::Downloading(progress) => rsx! {
            div { class: "card surface stack", role: "status", style: "gap:10px",
                div { class: "row",
                    b { class: "grow", "Downloading {label}" }
                    Button {
                        size: ButtonSize::Small,
                        onclick: move |_| models.cancel(&id),
                        "Cancel"
                    }
                }
                ProgressBar {
                    progress: progress.fraction().map_or(Progress::Indeterminate, Progress::Determinate),
                    label: "Download of {label}",
                }
                div { class: "dlmeta",
                    span { {transferred(&progress)} }
                    span { {rate_and_eta(&progress)} }
                }
            }
        },
        ModelState::Verifying => rsx! {
            div { class: "row", role: "status",
                Spinner {}
                span { class: "muted", "Checking {label}…" }
            }
        },
        ModelState::Failed(failure) => {
            let text = copy::failure(&failure);
            rsx! {
                Banner { tone: BannerTone::Bad, title: "{text.title}", role: "alert", "{text.body}" }
            }
        }
        ModelState::NeedsRepair(damage) => rsx! {
            Banner { tone: BannerTone::Bad, title: "{label} is damaged", role: "alert",
                {copy::damage(&damage)}
            }
        },
        ModelState::Unavailable { reason, .. } => rsx! {
            Banner { tone: BannerTone::Bad, title: "{label} can't run here", role: "alert",
                {blocked_text(&Blocker::CannotRun(reason))}
            }
        },
    };

    rsx! {
        {state_notice}
        if plan == StartPlan::Blocked(Blocker::OtherDownload) {
            p { class: "foot", role: "status", style: "margin:0",
                {blocked_text(&Blocker::OtherDownload)}
            }
        }
    }
}
