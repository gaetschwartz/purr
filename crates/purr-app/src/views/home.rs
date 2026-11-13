//! The Transcribe screen at `/`: choose a file, check the run bar, start. A run lives at its own
//! `/transcription/:id` address from the moment it starts.

mod cancelled_notice;
mod model_notice;
mod recent;
mod run_link;

use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonSize, ButtonVariant},
        device_pill::DevicePill,
        drop_pad::{DropPad, FILE_INPUT_ID},
        file_card::FileCard,
        icon::IconName,
        language_picker::LanguagePill,
        model_picker::ModelPill,
        page::Page,
    },
    hooks::use_models::use_models,
    main_app::Route,
    platform::{IS_WEB, Platform as _, get_platform},
    settings_store::use_settings,
    shortcuts::{Key, KeyChord, Shortcut, use_shortcut},
    transcribe::{
        chosen_file::OverviewState,
        expected_browser_engine::expected_browser_engine,
        language_choice::LanguageChoice,
        session::use_transcribe_session,
        start_plan::{FileReadiness, ModelChoice, StartPlan, plan_start},
        timing,
    },
    utils::{byte_size::ByteSize, file_kind::kind_label},
};
use cancelled_notice::CancelledNotice;
use dioxus::{html::HasFileData, prelude::*};
use model_notice::ModelNotice;
use purr_common::browser_engine::BrowserEngine;
use recent::Recent;
use run_link::RunLink;

const OPEN_FILE_CHOOSER: &str = "document.getElementById('file-input')?.click();";

fn size_limit() -> Option<String> {
    get_platform()
        .max_upload_bytes()
        .map(|limit| format!("Up to {}", ByteSize(limit)))
}

/// The Transcribe view: drop pad or file card, then the run bar and the latest transcriptions.
#[component]
pub fn Home() -> Element {
    let session = use_transcribe_session();
    let navigator = use_navigator();
    let models = use_models();
    let settings = use_settings();

    use_shortcut(
        Shortcut::new(KeyChord::primary(Key::Char('o')), "Choose an audio file").in_inputs(),
        move || {
            document::eval(OPEN_FILE_CHOOSER);
        },
    );

    let (model_id, language_setting) = settings.with(|settings| {
        (
            settings.model.model_name.clone(),
            settings.ui.language.clone(),
        )
    });
    let selection = models.selection(&model_id);
    let selected = selection
        .as_ref()
        .and_then(|selection| selection.row().cloned());
    let file = session.file();
    let overview = file.as_ref().map(|file| session.overview(&file.id));
    let readiness = match (&file, &overview) {
        (None, _) => FileReadiness::Missing,
        (Some(_), Some(OverviewState::Undecodable(_))) => FileReadiness::Undecodable,
        (Some(_), Some(OverviewState::Reading | OverviewState::Ready { .. }) | None) => {
            FileReadiness::Usable
        }
    };
    let choice = match (&selection, &selected) {
        (None, _) => ModelChoice::Loading,
        (Some(_), None) => ModelChoice::Missing,
        (Some(_), Some(row)) => ModelChoice::Chosen(row.state.clone()),
    };
    let other_download = selected
        .as_ref()
        .is_some_and(|row| models.is_downloading() && !models.is_busy(&row.entry.id));
    let plan = plan_start(readiness, &choice, other_download);
    let running_id = use_memo(move || session.running_id());
    let running = running_id();
    let can_start = matches!(plan, StartPlan::Start(_)) && running.is_none();
    let devices = settings.with(|settings| settings.performance.device_order());
    let browser_engine = IS_WEB.then(|| {
        expected_browser_engine(
            &devices,
            &models.gpu.read(),
            selected
                .as_ref()
                .is_some_and(|row| row.entry.requires_shader_f16),
        )
    });
    let mut advanced = use_signal(|| false);
    let cpu_notice = match browser_engine {
        Some(BrowserEngine::Cpu(reason)) => reason.notice(),
        Some(BrowserEngine::WebGpu) | None => None,
    };

    let start = move |_| {
        if let Some(row) = &selected {
            if let Some(id) = session.start(
                row.entry.clone(),
                LanguageChoice::from_setting(&language_setting),
            ) {
                navigator.push(Route::Transcription { id });
            }
        }
    };

    rsx! {
        Page { title: "Transcribe",
            div {
                class: "stack",
                ondragover: move |event: DragEvent| event.prevent_default(),
                ondrop: move |event: DragEvent| {
                    event.prevent_default();
                    if let Some(dropped) = event.data().files().into_iter().next() {
                        session.choose(dropped);
                    }
                },
                if let Some(id) = running {
                    RunLink { id }
                }
                if let Some(cancelled) = session.cancelled() {
                    CancelledNotice { cancelled, session }
                }
                if let Some(problem) = session.problem() {
                    Banner {
                        tone: BannerTone::Bad,
                        title: problem.title(),
                        role: "alert",
                        actions: rsx! {
                            Button {
                                variant: ButtonVariant::Primary,
                                size: ButtonSize::Small,
                                onclick: move |_| {
                                    document::eval(OPEN_FILE_CHOOSER);
                                },
                                "Choose another file"
                            }
                            Button {
                                size: ButtonSize::Small,
                                onclick: move |_| session.dismiss_problem(),
                                "Dismiss"
                            }
                        },
                        "{problem.message()}"
                    }
                }
                if let Some(file) = file {
                    FileCard {
                        name: file.name.clone(),
                        details: file_details(overview.as_ref(), file.size, &file.name),
                        problem: undecodable(overview.as_ref(), &file.name),
                        on_replace: move |()| {
                            document::eval(OPEN_FILE_CHOOSER);
                        },
                        on_remove: move |()| session.clear_file(),
                    }
                    input {
                        r#type: "file",
                        class: "sr",
                        id: FILE_INPUT_ID,
                        tabindex: "-1",
                        aria_hidden: "true",
                        onchange: move |event: FormEvent| {
                            if let Some(picked) = event.files().into_iter().next() {
                                session.choose(picked);
                            }
                        },
                    }
                } else {
                    DropPad {
                        reading: session.reading(),
                        limit: size_limit(),
                        on_file: move |picked| session.choose(picked),
                    }
                }
                div {
                    class: "runbar surface",
                    role: "group",
                    aria_label: "Transcription options",
                    ModelPill { id: "pick-model" }
                    LanguagePill { id: "pick-lang" }
                    Button {
                        variant: ButtonVariant::Primary,
                        size: ButtonSize::Large,
                        icon: IconName::Wave,
                        id: "start-btn",
                        disabled: !can_start,
                        onclick: start,
                        "Transcribe"
                    }
                }
                if !IS_WEB {
                    div { class: "advrow",
                        button {
                            id: "advanced-toggle",
                            class: "advtoggle",
                            r#type: "button",
                            aria_expanded: advanced(),
                            aria_controls: "advanced-options",
                            onclick: move |_| advanced.toggle(),
                            "Advanced"
                        }
                        if advanced() {
                            div { id: "advanced-options", class: "advopts", DevicePill { id: "pick-device" } }
                        }
                    }
                }
                ModelNotice { selection, plan }
                if let Some(notice) = cpu_notice {
                    p { class: "foot", role: "status", id: "cpu-notice", style: "margin:0",
                        "{notice}"
                    }
                }
                Recent {}
            }
        }
    }
}

fn file_details(overview: Option<&OverviewState>, size: u64, name: &str) -> String {
    let length = match overview {
        Some(OverviewState::Ready {
            duration_seconds, ..
        }) => timing::clock(*duration_seconds),
        Some(OverviewState::Reading) | None => "Reading…".to_owned(),
        Some(OverviewState::Undecodable(_)) => "Unknown length".to_owned(),
    };
    format!("{length} · {} · {}", ByteSize(size), kind_label(name))
}

fn undecodable(overview: Option<&OverviewState>, name: &str) -> Option<String> {
    match overview {
        Some(OverviewState::Undecodable(detail)) => Some(format!(
            "{name} can't be decoded as audio ({detail}). Try exporting it again from the app that recorded it, or choose another file."
        )),
        Some(OverviewState::Reading | OverviewState::Ready { .. }) | None => None,
    }
}
