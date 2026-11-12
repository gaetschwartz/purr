use super::{cancelled_notice::CancelledNotice, model_notice::ModelNotice, recent::Recent};
use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonSize, ButtonVariant},
        drop_pad::{DropPad, FILE_INPUT_ID},
        engine_chip::EngineChip,
        file_card::FileCard,
        icon::IconName,
        language_picker::LanguagePill,
        model_picker::ModelPill,
        page::Page,
    },
    hooks::{
        use_models::{GpuSupport, use_models},
        use_system_probe::use_system_probe,
    },
    platform::{IS_WEB, Platform as _, get_platform},
    settings_store::use_settings,
    shortcuts::{Key, KeyChord, Shortcut, use_shortcut},
    transcribe::{
        chosen_file::OverviewState,
        engine_label::EngineLabel,
        language_choice::LanguageChoice,
        session::use_transcribe_session,
        start_plan::{FileReadiness, ModelChoice, StartPlan, plan_start},
        timing,
    },
    utils::{byte_size::ByteSize, file_kind::kind_label},
    views::settings::{jump::open_section, sections::SettingsSection},
};
use dioxus::{html::HasFileData, prelude::*};

const OPEN_FILE_CHOOSER: &str = "document.getElementById('file-input')?.click();";

fn formats_line() -> String {
    let formats = "MP3, WAV, AAC, FLAC, OGG, Opus, M4A or WebM";
    get_platform().max_upload_bytes().map_or_else(
        || formats.to_owned(),
        |limit| format!("{formats} · up to {}", ByteSize(limit)),
    )
}

/// The idle and file-chosen states: drop pad or file card, then the run bar.
#[component]
pub fn Start() -> Element {
    let session = use_transcribe_session();
    let models = use_models();
    let settings = use_settings();
    let probe = use_system_probe();
    let navigator = use_navigator();

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
    let no_webgpu = IS_WEB && *models.gpu.read() == GpuSupport::Unavailable;
    let plan = plan_start(readiness, &choice, no_webgpu, other_download);
    let can_start = matches!(plan, StartPlan::Start(_));
    let engine = if IS_WEB {
        EngineLabel::browser(&models.gpu.read())
    } else {
        let policy = settings.with(|settings| settings.performance.accel_policy());
        EngineLabel::planned(policy, session.engine_memory().as_ref(), &probe.read())
    };
    let engine_settings = if IS_WEB {
        SettingsSection::Transcription
    } else {
        SettingsSection::Performance
    };

    let start = move |_| {
        if let Some(row) = &selected {
            session.start(
                row.entry.clone(),
                LanguageChoice::from_setting(&language_setting),
            );
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
                        accept: "audio/*",
                        onchange: move |event: FormEvent| {
                            if let Some(picked) = event.files().into_iter().next() {
                                session.choose(picked);
                            }
                        },
                    }
                } else {
                    DropPad {
                        reading: session.reading(),
                        formats: formats_line(),
                        on_file: move |picked| session.choose(picked),
                    }
                }
                div {
                    class: "runbar glass",
                    role: "group",
                    aria_label: "Transcription options",
                    ModelPill { id: "pick-model" }
                    LanguagePill { id: "pick-lang" }
                    EngineChip {
                        id: "engine-chip",
                        label: engine,
                        onclick: move |_| open_section(navigator, engine_settings),
                    }
                    span { class: "rb-spacer" }
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
                ModelNotice { selection, saved: model_id, plan }
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
