use super::engine_notice::EngineNotice;
use crate::{
    components::{
        audio_player::use_audio_player,
        button::{Button, ButtonSize, ButtonVariant},
        export_menu::ExportSource,
        icon::{Icon, IconName},
        model_picker::ModelPicker,
    },
    history::save_state::SaveState,
    main_app::Route,
    platform::playback_source::AudioSource,
    transcribe::{
        chosen_file::OverviewState,
        language_choice::LanguageChoice,
        run::Completion,
        session::{RunRecord, use_transcribe_session},
    },
    utils::language_label::language_label,
    views::result_frame::{ResultAudio, ResultDetails, ResultFrame},
};
use dioxus::prelude::*;
use std::rc::Rc;

/// A finished transcription: details, the player, the transcript and its exports.
#[component]
pub fn ResultView(record: RunRecord, completion: Completion) -> Element {
    let session = use_transcribe_session();
    let player = use_audio_player(&AudioSource::Upload(record.file.clone()));
    let mut picking = use_signal(|| false);

    let segments = Rc::new(record.run.transcript.segments());
    let overview = session.overview(&record.file.id);
    let duration = overview.duration().unwrap_or(completion.audio_seconds);
    let export = ExportSource {
        file_name: record.file.name.clone(),
        model: record.model.id.clone(),
        language: record.language_code(),
        duration_seconds: duration,
        segments: Rc::clone(&segments),
    };

    let language_text = match (&record.language, record.language_code()) {
        (LanguageChoice::Automatic, Some(code)) => format!("{} (detected)", language_label(&code)),
        (LanguageChoice::Automatic, None) => "Language not reported".to_owned(),
        (LanguageChoice::Spoken(_), _) => record.language.label(),
    };
    let peaks = match &overview {
        OverviewState::Ready { peaks, .. } => Some(Rc::clone(peaks)),
        OverviewState::Reading | OverviewState::Undecodable(_) => None,
    };
    let keeping = match session.save_state() {
        SaveState::Idle => rsx! {},
        SaveState::Saving => rsx! {
            span {
                Icon { name: IconName::Lock }
                "Saving to History…"
            }
        },
        SaveState::Saved(id) => rsx! {
            span {
                Icon { name: IconName::Lock }
                Link { to: Route::HistoryItem { id }, "Saved to History" }
            }
        },
        SaveState::Failed { .. } => rsx! {
            span { role: "alert",
                Icon { name: IconName::Warning }
                "Not saved to History"
                Button {
                    variant: ButtonVariant::Plain,
                    size: ButtonSize::Small,
                    id: "save-retry",
                    onclick: move |_| session.retry_save(),
                    "Try again"
                }
            }
        },
    };

    rsx! {
        ResultFrame {
            title: record.file.name.clone(),
            eyebrow: rsx! {
                p { class: "sub", style: "margin:0 0 2px",
                    Button {
                        variant: ButtonVariant::Plain,
                        size: ButtonSize::Small,
                        icon: IconName::Back,
                        id: "back-new",
                        style: "margin-left:-8px",
                        onclick: move |_| session.new_transcription(),
                        "New transcription"
                    }
                }
            },
            actions: rsx! {
                Button {
                    icon: IconName::Retry,
                    id: "run-again",
                    aria_haspopup: "dialog",
                    onclick: move |_| picking.toggle(),
                    "Run again"
                }
            },
            details: ResultDetails {
                duration_seconds: duration,
                model_label: record.model.label.clone(),
                language_text,
                took_seconds: completion.processing_seconds,
                engine: record.run.engine.as_ref().map(|ran| ran.backend.label()),
            },
            notice: rsx! {
                EngineNotice { record: record.clone() }
            },
            keeping,
            segments,
            export,
            audio: ResultAudio::Player { player, peaks },
        }
        if picking() {
            ModelPicker {
                anchor: "run-again",
                selected: record.model.id,
                on_close: move |()| picking.set(false),
                on_pick: move |entry| session.run_again(entry),
            }
        }
    }
}
