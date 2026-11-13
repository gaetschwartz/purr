//! A saved transcription opened from History, shown like a fresh result.

use crate::{
    components::{
        audio_card::WaveMode,
        audio_player::use_audio_player,
        banner::{Banner, BannerTone},
        button::{Button, ButtonSize, ButtonVariant},
        empty_state::EmptyState,
        icon::IconName,
        page::Page,
        presence::Presence,
        spinner::Spinner,
    },
    history::{export_source::export_source, store::use_history, wording::language_text},
    main_app::Route,
    platform::{self, IS_WEB, Platform as _, PlatformError, playback_source::AudioSource},
    transcribe::{engine_label::EngineLabel, session::use_transcribe_session},
    views::{
        history::{delete_dialog::DeleteDialog, rename_dialog::RenameDialog},
        result_frame::{
            ResultAudio, ResultFrame, ResultOrigin, ViewPhase,
            more_menu::{MoreAction, TranscriptionFacts},
        },
    },
};
use dioxus::prelude::*;
use purr_common::history::entry::{AudioAvailability, HistoryId, HistoryRecord, HistorySource};
use std::{path::PathBuf, rc::Rc};

/// Why a saved transcription could not be opened.
#[derive(Debug, Clone, PartialEq)]
enum Unopened {
    Missing,
    Failed(String),
}

fn load_failure(error: &PlatformError) -> Unopened {
    if let PlatformError::HistoryEntryNotFound { .. } = error {
        Unopened::Missing
    } else {
        Unopened::Failed(error.to_string())
    }
}

/// The saved transcription `id` with its stored segments; the player only when the audio is there.
#[component]
pub fn HistoryItem(id: HistoryId) -> Element {
    rsx! {
        Opening { key: "{id}", id }
    }
}

#[component]
fn Opening(id: HistoryId) -> Element {
    let navigator = use_navigator();
    let record = use_resource(move || {
        let id = id.clone();
        async move {
            platform::get_platform()
                .get_history_entry(&id)
                .await
                .map_err(|error| load_failure(&error))
        }
    });
    let loaded: Option<Result<HistoryRecord, Unopened>> = record.read().clone();

    match loaded {
        None => rsx! {
            Page { title: "History",
                div { class: "row", role: "status",
                    Spinner {}
                    span { class: "muted", "Opening the transcription…" }
                }
            }
        },
        Some(Ok(found)) => rsx! {
            Opened { record: found }
        },
        Some(Err(Unopened::Missing)) => rsx! {
            Page { title: "History",
                EmptyState {
                    icon: IconName::Cat,
                    title: "This transcription is gone",
                    actions: rsx! {
                        Button {
                            variant: ButtonVariant::Primary,
                            icon: IconName::Back,
                            onclick: move |_| {
                                navigator.push(Route::History {});
                            },
                            "Back to History"
                        }
                    },
                }
            }
        },
        Some(Err(Unopened::Failed(message))) => rsx! {
            Page { title: "History",
                Banner {
                    tone: BannerTone::Bad,
                    title: "Purr couldn't open this transcription",
                    role: "alert",
                    actions: rsx! {
                        Button {
                            size: ButtonSize::Small,
                            onclick: move |_| {
                                navigator.push(Route::History {});
                            },
                            "Back to History"
                        }
                    },
                    "{message}"
                }
            }
        },
    }
}

fn audio_note(audio: AudioAvailability) -> &'static str {
    match audio {
        AudioAvailability::Available => "",
        AudioAvailability::Missing => "The original audio file was moved or deleted.",
        AudioAvailability::NotKept if IS_WEB => "The browser doesn't keep the audio.",
        AudioAvailability::NotKept => "Where this audio came from wasn't recorded.",
    }
}

#[component]
fn Opened(record: HistoryRecord) -> Element {
    match (&record.entry.source, record.audio) {
        (HistorySource::Path { path }, AudioAvailability::Available) => rsx! {
            Playable { record: record.clone(), path: path.clone() }
        },
        (HistorySource::Path { .. } | HistorySource::NotKept, audio) => {
            let audio = ResultAudio::Unavailable(audio_note(audio).to_owned());
            rsx! {
                Entry { record: record.clone(), audio }
            }
        }
    }
}

#[component]
fn Playable(record: HistoryRecord, path: PathBuf) -> Element {
    let session = use_transcribe_session();
    let id = record.entry.id.clone();
    use_hook(|| session.serve_original(&id, path.clone()));
    let player = use_audio_player(&AudioSource::Original { id, path });
    let peaks = Rc::new(record.entry.waveform());
    let audio = ResultAudio::Player {
        player,
        peaks: (!peaks.is_empty()).then_some(peaks),
    };
    rsx! {
        Entry { record, audio }
    }
}

#[component]
fn Entry(record: HistoryRecord, audio: ResultAudio) -> Element {
    let navigator = use_navigator();
    let store = use_history();
    let mut renaming = use_signal(|| false);
    let mut deleting = use_signal(|| false);

    let entry = &record.entry;
    let title = store
        .listing()
        .and_then(|listing| {
            listing
                .summaries
                .iter()
                .find(|summary| summary.id == entry.id)
                .map(|summary| summary.title.clone())
        })
        .unwrap_or_else(|| entry.title.clone());

    rsx! {
        ResultFrame {
            title: title.clone(),
            phase: ViewPhase::Finished,
            origin: ResultOrigin::Saved,
            file_name: entry.file_name.clone(),
            duration_seconds: Some(entry.audio_seconds),
            audio,
            wave: WaveMode::Playback,
            status: rsx! {},
            segments: Rc::new(entry.segments.clone()),
            export: export_source(entry),
            facts: TranscriptionFacts {
                model_label: entry.model_label.clone(),
                language_text: language_text(entry.language.as_deref(), entry.language_origin),
                engine_text: entry
                    .engine
                    .as_ref()
                    .map(|engine| EngineLabel::saved(engine).to_string()),
            },
            more: vec![MoreAction::Rename, MoreAction::Delete],
            on_more: move |action| match action {
                MoreAction::Rename => renaming.set(true),
                MoreAction::Delete => deleting.set(true),
                MoreAction::RunAgain | MoreAction::NewTranscription => {}
            },
        }
        Presence { present: renaming(),
            RenameDialog {
                id: entry.id.clone(),
                title: title.clone(),
                on_close: move |()| renaming.set(false),
            }
        }
        Presence { present: deleting(),
            DeleteDialog {
                id: entry.id.clone(),
                title,
                on_close: move |()| deleting.set(false),
                on_deleted: move |()| {
                    navigator.push(Route::History {});
                },
            }
        }
    }
}
