//! One view for a run in progress, its result and its failure, so nothing remounts at the end.

mod glide_driver;
mod status;

use super::engine_notice::EngineNotice;
use crate::{
    components::{
        audio_card::WaveMode,
        audio_player::use_audio_player,
        banner::{Banner, BannerTone},
        button::{Button, ButtonSize, ButtonVariant},
        export_menu::ExportSource,
        icon::{Icon, IconName, IconSize},
        model_picker::ModelPicker,
    },
    history::save_state::SaveState,
    hooks::use_models::use_models,
    main_app::Route,
    platform::{IS_WEB, playback_source::AudioSource},
    shortcuts::{Key, KeyChord, Shortcut, use_shortcut},
    transcribe::{
        chosen_file::OverviewState,
        eta::EtaText,
        language_choice::LanguageChoice,
        run::{RunOutcome, Stage},
        session::{RunRecord, use_transcribe_session},
        timing,
    },
    utils::language_label::language_label,
    views::result_frame::{
        ResultAudio, ResultFrame, ViewPhase,
        more_menu::{MORE_ID, MoreAction, TranscriptionFacts},
    },
};
use dioxus::prelude::*;
use glide_driver::{GlideDriver, GlideInput, GlidePhase};
use status::RunStatus;
use std::rc::Rc;

const FAIL_MODEL_ID: &str = "fail-model";

/// The language line of the More menu.
fn language_text(record: &RunRecord) -> String {
    match (
        &record.language,
        record.language_code(),
        &record.run.outcome,
    ) {
        (LanguageChoice::Automatic, Some(code), _) => {
            format!("{} (detected)", language_label(&code))
        }
        (LanguageChoice::Automatic, None, RunOutcome::Running) => {
            "Language not known yet".to_owned()
        }
        (LanguageChoice::Automatic, None, RunOutcome::Completed(_) | RunOutcome::Failed(_)) => {
            "Language not reported".to_owned()
        }
        (LanguageChoice::Spoken(_), _, _) => record.language.label(),
    }
}

/// The heading of the error banner.
fn failure_heading(record: &RunRecord, duration: Option<f64>) -> String {
    let reached = record.run.transcribed_seconds();
    match (record.run.stage, duration) {
        (Stage::Transcribing, Some(total)) if reached > 0.0 => format!(
            "Purr stopped at {} of {}",
            timing::clock(reached),
            timing::clock(total)
        ),
        (Stage::LoadingModel, _) => format!("{} couldn't be loaded", record.model.label),
        (Stage::DetectingLanguage | Stage::Transcribing, _) => {
            format!("{} couldn't be transcribed", record.file.name)
        }
    }
}

/// The length of the audio: decoded, else as the finished run measured it.
fn audio_length(record: &RunRecord, overview: &OverviewState) -> Option<f64> {
    overview.duration().or(match &record.run.outcome {
        RunOutcome::Completed(completion) => Some(completion.audio_seconds),
        RunOutcome::Running | RunOutcome::Failed(_) => None,
    })
}

/// What the fill follows for `record`, with `duration` seconds of audio.
fn glide_input(record: &RunRecord, duration: f64) -> GlideInput {
    let target = record
        .run
        .fraction(duration)
        .map_or(0.0, |share| share * duration);
    let phase = match (&record.run.outcome, record.run.stage) {
        (RunOutcome::Running, Stage::LoadingModel | Stage::DetectingLanguage) => {
            GlidePhase::Waiting
        }
        (RunOutcome::Running, Stage::Transcribing) => GlidePhase::Transcribing,
        (RunOutcome::Completed(_), _) => GlidePhase::Finished,
        (RunOutcome::Failed(_), _) => GlidePhase::Stopped,
    };
    GlideInput {
        target,
        duration,
        phase,
    }
}

/// The run on the Transcribe screen: running, finished or failed, in the same elements.
#[component]
pub fn RunView(record: RunRecord) -> Element {
    let session = use_transcribe_session();
    let models = use_models();
    let navigator = use_navigator();
    let player = use_audio_player(&AudioSource::Upload(record.file.clone()));
    let mut picking = use_signal(|| None::<&'static str>);
    let eta = use_signal(|| None::<EtaText>);
    let arrived = use_signal(|| record.run.outcome != RunOutcome::Running);
    let input = use_memo(move || {
        session.record().map_or(
            GlideInput {
                target: 0.0,
                duration: 0.0,
                phase: GlidePhase::Waiting,
            },
            |record| {
                let duration = audio_length(&record, &session.overview(&record.file.id));
                glide_input(&record, duration.unwrap_or(0.0))
            },
        )
    });
    if !IS_WEB {
        use_shortcut(
            Shortcut::new(KeyChord::primary(Key::Char('n')), "New transcription").in_inputs(),
            move || {
                let ended = session
                    .record()
                    .is_some_and(|record| record.run.outcome != RunOutcome::Running);
                if ended {
                    session.new_transcription();
                }
            },
        );
    }

    let run = &record.run;
    let overview = session.overview(&record.file.id);
    let duration = audio_length(&record, &overview);
    let total = duration.unwrap_or(0.0);
    let done = input().target;
    let segments = Rc::new(run.transcript.segments());
    let export = ExportSource {
        file_name: record.file.name.clone(),
        model: record.model.id.clone(),
        language: record.language_code(),
        duration_seconds: total,
        segments: Rc::clone(&segments),
    };
    let peaks = match &overview {
        OverviewState::Ready { peaks, .. } => Some(Rc::clone(peaks)),
        OverviewState::Reading | OverviewState::Undecodable(_) => None,
    };

    let phase = match (&run.outcome, run.stage, arrived()) {
        (RunOutcome::Running, Stage::LoadingModel, _) => ViewPhase::Preparing,
        (RunOutcome::Running, Stage::DetectingLanguage, _) => ViewPhase::DetectingLanguage,
        (RunOutcome::Running, Stage::Transcribing, _) | (RunOutcome::Completed(_), _, false) => {
            ViewPhase::Transcribing
        }
        (RunOutcome::Completed(_), _, true) => ViewPhase::Finished,
        (RunOutcome::Failed(_), _, _) => ViewPhase::Stopped,
    };
    let status = match &run.outcome {
        RunOutcome::Running => Some(RunStatus::running(
            run.stage,
            &models.state(&record.model.id),
            eta(),
        )),
        RunOutcome::Failed(_) => Some(RunStatus::Stopped(done)),
        RunOutcome::Completed(_) => None,
    };
    let wave = match phase {
        ViewPhase::Finished => WaveMode::Playback,
        ViewPhase::Preparing
        | ViewPhase::DetectingLanguage
        | ViewPhase::Transcribing
        | ViewPhase::Stopped => {
            let reported = run.stage == Stage::Transcribing;
            let eta_words = match status {
                Some(RunStatus::Transcribing(Some(left))) => format!(", {left}"),
                Some(_) | None => String::new(),
            };
            WaveMode::Progress {
                done: reported.then_some(done),
                value_text: if reported {
                    format!(
                        "{} of {} transcribed{eta_words}",
                        timing::clock(done),
                        timing::clock(total)
                    )
                } else {
                    status.map(|status| status.to_string()).unwrap_or_default()
                },
                stopped: phase == ViewPhase::Stopped,
            }
        }
    };
    let on_stop = match (&run.outcome, phase) {
        (RunOutcome::Running, _) => Some(EventHandler::new(move |()| session.cancel())),
        (RunOutcome::Completed(_), ViewPhase::Transcribing) => {
            Some(EventHandler::new(move |()| {}))
        }
        (RunOutcome::Completed(_) | RunOutcome::Failed(_), _) => None,
    };
    let more = match &run.outcome {
        RunOutcome::Running => Vec::new(),
        RunOutcome::Completed(_) => vec![MoreAction::RunAgain, MoreAction::NewTranscription],
        RunOutcome::Failed(_) => vec![MoreAction::NewTranscription],
    };
    let failure = match &run.outcome {
        RunOutcome::Failed(failure) => Some(failure.clone()),
        RunOutcome::Running | RunOutcome::Completed(_) => None,
    };
    let save_failed = matches!(run.outcome, RunOutcome::Completed(_))
        && matches!(session.save_state(), SaveState::Failed { .. });

    rsx! {
        ResultFrame {
            title: record.file.name.clone(),
            phase,
            file_name: record.file.name.clone(),
            duration_seconds: duration,
            notice: rsx! {
                EngineNotice { record: record.clone() }
                if let Some(failure) = failure {
                    Banner {
                        tone: BannerTone::Bad,
                        title: failure_heading(&record, duration),
                        role: "alert",
                        actions: rsx! {
                            Button {
                                variant: ButtonVariant::Primary,
                                size: ButtonSize::Small,
                                icon: IconName::Retry,
                                id: "retry-run",
                                onclick: move |_| session.retry(),
                                "Try again"
                            }
                            Button {
                                size: ButtonSize::Small,
                                id: FAIL_MODEL_ID,
                                aria_haspopup: "dialog",
                                onclick: move |_| picking.set(Some(FAIL_MODEL_ID)),
                                "Use another model"
                            }
                            Button {
                                size: ButtonSize::Small,
                                onclick: move |_| session.new_transcription(),
                                "Choose another file"
                            }
                            Button {
                                variant: ButtonVariant::Plain,
                                size: ButtonSize::Small,
                                onclick: move |_| {
                                    navigator.push(Route::Logs {});
                                },
                                "View logs"
                            }
                        },
                        "{failure.message}"
                        if let Some(hint) = &failure.hint {
                            " {hint}"
                        }
                    }
                }
                if save_failed {
                    Banner {
                        tone: BannerTone::Warn,
                        title: "Not saved to History.",
                        role: "alert",
                        actions: rsx! {
                            Button {
                                size: ButtonSize::Small,
                                id: "save-retry",
                                onclick: move |_| session.retry_save(),
                                "Try again"
                            }
                        },
                        "The transcript stays on this page until you leave it."
                    }
                }
            },
            audio: ResultAudio::Player { player, peaks },
            wave,
            on_stop,
            status: rsx! {
                if let Some(status) = status {
                    if status.busy() {
                        span { class: "spinner", aria_hidden: "true" }
                    }
                    if matches!(status, RunStatus::Stopped(_)) {
                        Icon { name: IconName::Warning, size: IconSize::Small }
                    }
                    span { class: "eta", "{status}" }
                }
            },
            status_shown: status.is_some(),
            segments,
            export,
            facts: TranscriptionFacts {
                model_label: record.model.label.clone(),
                language_text: language_text(&record),
            },
            more,
            on_more: move |action| match action {
                MoreAction::RunAgain => picking.set(Some(MORE_ID)),
                MoreAction::NewTranscription => session.new_transcription(),
                MoreAction::Rename | MoreAction::Delete => {}
            },
        }
        GlideDriver { input, eta, arrived }
        if let Some(anchor) = picking() {
            ModelPicker {
                anchor,
                selected: record.model.id.clone(),
                on_close: move |()| picking.set(None),
                on_pick: move |entry| session.run_again(entry),
            }
        }
    }
}
