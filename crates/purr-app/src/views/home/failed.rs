use crate::{
    components::{
        button::{Button, ButtonVariant},
        icon::{Icon, IconName, IconSize},
        model_picker::ModelPicker,
        page::Page,
    },
    main_app::Route,
    transcribe::{
        run::{RunFailure, Stage},
        session::{RunRecord, use_transcribe_session},
        timing,
    },
};
use dioxus::prelude::*;

/// A run that stopped with an error: what happened, how to fix it, and the text kept so far.
#[component]
pub fn Failed(record: RunRecord, failure: RunFailure) -> Element {
    let session = use_transcribe_session();
    let navigator = use_navigator();
    let mut picking = use_signal(|| false);
    let segments = record.run.transcript.segments();
    let reached = record.run.transcribed_seconds();
    let duration = session.overview(&record.file.id).duration();

    let heading = match (record.run.stage, duration) {
        (Stage::Transcribing, Some(total)) if reached > 0.0 => format!(
            "Purr stopped at {} of {}",
            timing::clock(reached),
            timing::clock(total)
        ),
        (Stage::LoadingModel, _) => format!("{} couldn't be loaded", record.model.label),
        (Stage::DetectingLanguage | Stage::Transcribing, _) => {
            format!("{} couldn't be transcribed", record.file.name)
        }
    };

    rsx! {
        Page { title: "Transcription failed",
            section { class: "card glass stack", style: "gap:16px", role: "alert",
                div { class: "row", style: "align-items:flex-start",
                    span { class: "fileicon bad",
                        Icon { name: IconName::Warning, size: IconSize::Large }
                    }
                    div { class: "grow",
                        h2 { class: "t2", "{heading}" }
                        p { class: "muted", style: "margin:4px 0 0", "{failure.message}" }
                        if let Some(hint) = &failure.hint {
                            p { class: "muted", style: "margin:4px 0 0", "{hint}" }
                        }
                        if !segments.is_empty() {
                            p { class: "muted", style: "margin:4px 0 0",
                                "The text recognised so far is kept below."
                            }
                        }
                    }
                }
                div { class: "row wrap",
                    Button {
                        variant: ButtonVariant::Primary,
                        icon: IconName::Retry,
                        id: "retry-run",
                        onclick: move |_| session.retry(),
                        "Try again"
                    }
                    Button {
                        id: "fail-model",
                        aria_haspopup: "dialog",
                        onclick: move |_| picking.toggle(),
                        "Use another model"
                    }
                    Button {
                        onclick: move |_| session.new_transcription(),
                        "Choose another file"
                    }
                    Button {
                        variant: ButtonVariant::Plain,
                        onclick: move |_| {
                            navigator.push(Route::Logs {});
                        },
                        "View logs"
                    }
                }
            }
            if !segments.is_empty() {
                section { class: "card glass", style: "padding:12px 10px", aria_label: "Recognised before the error",
                    h2 { class: "t3", style: "padding:4px 10px 8px", "Recognised before the error" }
                    ol { class: "live", style: "margin:0;padding:0",
                        for (index , segment) in segments.iter().enumerate() {
                            li { key: "{index}", class: "seg-row",
                                span { class: "ts", {timing::clock(segment.start_time)} }
                                span { class: "tx", "{segment.text.trim()}" }
                            }
                        }
                    }
                }
            }
        }
        if picking() {
            ModelPicker {
                anchor: "fail-model",
                selected: record.model.id,
                on_close: move |()| picking.set(false),
                on_pick: move |entry| session.run_again(entry),
            }
        }
    }
}
