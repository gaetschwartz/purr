use super::engine_notice::EngineNotice;
use crate::{
    components::{
        button::Button,
        icon::{Icon, IconName, IconSize},
        page::Page,
        progress_bar::{Progress, ProgressBar},
        spinner::Spinner,
    },
    hooks::{use_models::use_models, use_system_probe::use_system_probe},
    model_catalog::state::ModelState,
    platform::IS_WEB,
    transcribe::{
        engine_label::{EngineLabel, planned_browser_engine},
        language_choice::LanguageChoice,
        run::{ModelLoad, Stage},
        session::{RunRecord, use_transcribe_session},
        timing,
    },
    utils::{
        byte_size::ByteSize, clock::now_ms, counted::counted, delay::sleep,
        language_label::language_label,
    },
};
use dioxus::prelude::*;
use std::time::Duration;

const TICK: Duration = Duration::from_millis(500);

/// A transcription in progress: the stages, live numbers and the text as it arrives.
#[component]
pub fn Running(record: RunRecord) -> Element {
    let session = use_transcribe_session();
    let models = use_models();
    let probe = use_system_probe();
    let mut now = use_signal(now_ms);
    use_future(move || async move {
        loop {
            sleep(TICK).await;
            now.set(now_ms());
        }
    });

    let run = &record.run;
    let overview = session.overview(&record.file.id);
    let duration = overview.duration();
    let fraction = duration.and_then(|seconds| run.fraction(seconds));
    let speed = run.live_speed(now());
    let segments = run.transcript.segments();
    let engine = if IS_WEB {
        let gpu = models.gpu.read();
        let engine = run.browser_engine.unwrap_or_else(|| {
            planned_browser_engine(record.accel, &gpu, session.browser_engine())
        });
        EngineLabel::browser(engine, &gpu)
    } else {
        match &run.engine {
            Some(ran) => EngineLabel::ran(ran, &probe.read()),
            None => EngineLabel::planned(
                record.accel,
                session.engine_memory().as_ref(),
                &probe.read(),
            ),
        }
    };
    let file_meta = [
        duration.map(timing::clock),
        Some(record.model.label.clone()),
        Some(engine.to_string()),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(" · ");

    let model_state = models.state(&record.model.id);
    let download = model_state.progress().copied();
    let loading_sub = match (
        &model_state,
        download,
        run.model_load.and_then(ModelLoad::fraction),
    ) {
        (_, Some(progress), _) => progress.fraction().map_or_else(
            || "Downloading…".to_owned(),
            |done| format!("Downloading {}%", (done * 100.0).round()),
        ),
        (ModelState::Verifying, None, _) => "Checking the download…".to_owned(),
        (_, None, Some(done)) => format!("Loading {}%", (done * 100.0).round()),
        (_, None, None) => ByteSize(record.model.size_bytes).to_string(),
    };
    let detected = run.detected_language.as_deref().map(language_label);
    let language_sub = match (&record.language, &detected) {
        (LanguageChoice::Automatic, Some(name)) => name.clone(),
        (LanguageChoice::Automatic, None) if run.stage > Stage::DetectingLanguage => {
            "Not reported".to_owned()
        }
        (LanguageChoice::Automatic, None) => "Listening…".to_owned(),
        (LanguageChoice::Spoken(_), _) => format!("{} · chosen", record.language.label()),
    };
    let transcribing_sub = if run.stage == Stage::Transcribing {
        counted(segments.len(), "segment", "segments")
    } else {
        "Waiting".to_owned()
    };
    let language_stat = match (&record.language, detected) {
        (LanguageChoice::Automatic, Some(name)) => name,
        (LanguageChoice::Automatic, None) => "Listening…".to_owned(),
        (LanguageChoice::Spoken(_), _) => record.language.label(),
    };
    let language_title = match record.language {
        LanguageChoice::Automatic => "Detecting language",
        LanguageChoice::Spoken(_) => "Language",
    };
    let language_key = match record.language {
        LanguageChoice::Automatic => "Detected language",
        LanguageChoice::Spoken(_) => "Language",
    };
    let progress = match (run.stage, fraction) {
        (Stage::Transcribing, Some(done)) => Progress::Determinate(done),
        (Stage::LoadingModel, _) => download
            .and_then(|progress| progress.fraction())
            .or_else(|| run.model_load.and_then(ModelLoad::fraction))
            .map_or(Progress::Indeterminate, Progress::Determinate),
        (Stage::DetectingLanguage | Stage::Transcribing, _) => Progress::Indeterminate,
    };
    let percent_text = fraction.map_or_else(
        || "–".to_owned(),
        |done| format!("{}%", (done * 100.0).round()),
    );

    rsx! {
        Page {
            title: "Transcribing",
            actions: rsx! {
                Button {
                    icon: IconName::Stop,
                    id: "cancel-run",
                    onclick: move |_| session.cancel(),
                    "Cancel"
                }
            },
            EngineNotice { record: record.clone() }
            section { class: "card glass stack", style: "gap:16px", aria_label: "Progress",
                div { style: "min-width:0",
                    div { class: "ellipsis", style: "font-weight:600", "{record.file.name}" }
                    div { class: "faint tnum", style: "font-size:13px", "{file_meta}" }
                }
                ol { class: "stages", aria_label: "Stages",
                    StageItem {
                        index: 0,
                        current: run.stage,
                        stage: Stage::LoadingModel,
                        title: "Loading model",
                        sub: loading_sub,
                    }
                    StageItem {
                        index: 1,
                        current: run.stage,
                        stage: Stage::DetectingLanguage,
                        title: language_title,
                        sub: language_sub,
                    }
                    StageItem {
                        index: 2,
                        current: run.stage,
                        stage: Stage::Transcribing,
                        title: "Transcribing",
                        sub: transcribing_sub,
                    }
                }
                ProgressBar {
                    progress,
                    label: "Transcription progress",
                    value_text: percent_text,
                }
                div { class: "stats",
                    Stat { label: "Elapsed", value: timing::clock(run.elapsed_seconds(now())) }
                    Stat { label: "Speed", value: speed.map_or_else(|| "–".to_owned(), timing::speed_text) }
                    Stat { label: language_key, value: language_stat, small: true }
                }
            }
            section { class: "card glass", aria_label: "Live transcript", style: "padding:12px 10px",
                h2 { class: "t3", style: "padding:4px 10px 8px", "Live transcript" }
                ol { class: "live", style: "margin:0;padding:0", aria_live: "polite",
                    for (index , segment) in segments.iter().enumerate() {
                        li { key: "{index}", class: "seg-row new",
                            span { class: "ts", {timing::clock(segment.start_time)} }
                            span { class: "tx", dir: "auto", "{segment.text.trim()}" }
                        }
                    }
                    if run.stage == Stage::Transcribing {
                        li { class: "seg-row", aria_hidden: "true",
                            span { class: "ts" }
                            span { class: "tx",
                                span { class: "caret" }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn StageItem(index: u8, current: Stage, stage: Stage, title: &'static str, sub: String) -> Element {
    let (class, state) = match current.cmp(&stage) {
        std::cmp::Ordering::Greater => ("stg done", "done"),
        std::cmp::Ordering::Equal => ("stg active", "in progress"),
        std::cmp::Ordering::Less => ("stg", "waiting"),
    };
    rsx! {
        li { class, aria_current: (current == stage).then_some("step"),
            span { class: "dot", aria_hidden: "true",
                match current.cmp(&stage) {
                    std::cmp::Ordering::Greater => rsx! {
                        Icon { name: IconName::Check, size: IconSize::Small }
                    },
                    std::cmp::Ordering::Equal => rsx! {
                        Spinner {}
                    },
                    std::cmp::Ordering::Less => rsx! {
                        span { style: "font-size:11px;font-weight:700", "{index + 1}" }
                    },
                }
            }
            span { style: "min-width:0",
                b { "{title}" }
                span { class: "ss", "{sub}" }
                span { class: "sr", ", {state}" }
            }
        }
    }
}

#[component]
fn Stat(label: &'static str, value: String, #[props(default)] small: bool) -> Element {
    rsx! {
        div { class: "stat",
            div { class: "k", "{label}" }
            div {
                class: "v",
                style: if small { "font-size:17px;line-height:30px" },
                "{value}"
            }
        }
    }
}
