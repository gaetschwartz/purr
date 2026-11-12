//! The result page of a transcription: details, the player, the transcript and its exports.
//!
//! A fresh run and a saved transcription both render through it.

mod playback_shortcut;
mod transcript_card;

use crate::{
    components::{
        audio_player::{AudioPlayer, PlayerHandle},
        button::Button,
        export_menu::{ExportMenu, ExportSource, copy_export},
        icon::{Icon, IconName},
        page::Page,
        toast::use_toasts,
    },
    shortcuts::{Key, KeyChord, Shortcut, use_shortcut},
    transcribe::timing,
    utils::primary_key::use_primary_key_label,
};
use dioxus::prelude::*;
use playback_shortcut::PlaybackShortcut;
use purr_common::{export_format::ExportFormat, platform::TranscriptSegment};
use std::rc::Rc;
use transcript_card::{SEARCH_ID, TranscriptCard};

/// What the page says about the transcription.
#[derive(Debug, Clone, PartialEq)]
pub struct ResultDetails {
    pub duration_seconds: f64,
    pub model_label: String,
    pub language_text: String,
    pub took_seconds: f64,
    /// The hardware the run used, when the platform reported it.
    pub engine: Option<String>,
}

/// What to show where the player goes.
#[derive(Clone, PartialEq)]
pub enum ResultAudio {
    /// The audio plays; `peaks` draw the waveform.
    Player {
        player: PlayerHandle,
        peaks: Option<Rc<Vec<f32>>>,
    },
    /// The audio cannot be played; the note says why.
    Unavailable(String),
}

/// The page frame of a result: title, actions, details, audio and transcript.
#[component]
pub fn ResultFrame(
    title: String,
    eyebrow: Element,
    /// Buttons that come before Copy and Export.
    actions: Element,
    details: ResultDetails,
    /// A last detail, such as where the transcription is kept.
    keeping: Element,
    /// A banner above the details.
    notice: Option<Element>,
    segments: Rc<Vec<TranscriptSegment>>,
    export: ExportSource,
    audio: ResultAudio,
) -> Element {
    let toasts = use_toasts();
    let primary_key = use_primary_key_label();

    let copy_source = export.clone();
    use_shortcut(
        Shortcut::new(
            KeyChord::primary(Key::Char('c')).with_shift(),
            "Copy the transcript",
        )
        .in_inputs(),
        move || copy_export(&copy_source, ExportFormat::Txt, toasts),
    );
    use_shortcut(
        Shortcut::new(KeyChord::primary(Key::Char('f')), "Search the transcript").in_inputs(),
        move || {
            document::eval(&format!("document.getElementById({SEARCH_ID:?})?.focus()"));
        },
    );

    let speed = timing::speed(details.duration_seconds, details.took_seconds)
        .map(|speed| format!(" · {} real time", timing::speed_text(speed)))
        .unwrap_or_default();
    let engine = details
        .engine
        .as_ref()
        .map(|engine| format!(" · {engine}"))
        .unwrap_or_default();
    let copy_now = export.clone();
    let player = match &audio {
        ResultAudio::Player { player, .. } => Some(*player),
        ResultAudio::Unavailable(_) => None,
    };
    let hint = if player.is_some() {
        format!(
            "Click any line to play from there. Space plays or pauses. {primary_key}+F searches, {primary_key}+Shift+C copies everything."
        )
    } else {
        format!("{primary_key}+F searches, {primary_key}+Shift+C copies everything.")
    };

    rsx! {
        Page {
            title,
            file_title: true,
            eyebrow,
            actions: rsx! {
                div { class: "row wrap",
                    {actions}
                    Button {
                        icon: IconName::Copy,
                        id: "copy-btn",
                        disabled: segments.is_empty(),
                        onclick: move |_| copy_export(&copy_now, ExportFormat::Txt, toasts),
                        "Copy"
                    }
                    ExportMenu { id: "export-btn", source: export }
                }
            },
            {notice}
            div { class: "metas tnum", aria_label: "Details",
                span {
                    Icon { name: IconName::Clock }
                    {timing::clock(details.duration_seconds)}
                }
                span {
                    Icon { name: IconName::Cube }
                    "{details.model_label}"
                }
                span {
                    Icon { name: IconName::Globe }
                    "{details.language_text}"
                }
                span {
                    Icon { name: IconName::Bolt }
                    "Took {timing::took(details.took_seconds)}{speed}{engine}"
                }
                {keeping}
            }
            match audio {
                ResultAudio::Player { player, peaks } => rsx! {
                    AudioPlayer { player, duration: Some(details.duration_seconds), peaks }
                },
                ResultAudio::Unavailable(note) => rsx! {
                    p {
                        class: "foot",
                        role: "note",
                        style: "margin:0;display:flex;gap:8px;align-items:flex-start",
                        Icon { name: IconName::Info }
                        span { "{note}" }
                    }
                },
            }
            if let Some(player) = player {
                PlaybackShortcut { player }
            }
            TranscriptCard { segments: Rc::clone(&segments), player }
            p { class: "foot", style: "text-align:center", "{hint}" }
        }
    }
}
