use super::{
    audio_player::{PlayerHandle, WAVE_ID},
    file_name::FileName,
    icon::{Icon, IconName},
};
use crate::transcribe::{playback, timing};
use dioxus::prelude::*;
use std::rc::Rc;

const SEEK_STEP_SECONDS: f64 = 5.0;

/// What the waveform of an [`AudioCard`] shows.
#[derive(Debug, Clone, PartialEq)]
pub enum WaveMode {
    /// The audio plays and the wave seeks: arrows step 5 s, Home and End jump.
    Playback,
    /// A run fills the wave; the fill edge is drawn through the `--done` property.
    Progress {
        /// Seconds transcribed, once the platform reports any.
        done: Option<f64>,
        /// The progress in words for screen readers.
        value_text: String,
        /// The run stopped with an error and the fill is frozen.
        stopped: bool,
    },
}

/// The bars of the waveform, one set per layer; re-rendered only when `peaks` change.
#[component]
fn Bars(peaks: Option<Rc<Vec<f32>>>) -> Element {
    let heights: Vec<String> = peaks
        .map_or_else(|| vec![0.15; 60], |peaks| peaks.as_ref().clone())
        .iter()
        .map(|height| format!("height:{}%", (height.max(0.08) * 100.0).round()))
        .collect();
    rsx! {
        for class in ["layer-bars", "layer-fill tx", "layer-fill pl"] {
            span { key: "{class}", class,
                for (index , style) in heights.iter().enumerate() {
                    i { key: "{index}", style: "{style}" }
                }
            }
        }
        span { class: "edge" }
    }
}

/// The audio card: a round button, the waveform and the time on line 1; the file name and a
/// status slot on line 2.
///
/// While `on_stop` is set the button stops the run; otherwise it plays and pauses `player`.
/// Every state renders the same elements, so the button keeps focus when the run ends.
#[component]
pub fn AudioCard(
    player: PlayerHandle,
    /// The length of the audio from the decoder, used until the player knows it.
    duration: Option<f64>,
    /// Waveform bars between 0 and 1; a flat line while unknown.
    peaks: Option<Rc<Vec<f32>>>,
    #[props(into)] file_name: String,
    wave: WaveMode,
    on_stop: Option<EventHandler<()>>,
    /// The phase or the time left, at the end of line 2.
    status: Element,
    #[props(default)] status_shown: bool,
) -> Element {
    let length = player.length().or(duration).unwrap_or(0.0);
    let position = player.position();
    let playing = player.is_playing();
    let played = if length > 0.0 {
        (position / length * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };
    let stopping = on_stop.is_some();
    let (button_class, button_label) = match (stopping, playing) {
        (true, _) => ("play stop", "Cancel transcription"),
        (false, true) => ("play", "Pause"),
        (false, false) => ("play", "Play"),
    };
    let seekable = wave == WaveMode::Playback;
    let (role, label, value_now, value_text, time) = match &wave {
        WaveMode::Playback => (
            "slider",
            "Playback position",
            Some(position),
            format!("{} of {}", timing::clock(position), timing::clock(length)),
            position,
        ),
        WaveMode::Progress {
            done,
            value_text,
            stopped,
        } => (
            "progressbar",
            if *stopped {
                "Transcribed before the error"
            } else {
                "Transcription progress"
            },
            *done,
            value_text.clone(),
            done.unwrap_or(0.0),
        ),
    };

    rsx! {
        div { class: "player glass", role: "group", aria_label: "Source audio",
            button {
                class: button_class,
                r#type: "button",
                id: "play-btn",
                aria_label: button_label,
                title: stopping.then_some("Cancel"),
                onclick: move |_| match on_stop {
                    Some(stop) => stop.call(()),
                    None => player.toggle(),
                },
                span { class: "g play-g",
                    Icon { name: if playing { IconName::Pause } else { IconName::Play } }
                }
                span { class: "g stop-g",
                    Icon { name: IconName::Stop }
                }
            }
            div {
                class: "wave gwave",
                id: WAVE_ID,
                style: "--played:{played:.3}%",
                role,
                tabindex: seekable.then_some("0"),
                aria_label: label,
                aria_valuemin: "0",
                aria_valuemax: "{length.round()}",
                aria_valuenow: value_now.map(|now| now.round().to_string()),
                aria_valuetext: "{value_text}",
                onkeydown: move |event: KeyboardEvent| {
                    if !seekable {
                        return;
                    }
                    let target = match event.key() {
                        Key::Named(NamedKey::ArrowLeft | NamedKey::ArrowDown) => {
                            Some(playback::step(position, -SEEK_STEP_SECONDS, length))
                        }
                        Key::Named(NamedKey::ArrowRight | NamedKey::ArrowUp) => {
                            Some(playback::step(position, SEEK_STEP_SECONDS, length))
                        }
                        Key::Named(NamedKey::Home) => Some(0.0),
                        Key::Named(NamedKey::End) => Some(length),
                        Key::Named(_) | Key::Character(_) => None,
                    };
                    if let Some(target) = target {
                        event.prevent_default();
                        player.seek(target);
                    }
                },
                Bars { peaks }
            }
            span { class: "ptime tnum", "{timing::clock(time)} / {timing::clock(length)}" }
            div { class: "pfile",
                FileName { name: file_name }
                span { class: "pstat",
                    span { class: "run-st", "data-off": (!status_shown).then_some("true"), {status} }
                }
            }
        }
        if let Some(error) = player.error() {
            p { class: "foot", role: "alert", style: "margin:0",
                "The audio can't be played here: {error}."
            }
        }
    }
}

/// The audio card of a transcription whose audio cannot be played: the file name and why.
#[component]
pub fn SilentAudioCard(#[props(into)] file_name: String, #[props(into)] note: String) -> Element {
    rsx! {
        div { class: "player silent glass", role: "group", aria_label: "Source audio",
            div { class: "pfile",
                FileName { name: file_name }
            }
            p { class: "foot", role: "note", style: "margin:0", "{note}" }
        }
    }
}
