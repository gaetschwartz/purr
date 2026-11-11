use super::icon::{Icon, IconName};
use crate::{
    platform::playback_source,
    transcribe::{chosen_file::ChosenFile, playback, timing},
};
use dioxus::{document::Eval, prelude::*};
use serde::{Deserialize, Serialize};
use std::rc::Rc;

const SEEK_STEP_SECONDS: f64 = 5.0;

const PLAYER: &str = r"
const init = await dioxus.recv();
const audio = new Audio();
audio.preload = 'auto';
let objectUrl = null;
const report = () => dioxus.send({
  kind: 'time',
  position: audio.currentTime || 0,
  duration: Number.isFinite(audio.duration) ? audio.duration : null,
  playing: !audio.paused && !audio.ended,
});
const fail = (message) => dioxus.send({ kind: 'error', message: String(message) });
for (const type of ['timeupdate', 'play', 'pause', 'ended', 'loadedmetadata', 'seeked']) audio.addEventListener(type, report);
audio.addEventListener('error', () => fail(audio.error ? `the browser could not play it (media error ${audio.error.code})` : 'the browser could not play it'));
const onClick = (event) => {
  const wave = event.target.closest && event.target.closest('#' + CSS.escape(init.wave));
  if (!wave || !Number.isFinite(audio.duration)) return;
  const box = wave.getBoundingClientRect();
  const fraction = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
  audio.currentTime = fraction * audio.duration;
  report();
};
document.addEventListener('click', onClick);
try {
  const response = await fetch(init.url);
  if (!response.ok) throw new Error(`loading the audio failed with HTTP ${response.status}`);
  objectUrl = URL.createObjectURL(await response.blob());
  audio.src = objectUrl;
} catch (error) {
  fail(error);
}
const play = async () => { try { await audio.play(); } catch (error) { fail(error); } };
for (;;) {
  const command = await dioxus.recv();
  if (command.kind === 'play') await play();
  else if (command.kind === 'pause') audio.pause();
  else if (command.kind === 'seek') {
    audio.currentTime = command.position;
    report();
    if (command.play) await play();
  } else if (command.kind === 'close') {
    audio.pause();
    document.removeEventListener('click', onClick);
    audio.removeAttribute('src');
    audio.load();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    break;
  }
}
";

#[derive(Serialize)]
struct PlayerInit<'a> {
    url: &'a str,
    wave: &'a str,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Command {
    Play,
    Pause,
    Seek { position: f64, play: bool },
    Close,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum PlayerEvent {
    Time {
        position: f64,
        duration: Option<f64>,
        playing: bool,
    },
    Error {
        message: String,
    },
}

/// The DOM id of the player's waveform, which takes clicks to seek.
const WAVE_ID: &str = "player-wave";

/// Plays one file in the page and reports where it is; from [`use_audio_player`]. Cheap to copy.
#[derive(Clone, Copy)]
pub struct PlayerHandle {
    position: Signal<f64>,
    duration: Signal<Option<f64>>,
    playing: Signal<bool>,
    error: Signal<Option<String>>,
    eval: Eval,
}

impl PartialEq for PlayerHandle {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position
    }
}

impl PlayerHandle {
    /// Seconds into the audio (subscribes the caller).
    #[must_use]
    pub fn position(&self) -> f64 {
        *self.position.read()
    }

    #[must_use]
    pub fn is_playing(&self) -> bool {
        *self.playing.read()
    }

    /// Why the audio cannot be played, if it cannot.
    #[must_use]
    pub fn error(&self) -> Option<String> {
        self.error.read().clone()
    }

    fn send(&self, command: &Command) {
        if let Err(error) = self.eval.send(command) {
            let mut failure = self.error;
            failure.set(Some(format!("the player stopped responding ({error})")));
        }
    }

    pub fn toggle(&self) {
        let command = if self.is_playing() {
            Command::Pause
        } else {
            Command::Play
        };
        self.send(&command);
    }

    /// Jumps to `position` seconds and plays from there.
    pub fn play_from(&self, position: f64) {
        self.send(&Command::Seek {
            position,
            play: true,
        });
    }

    fn seek(&self, position: f64) {
        self.send(&Command::Seek {
            position,
            play: false,
        });
    }
}

/// Loads `file` into a hidden audio element of the page, released when the caller unmounts.
pub fn use_audio_player(file: &ChosenFile) -> PlayerHandle {
    let source = use_hook(|| playback_source::url(file).map_err(|error| error.to_string()));
    let handle = use_hook(|| PlayerHandle {
        position: Signal::new(0.0),
        duration: Signal::new(None),
        playing: Signal::new(false),
        error: Signal::new(None),
        eval: document::eval(PLAYER),
    });

    let init = source.clone();
    use_future(move || {
        let init = init.clone();
        async move {
            let PlayerHandle {
                mut position,
                mut duration,
                mut playing,
                mut error,
                mut eval,
            } = handle;
            match &init {
                Ok(url) => {
                    if let Err(failure) = eval.send(PlayerInit { url, wave: WAVE_ID }) {
                        error.set(Some(format!("the player could not start ({failure})")));
                        return;
                    }
                }
                Err(message) => {
                    error.set(Some(message.clone()));
                    return;
                }
            }
            loop {
                match eval.recv::<PlayerEvent>().await {
                    Ok(PlayerEvent::Time {
                        position: at,
                        duration: length,
                        playing: running,
                    }) => {
                        position.set(at);
                        if length.is_some() {
                            duration.set(length);
                        }
                        playing.set(running);
                    }
                    Ok(PlayerEvent::Error { message }) => {
                        tracing::warn!("Playback failed: {message}");
                        error.set(Some(message));
                        playing.set(false);
                    }
                    Err(failure) => {
                        tracing::debug!("The player script ended: {failure}");
                        break;
                    }
                }
            }
        }
    });

    use_drop(move || {
        if let Err(error) = handle.eval.send(Command::Close) {
            tracing::debug!("The player was already gone: {error}");
        }
        if let Ok(url) = &source {
            playback_source::release(url);
        }
    });
    handle
}

/// Play button, waveform (a slider: arrows step 5 s, Home and End jump) and time, for `player`.
#[component]
pub fn AudioPlayer(
    player: PlayerHandle,
    /// The length of the audio from the decoder, used until the player knows it.
    duration: Option<f64>,
    /// Waveform bars between 0 and 1; a flat line while unknown.
    peaks: Option<Rc<Vec<f32>>>,
) -> Element {
    let length = player.duration.read().or(duration).unwrap_or(0.0);
    let position = player.position();
    let playing = player.is_playing();
    let bars: Vec<f32> = peaks.map_or_else(|| vec![0.15; 60], |peaks| peaks.as_ref().clone());
    let played = if length > 0.0 {
        (position / length * bars.len() as f64).floor() as usize
    } else {
        0
    };
    let value_text = format!("{} of {}", timing::clock(position), timing::clock(length));

    rsx! {
        div { class: "player glass", role: "group", aria_label: "Source audio",
            button {
                class: "play",
                r#type: "button",
                id: "play-btn",
                aria_label: if playing { "Pause" } else { "Play" },
                onclick: move |_| player.toggle(),
                Icon { name: if playing { IconName::Pause } else { IconName::Play } }
            }
            div {
                class: "wave",
                id: WAVE_ID,
                role: "slider",
                tabindex: "0",
                aria_label: "Playback position",
                aria_valuemin: "0",
                aria_valuemax: "{length.round()}",
                aria_valuenow: "{position.round()}",
                aria_valuetext: "{value_text}",
                onkeydown: move |event: KeyboardEvent| {
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
                for (index , height) in bars.iter().enumerate() {
                    i {
                        key: "{index}",
                        class: if index < played { "p" },
                        style: "height:{(height.max(0.08) * 100.0).round()}%",
                    }
                }
            }
            span { class: "ptime", "{timing::clock(position)} / {timing::clock(length)}" }
        }
        if let Some(error) = player.error() {
            p { class: "foot", role: "alert", style: "margin:0",
                "The audio can't be played here: {error}. The transcript is unaffected."
            }
        }
    }
}
