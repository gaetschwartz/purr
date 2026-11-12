use crate::platform::playback_source::{self, AudioSource};
use dioxus::{document::Eval, prelude::*};
use serde::{Deserialize, Serialize};

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
  if (!wave || wave.getAttribute('role') !== 'slider' || !Number.isFinite(audio.duration)) return;
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

/// The DOM id of the player's waveform, which takes clicks to seek while it is a slider.
pub const WAVE_ID: &str = "player-wave";

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

    /// The length of the audio, once the player knows it (subscribes the caller).
    #[must_use]
    pub fn length(&self) -> Option<f64> {
        *self.duration.read()
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

    /// Jumps to `position` seconds without starting playback.
    pub fn seek(&self, position: f64) {
        self.send(&Command::Seek {
            position,
            play: false,
        });
    }
}

/// Loads `source` into a hidden audio element of the page, released when the caller unmounts.
pub fn use_audio_player(source: &AudioSource) -> PlayerHandle {
    let source = use_hook(|| playback_source::url(source).map_err(|error| error.to_string()));
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
