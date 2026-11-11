use crate::{
    components::audio_player::PlayerHandle,
    shortcuts::{Key, KeyChord, Shortcut, use_shortcut},
};
use dioxus::prelude::*;

/// Space plays or pauses the audio, except on a button, which keeps Space for itself.
#[component]
pub fn PlaybackShortcut(player: PlayerHandle) -> Element {
    use_shortcut(
        Shortcut::new(KeyChord::key(Key::Space), "Play or pause").off_controls(),
        move || player.toggle(),
    );
    rsx! {}
}
