use crate::{
    components::audio_player::PlayerHandle,
    transcribe::row_text::{WordSpan, pieces},
};
use dioxus::prelude::*;
use std::ops::Range;

/// The text of one row with its search matches highlighted and its unsure words underlined.
///
/// With `seek`, a click on a word that has a time plays the audio from there. Pointer only: the
/// keyboard seeks by the row's timestamp.
#[component]
pub fn MarkedText(
    text: String,
    #[props(default)] words: Vec<WordSpan>,
    #[props(default)] unsure: Vec<usize>,
    matches: (Vec<Range<usize>>, usize),
    current: Option<usize>,
    seek: Option<PlayerHandle>,
) -> Element {
    let (ranges, first) = matches;
    rsx! {
        for (position , piece) in pieces(&text, &words, &unsure, &ranges, first).into_iter().enumerate() {
            {
                let is_unsure = piece.word.as_ref().is_some_and(|word| word.unsure);
                let label = piece.word.as_ref().is_some_and(|word| word.unsure && word.last);
                let target = seek.zip(piece.word.as_ref().and_then(|word| word.start));
                let class = match (target.is_some(), is_unsure) {
                    (true, true) => Some("wd unsure"),
                    (true, false) => Some("wd"),
                    (false, true) => Some("unsure"),
                    (false, false) => None,
                };
                let inner = rsx! {
                    if let Some(index) = piece.found {
                        mark { class: if current == Some(index) { "cur" }, "{piece.text}" }
                    } else {
                        "{piece.text}"
                    }
                    if label {
                        span { class: "sr", " (unsure)" }
                    }
                };
                match target {
                    Some((player, start)) => rsx! {
                        span {
                            key: "{position}",
                            class,
                            onclick: move |_| player.play_from(start),
                            {inner}
                        }
                    },
                    None => rsx! {
                        span { key: "{position}", class, {inner} }
                    },
                }
            }
        }
    }
}
