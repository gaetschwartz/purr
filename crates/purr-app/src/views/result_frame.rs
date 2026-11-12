//! The run view frame: banners, the audio card and the transcript card, under a hidden title.
//!
//! A run in progress, its result, a failed run and a saved transcription all render through it,
//! so a run that ends changes attributes in place instead of mounting a new page.

pub mod more_menu;
mod playback_shortcut;
mod transcript_card;

use crate::{
    components::{
        audio_card::{AudioCard, SilentAudioCard, WaveMode},
        audio_player::PlayerHandle,
        export_menu::{ExportSource, copy_export},
        page::{Page, TitleStyle},
        toast::use_toasts,
    },
    shortcuts::{Key, KeyChord, Shortcut, use_shortcut},
};
use dioxus::prelude::*;
use more_menu::{MoreAction, TranscriptionFacts};
use playback_shortcut::PlaybackShortcut;
use purr_common::{export_format::ExportFormat, platform::TranscriptSegment};
use std::rc::Rc;
use transcript_card::{SEARCH_ID, TranscriptCard};

/// Where the view stands; it drives the classes the transitions hang from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewPhase {
    /// Downloading or loading the model.
    Preparing,
    DetectingLanguage,
    Transcribing,
    /// The transcript is final.
    Finished,
    /// The run stopped with an error; the text so far stays.
    Stopped,
}

impl ViewPhase {
    const fn class(self) -> &'static str {
        match self {
            Self::Preparing => "runview waiting",
            Self::DetectingLanguage => "runview detecting",
            Self::Transcribing => "runview",
            Self::Finished => "runview done",
            Self::Stopped => "runview stopped",
        }
    }

    const fn is_live(self) -> bool {
        match self {
            Self::Preparing | Self::DetectingLanguage | Self::Transcribing => true,
            Self::Finished | Self::Stopped => false,
        }
    }
}

/// What to show where the audio card goes.
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

/// The run view: banners, the audio card, then the transcript with its toolbar.
#[component]
pub fn ResultFrame(
    /// The hidden page heading.
    title: String,
    phase: ViewPhase,
    file_name: String,
    duration_seconds: Option<f64>,
    /// Banners above the audio card.
    notice: Option<Element>,
    audio: ResultAudio,
    wave: WaveMode,
    /// Set while the round button stops the run.
    on_stop: Option<EventHandler<()>>,
    /// The phase or time left on the audio card.
    status: Element,
    #[props(default)] status_shown: bool,
    segments: Rc<Vec<TranscriptSegment>>,
    export: ExportSource,
    facts: TranscriptionFacts,
    more: Vec<MoreAction>,
    on_more: EventHandler<MoreAction>,
) -> Element {
    let toasts = use_toasts();
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

    let live = phase.is_live();
    let player = match &audio {
        ResultAudio::Player { player, .. } => Some(*player),
        ResultAudio::Unavailable(_) => None,
    };
    let seekable = player.filter(|_| phase == ViewPhase::Finished);

    rsx! {
        Page {
            title,
            title_style: TitleStyle::Hidden,
            class: phase.class(),
            {notice}
            match audio {
                ResultAudio::Player { player, peaks } => rsx! {
                    AudioCard {
                        player,
                        duration: duration_seconds,
                        peaks,
                        file_name,
                        wave,
                        on_stop,
                        status,
                        status_shown,
                    }
                },
                ResultAudio::Unavailable(note) => rsx! {
                    SilentAudioCard { file_name, note }
                },
            }
            if let Some(player) = player.filter(|_| !live) {
                PlaybackShortcut { player }
            }
            TranscriptCard {
                segments: Rc::clone(&segments),
                player: seekable,
                live,
                export,
                export_shown: !live,
                facts,
                more,
                on_more,
            }
        }
    }
}
