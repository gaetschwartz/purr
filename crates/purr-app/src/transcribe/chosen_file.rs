//! The audio file picked for transcription and why a pick can fail.

use crate::utils::byte_size::ByteSize;
use bytes::Bytes;
use purr_common::platform::FileId;
use std::{path::PathBuf, rc::Rc};

/// Bars in the waveform: enough for a wide player, few enough to stay light.
pub const WAVEFORM_BARS: usize = 120;

/// A file the platform stored for transcription, with its bytes kept for playback.
#[derive(Debug, Clone)]
pub struct ChosenFile {
    pub id: FileId,
    pub name: String,
    pub size: u64,
    pub content_type: Option<String>,
    pub bytes: Bytes,
    /// Where the file was picked from on the computer, when the platform says.
    pub source_path: Option<PathBuf>,
}

impl PartialEq for ChosenFile {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

/// What the platform's decoder revealed of the file: its length and a waveform.
#[derive(Debug, Clone, PartialEq)]
pub enum OverviewState {
    Reading,
    Ready {
        duration_seconds: f64,
        peaks: Rc<Vec<f32>>,
    },
    /// The audio could not be decoded; the detail is the decoder's own words.
    Undecodable(String),
}

impl OverviewState {
    /// The length of the audio, once known.
    #[must_use]
    pub const fn duration(&self) -> Option<f64> {
        match self {
            Self::Ready {
                duration_seconds, ..
            } => Some(*duration_seconds),
            Self::Reading | Self::Undecodable(_) => None,
        }
    }
}

/// Why a picked file was not accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileProblem {
    Empty {
        name: String,
    },
    TooLarge {
        name: String,
        size: u64,
        limit: u64,
    },
    /// The browser or webview could not read the file.
    Unreadable {
        name: String,
        detail: String,
    },
    /// The platform refused to store it.
    Rejected {
        name: String,
        detail: String,
    },
}

impl FileProblem {
    /// The banner title.
    #[must_use]
    pub const fn title(&self) -> &'static str {
        match self {
            Self::Empty { .. } => "This file is empty",
            Self::TooLarge { .. } => "This file is too large",
            Self::Unreadable { .. } => "Purr couldn't read this file",
            Self::Rejected { .. } => "Purr couldn't keep this file",
        }
    }

    /// What happened and what to do, in a sentence or two.
    #[must_use]
    pub fn message(&self) -> String {
        match self {
            Self::Empty { name } => format!(
                "{name} is 0 bytes long, so there's nothing to transcribe. Check that the recording finished saving."
            ),
            Self::TooLarge { name, size, limit } => format!(
                "{name} is {}. Purr accepts files up to {} here. Convert it to M4A or MP3, which is usually much smaller, or split it into parts.",
                ByteSize(*size),
                ByteSize(*limit)
            ),
            Self::Unreadable { name, detail } => {
                format!(
                    "{name} could not be read: {detail}. Check that it is still there and try again."
                )
            }
            Self::Rejected { name, detail } => {
                format!("{name} could not be stored for transcription: {detail}.")
            }
        }
    }
}
