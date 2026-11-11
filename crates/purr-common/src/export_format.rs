//! The file formats a transcript can be exported in.

use serde::{Deserialize, Serialize};

/// A file format the transcript can be exported in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ExportFormat {
    #[default]
    Txt,
    Srt,
    Vtt,
    Json,
}

impl ExportFormat {
    /// Every format, in menu order.
    pub const ALL: [Self; 4] = [Self::Txt, Self::Srt, Self::Vtt, Self::Json];

    /// What the format is, in plain words.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Txt => "Plain text",
            Self::Srt => "Subtitles",
            Self::Vtt => "Web subtitles",
            Self::Json => "Data",
        }
    }

    /// The short name shown next to the label.
    #[must_use]
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::Txt => "TXT",
            Self::Srt => "SRT",
            Self::Vtt => "VTT",
            Self::Json => "JSON",
        }
    }

    /// One sentence on when to pick it.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Txt => "Plain text, one line per segment.",
            Self::Srt => "Subtitles for video players and editors.",
            Self::Vtt => "Subtitles for web video.",
            Self::Json => "Segments with start and end times, for scripts and other apps.",
        }
    }

    /// The file extension, without the dot.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Txt => "txt",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Json => "json",
        }
    }

    /// The media type of the file.
    #[must_use]
    pub const fn content_type(self) -> &'static str {
        match self {
            Self::Txt => "text/plain;charset=utf-8",
            Self::Srt => "application/x-subrip;charset=utf-8",
            Self::Vtt => "text/vtt;charset=utf-8",
            Self::Json => "application/json;charset=utf-8",
        }
    }
}
