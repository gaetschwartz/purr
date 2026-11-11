//! The transcript as plain text, SubRip, WebVTT or JSON.

use crate::utils::file_kind::file_stem;
use purr_common::{export_format::ExportFormat, platform::TranscriptSegment};
use serde::Serialize;

/// The name to save the export of `audio_file_name` under.
#[must_use]
pub fn export_file_name(format: ExportFormat, audio_file_name: &str) -> String {
    format!("{}.{}", file_stem(audio_file_name), format.extension())
}

/// What an export describes besides the segments.
#[derive(Debug, Clone, Copy)]
pub struct ExportDocument<'a> {
    pub file_name: &'a str,
    pub model: &'a str,
    /// The Whisper code of the spoken language, when known.
    pub language: Option<&'a str>,
    pub duration_seconds: f64,
    pub segments: &'a [TranscriptSegment],
}

/// Renders `document` in `format`.
pub fn render(format: ExportFormat, document: &ExportDocument<'_>) -> serde_json::Result<String> {
    Ok(match format {
        ExportFormat::Txt => plain_text(document.segments),
        ExportFormat::Srt => subrip(document.segments),
        ExportFormat::Vtt => webvtt(document.segments),
        ExportFormat::Json => json(document)?,
    })
}

/// One trimmed line per segment.
#[must_use]
pub fn plain_text(segments: &[TranscriptSegment]) -> String {
    segments
        .iter()
        .map(|segment| segment.text.trim())
        .collect::<Vec<_>>()
        .join("\n")
}

fn subrip(segments: &[TranscriptSegment]) -> String {
    segments
        .iter()
        .enumerate()
        .map(|(index, segment)| {
            format!(
                "{}\n{} --> {}\n{}\n",
                index + 1,
                cue_time(segment.start_time, ','),
                cue_time(cue_end(segment), ','),
                segment.text.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn webvtt(segments: &[TranscriptSegment]) -> String {
    let cues = segments
        .iter()
        .map(|segment| {
            format!(
                "{} --> {}\n{}\n",
                cue_time(segment.start_time, '.'),
                cue_time(cue_end(segment), '.'),
                segment.text.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("WEBVTT\n\n{cues}")
}

#[derive(Serialize)]
struct JsonDocument<'a> {
    file: &'a str,
    language: Option<&'a str>,
    model: &'a str,
    duration: f64,
    segments: Vec<JsonSegment<'a>>,
}

#[derive(Serialize)]
struct JsonSegment<'a> {
    start: f64,
    end: f64,
    text: &'a str,
}

fn json(document: &ExportDocument<'_>) -> serde_json::Result<String> {
    serde_json::to_string_pretty(&JsonDocument {
        file: document.file_name,
        language: document.language,
        model: document.model,
        duration: document.duration_seconds,
        segments: document
            .segments
            .iter()
            .map(|segment| JsonSegment {
                start: segment.start_time,
                end: cue_end(segment),
                text: segment.text.trim(),
            })
            .collect(),
    })
}

/// Cue ends never come before their start.
const fn cue_end(segment: &TranscriptSegment) -> f64 {
    segment.end_time.max(segment.start_time)
}

fn cue_time(seconds: f64, fraction_separator: char) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}{fraction_separator}{:03}",
        total_ms / 3_600_000,
        (total_ms / 60_000) % 60,
        (total_ms / 1000) % 60,
        total_ms % 1000
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(text: &str, start: f64, end: f64) -> TranscriptSegment {
        TranscriptSegment {
            text: text.to_owned(),
            start_time: start,
            end_time: end,
        }
    }

    fn sample() -> Vec<TranscriptSegment> {
        vec![
            segment(" Bonjour", 0.0, 1.5),
            segment(" tout le monde", 3_725.25, 3_727.0),
        ]
    }

    fn document(segments: &[TranscriptSegment]) -> ExportDocument<'_> {
        ExportDocument {
            file_name: "talk.ogg",
            model: "base",
            language: Some("fr"),
            duration_seconds: 3_727.5,
            segments,
        }
    }

    #[test]
    fn text_has_one_trimmed_line_per_segment() -> serde_json::Result<()> {
        let segments = sample();
        assert_eq!(
            render(ExportFormat::Txt, &document(&segments))?,
            "Bonjour\ntout le monde"
        );
        Ok(())
    }

    #[test]
    fn srt_numbers_cues_and_uses_commas() -> serde_json::Result<()> {
        let segments = sample();
        assert_eq!(
            render(ExportFormat::Srt, &document(&segments))?,
            "1\n00:00:00,000 --> 00:00:01,500\nBonjour\n\n2\n01:02:05,250 --> 01:02:07,000\ntout le monde\n"
        );
        Ok(())
    }

    #[test]
    fn vtt_has_a_header_and_uses_dots() -> serde_json::Result<()> {
        let segments = sample();
        assert_eq!(
            render(ExportFormat::Vtt, &document(&segments))?,
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nBonjour\n\n01:02:05.250 --> 01:02:07.000\ntout le monde\n"
        );
        Ok(())
    }

    #[test]
    fn a_cue_never_ends_before_it_starts() -> serde_json::Result<()> {
        let segments = vec![segment("x", 4.0, 0.0)];
        assert_eq!(
            render(ExportFormat::Srt, &document(&segments))?,
            "1\n00:00:04,000 --> 00:00:04,000\nx\n"
        );
        Ok(())
    }

    #[test]
    fn json_carries_the_details_and_the_segments() -> serde_json::Result<()> {
        let segments = vec![segment(" Bonjour ", 0.0, 1.5)];
        let rendered = render(ExportFormat::Json, &document(&segments))?;
        let value: serde_json::Value = serde_json::from_str(&rendered)?;
        assert_eq!(
            value,
            serde_json::json!({
                "file": "talk.ogg",
                "language": "fr",
                "model": "base",
                "duration": 3727.5,
                "segments": [{ "start": 0.0, "end": 1.5, "text": "Bonjour" }]
            })
        );
        Ok(())
    }

    #[test]
    fn export_names_replace_the_audio_extension() {
        assert_eq!(
            export_file_name(ExportFormat::Vtt, "WhatsApp Ptt 2026.ogg"),
            "WhatsApp Ptt 2026.vtt"
        );
        assert_eq!(
            export_file_name(ExportFormat::Json, ".ogg"),
            "transcript.json"
        );
    }
}
