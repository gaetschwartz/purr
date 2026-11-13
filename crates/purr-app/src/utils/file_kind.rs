/// The MIME type to serve an audio file of this name as.
#[cfg(any(test, all(feature = "desktop", not(purr_web))))]
#[must_use]
pub fn audio_mime(file_name: &str) -> &'static str {
    let extension = file_name
        .rsplit_once('.')
        .map(|(_, extension)| extension.to_ascii_lowercase());
    match extension.as_deref() {
        Some("mp3") => "audio/mpeg",
        Some("wav") => "audio/wav",
        Some("flac") => "audio/flac",
        Some("ogg" | "oga" | "opus") => "audio/ogg",
        Some("m4a" | "m4b" | "mp4") => "audio/mp4",
        Some("aac") => "audio/aac",
        Some("webm" | "weba") => "audio/webm",
        Some("aif" | "aiff") => "audio/aiff",
        Some("mkv") => "video/x-matroska",
        Some("mov") => "video/quicktime",
        Some("m4v") => "video/mp4",
        Some("avi") => "video/x-msvideo",
        Some("wmv" | "asf") => "video/x-ms-wmv",
        Some("flv") => "video/x-flv",
        Some("mpg" | "mpeg") => "video/mpeg",
        Some("ts" | "mts" | "m2ts") => "video/mp2t",
        Some("3gp") => "video/3gpp",
        Some("3g2") => "video/3gpp2",
        Some("ogv") => "video/ogg",
        Some(_) | None => "application/octet-stream",
    }
}

/// The short type shown next to a file: its extension in capitals, or `Audio` without one.
#[must_use]
pub fn kind_label(file_name: &str) -> String {
    file_name
        .rsplit_once('.')
        .map(|(_, extension)| extension.trim())
        .filter(|extension| !extension.is_empty() && extension.len() <= 5)
        .map_or_else(|| "Audio".to_owned(), str::to_ascii_uppercase)
}

/// The file name without its extension, or `transcript` when nothing usable is left.
#[must_use]
pub fn file_stem(file_name: &str) -> String {
    let stem = file_name
        .rsplit_once('.')
        .map_or(file_name, |(stem, _extension)| stem)
        .trim();
    if stem.is_empty() {
        "transcript".to_owned()
    } else {
        stem.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stem_drops_only_the_last_extension() {
        assert_eq!(
            file_stem("WhatsApp Ptt 2026-11-11.ogg"),
            "WhatsApp Ptt 2026-11-11"
        );
        assert_eq!(file_stem("talk.final.mp3"), "talk.final");
    }

    #[test]
    fn audio_is_served_under_its_own_type() {
        assert_eq!(audio_mime("memo.OGG"), "audio/ogg");
        assert_eq!(audio_mime("talk.m4a"), "audio/mp4");
        assert_eq!(audio_mime("mystery"), "application/octet-stream");
    }

    #[test]
    fn video_containers_are_served_under_a_video_type() {
        assert_eq!(audio_mime("clip.mkv"), "video/x-matroska");
        assert_eq!(audio_mime("clip.MOV"), "video/quicktime");
        assert_eq!(audio_mime("clip.ts"), "video/mp2t");
    }

    #[test]
    fn kind_is_the_upper_case_extension() {
        assert_eq!(kind_label("memo.ogg"), "OGG");
        assert_eq!(kind_label("talk.final.m4a"), "M4A");
        assert_eq!(kind_label("recording"), "Audio");
        assert_eq!(kind_label("weird.something-long"), "Audio");
    }

    #[test]
    fn stem_falls_back_when_empty() {
        assert_eq!(file_stem(""), "transcript");
        assert_eq!(file_stem(".ogg"), "transcript");
    }
}
