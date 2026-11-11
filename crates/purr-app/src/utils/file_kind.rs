fn has_audio_extension(file_name: &str) -> bool {
    let Some((_, extension)) = file_name.rsplit_once('.') else {
        return false;
    };
    matches!(
        extension.to_ascii_lowercase().as_str(),
        "aac"
            | "aif"
            | "aiff"
            | "amr"
            | "flac"
            | "m4a"
            | "m4b"
            | "mka"
            | "mp3"
            | "mp4"
            | "oga"
            | "ogg"
            | "opus"
            | "wav"
            | "weba"
            | "webm"
            | "wma"
    )
}

/// Whether a file can plausibly hold audio, judged by its extension and the MIME type the platform reported.
///
/// A known audio extension wins over the reported type, because desktop webviews report `text/html`
/// for extensions they do not know. An empty type means the platform does not know the file, so the
/// decoder gets to decide.
#[must_use]
pub fn may_contain_audio(content_type: &str, file_name: &str) -> bool {
    if has_audio_extension(file_name) {
        return true;
    }
    let essence = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    essence.is_empty()
        || essence.starts_with("audio/")
        || matches!(
            essence.as_str(),
            "video/mp4"
                | "video/webm"
                | "video/ogg"
                | "application/ogg"
                | "application/octet-stream"
        )
}

/// The MIME type to serve an audio file of this name as.
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

    #[test]
    fn audio_types_are_accepted() {
        assert!(may_contain_audio("audio/mpeg", "talk"));
        assert!(may_contain_audio("audio/ogg; codecs=opus", "talk"));
        assert!(may_contain_audio("AUDIO/WAV", "talk"));
    }

    #[test]
    fn known_audio_extensions_win_over_a_wrong_reported_type() {
        assert!(may_contain_audio(
            "text/html; charset=utf-8",
            "recording.wav"
        ));
        assert!(may_contain_audio("text/html", "Voice Note.MP3"));
        assert!(may_contain_audio("text/html", "memo.flac"));
        assert!(may_contain_audio(
            "text/html",
            "WhatsApp Ptt 2026-11-11 at 07.32.47.ogg"
        ));
    }

    #[test]
    fn a_wrong_type_without_an_audio_extension_is_rejected() {
        assert!(!may_contain_audio("text/html", "page.html"));
        assert!(!may_contain_audio("text/html", "notes"));
        assert!(!may_contain_audio("text/html", "archive.wav.zip"));
    }

    #[test]
    fn containers_that_carry_audio_are_accepted() {
        assert!(may_contain_audio("video/mp4", "clip"));
        assert!(may_contain_audio("video/webm", "clip"));
        assert!(may_contain_audio("application/ogg", "clip"));
    }

    #[test]
    fn unknown_types_are_left_to_the_decoder() {
        assert!(may_contain_audio("", "clip"));
        assert!(may_contain_audio("application/octet-stream", "clip"));
    }

    #[test]
    fn documents_and_images_are_rejected() {
        assert!(!may_contain_audio("text/plain", "notes.txt"));
        assert!(!may_contain_audio("image/png", "photo.png"));
        assert!(!may_contain_audio("application/pdf", "paper.pdf"));
    }
}
