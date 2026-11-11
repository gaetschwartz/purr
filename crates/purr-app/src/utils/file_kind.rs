/// Whether a MIME type reported by the browser can plausibly hold audio.
///
/// An empty type means the browser does not know the file, so the decoder gets to decide.
#[must_use]
pub fn may_contain_audio(content_type: &str) -> bool {
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
    fn stem_falls_back_when_empty() {
        assert_eq!(file_stem(""), "transcript");
        assert_eq!(file_stem(".ogg"), "transcript");
    }

    #[test]
    fn audio_types_are_accepted() {
        assert!(may_contain_audio("audio/mpeg"));
        assert!(may_contain_audio("audio/ogg; codecs=opus"));
        assert!(may_contain_audio("AUDIO/WAV"));
    }

    #[test]
    fn containers_that_carry_audio_are_accepted() {
        assert!(may_contain_audio("video/mp4"));
        assert!(may_contain_audio("video/webm"));
        assert!(may_contain_audio("application/ogg"));
    }

    #[test]
    fn unknown_types_are_left_to_the_decoder() {
        assert!(may_contain_audio(""));
        assert!(may_contain_audio("application/octet-stream"));
    }

    #[test]
    fn documents_and_images_are_rejected() {
        assert!(!may_contain_audio("text/plain"));
        assert!(!may_contain_audio("image/png"));
        assert!(!may_contain_audio("application/pdf"));
    }
}
