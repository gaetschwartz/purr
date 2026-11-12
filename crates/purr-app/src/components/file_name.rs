use dioxus::prelude::*;

/// Characters kept at the end of a shortened name, so the extension and a "part 2" stay visible.
const TAIL_CHARS: usize = 10;

/// The name split for a middle ellipsis: the head shrinks, the tail always shows.
fn split_name(name: &str) -> (&str, &str) {
    let count = name.chars().count();
    let cut = name
        .char_indices()
        .nth(count.saturating_sub(TAIL_CHARS))
        .map_or(name.len(), |(index, _)| index);
    name.split_at(cut)
}

/// Whether the first strong character of `text` is right-to-left.
fn starts_right_to_left(text: &str) -> bool {
    text.chars().find(|c| c.is_alphabetic()).is_some_and(
        |c| matches!(u32::from(c), 0x0590..=0x08FF | 0xFB1D..=0xFDFF | 0xFE70..=0xFEFF),
    )
}

/// A file name on one line with a middle ellipsis; the full name is in its tooltip.
#[component]
pub fn FileName(#[props(into)] name: String) -> Element {
    let (head, tail) = split_name(&name);
    let dir = if starts_right_to_left(&name) {
        "rtl"
    } else {
        "ltr"
    };
    rsx! {
        span { class: "fname", title: "{name}", dir,
            bdi { class: "hd", "{head}" }
            bdi { class: "tl", "{tail}" }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tail_keeps_the_last_ten_characters() {
        assert_eq!(
            split_name("team-sync-2026-11-10.m4a"),
            ("team-sync-2026", "-11-10.m4a")
        );
        assert_eq!(split_name("jfk.wav"), ("", "jfk.wav"));
        assert_eq!(
            split_name("اجتماع الجزء الثاني.m4a"),
            ("اجتماع الجزء ", "الثاني.m4a")
        );
    }

    #[test]
    fn the_direction_follows_the_first_letter() {
        assert!(starts_right_to_left("2026 اجتماع.m4a"));
        assert!(!starts_right_to_left("team اجتماع.m4a"));
        assert!(!starts_right_to_left("1234.wav"));
    }
}
