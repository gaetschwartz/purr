//! Finding a query in the transcript and stepping through the matches.

use std::ops::Range;

/// Byte ranges of `text` where `query` appears, ignoring case, without overlaps.
///
/// Characters are compared one by one through their lower-case forms, so the ranges always index
/// the original text even where lower-casing changes a character's length.
#[must_use]
pub fn find(text: &str, query: &str) -> Vec<Range<usize>> {
    let needle: Vec<char> = query.trim().chars().collect();
    if needle.is_empty() {
        return Vec::new();
    }
    let mut found = Vec::new();
    let mut resume_at = 0;
    for (start, _) in text.char_indices() {
        if start < resume_at {
            continue;
        }
        if let Some(end) = match_at(text, start, &needle) {
            found.push(start..end);
            resume_at = end;
        }
    }
    found
}

fn match_at(text: &str, start: usize, needle: &[char]) -> Option<usize> {
    let mut rest = text.get(start..)?.char_indices();
    let mut end = start;
    for wanted in needle {
        let (offset, got) = rest.next()?;
        if !same_letter(got, *wanted) {
            return None;
        }
        end = start + offset + got.len_utf8();
    }
    Some(end)
}

fn same_letter(a: char, b: char) -> bool {
    a == b || a.to_lowercase().eq(b.to_lowercase())
}

/// One match: the segment it is in and where in the segment's text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Match {
    pub segment: usize,
    pub range: Range<usize>,
}

/// Every match of `query` across `texts`, in reading order.
#[must_use]
pub fn find_all<'a>(texts: impl IntoIterator<Item = &'a str>, query: &str) -> Vec<Match> {
    texts
        .into_iter()
        .enumerate()
        .flat_map(|(segment, text)| {
            find(text, query)
                .into_iter()
                .map(move |range| Match { segment, range })
        })
        .collect()
}

/// The match after `current` among `count`, wrapping to the first.
#[must_use]
pub const fn next(current: usize, count: usize) -> usize {
    if count == 0 { 0 } else { (current + 1) % count }
}

/// The match before `current` among `count`, wrapping to the last.
#[must_use]
pub const fn previous(current: usize, count: usize) -> usize {
    if count == 0 {
        0
    } else {
        (current + count - 1) % count
    }
}

/// A piece of a segment's text, marked when it is a match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece<'a> {
    Plain(&'a str),
    /// A match; `index` is its position among all the matches of the transcript.
    Marked {
        text: &'a str,
        index: usize,
    },
}

/// Splits `text` around `ranges`, numbering the marks from `first_index`.
#[must_use]
pub fn pieces<'a>(text: &'a str, ranges: &[Range<usize>], first_index: usize) -> Vec<Piece<'a>> {
    let mut out = Vec::new();
    let mut cursor = 0;
    for (offset, range) in ranges.iter().enumerate() {
        if let Some(plain) = text
            .get(cursor..range.start)
            .filter(|plain| !plain.is_empty())
        {
            out.push(Piece::Plain(plain));
        }
        if let Some(marked) = text.get(range.clone()) {
            out.push(Piece::Marked {
                text: marked,
                index: first_index + offset,
            });
        }
        cursor = range.end;
    }
    if let Some(rest) = text.get(cursor..).filter(|rest| !rest.is_empty()) {
        out.push(Piece::Plain(rest));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_every_match_ignoring_case() {
        assert_eq!(
            find("The theme of the day", "the"),
            vec![0..3, 4..7, 13..16]
        );
    }

    #[test]
    fn a_blank_query_finds_nothing() {
        assert_eq!(find("anything", "  "), Vec::<Range<usize>>::new());
    }

    #[test]
    fn matches_do_not_overlap() {
        assert_eq!(find("aaaa", "aa"), vec![0..2, 2..4]);
    }

    #[test]
    fn ranges_index_the_original_text_when_lowercasing_changes_lengths() {
        let text = "İstanbul STRASSE straße";
        let ranges = find(text, "straße");
        assert_eq!(
            ranges
                .iter()
                .filter_map(|range| text.get(range.clone()))
                .collect::<Vec<_>>(),
            vec!["straße"]
        );
        let accented = find("Éric et éric", "ÉRIC");
        assert_eq!(accented, vec![0..5, 9..14]);
    }

    #[test]
    fn matches_are_numbered_across_segments() {
        let found = find_all(["no match", "a cat", "cat and cat"], "cat");
        assert_eq!(
            found,
            vec![
                Match {
                    segment: 1,
                    range: 2..5
                },
                Match {
                    segment: 2,
                    range: 0..3
                },
                Match {
                    segment: 2,
                    range: 8..11
                },
            ]
        );
    }

    #[test]
    fn navigation_wraps_both_ways() {
        assert_eq!(next(2, 3), 0);
        assert_eq!(next(0, 3), 1);
        assert_eq!(previous(0, 3), 2);
        assert_eq!(previous(2, 3), 1);
        assert_eq!(next(0, 0), 0);
        assert_eq!(previous(0, 0), 0);
    }

    #[test]
    fn pieces_alternate_plain_and_marked_text() {
        assert_eq!(
            pieces("a cat sat", &[Range { start: 2, end: 5 }], 4),
            vec![
                Piece::Plain("a "),
                Piece::Marked {
                    text: "cat",
                    index: 4
                },
                Piece::Plain(" sat"),
            ]
        );
        assert_eq!(pieces("plain", &[], 0), vec![Piece::Plain("plain")]);
    }
}
