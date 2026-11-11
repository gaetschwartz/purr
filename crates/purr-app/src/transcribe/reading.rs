//! The Reading layout: segments gathered into paragraphs.

use std::ops::Range;

/// Segments per paragraph in the Reading layout.
pub const SEGMENTS_PER_PARAGRAPH: usize = 4;

/// The segment indices of each paragraph, `per` segments at a time.
#[must_use]
pub fn paragraphs(count: usize, per: usize) -> Vec<Range<usize>> {
    let per = per.max(1);
    (0..count)
        .step_by(per)
        .map(|start| start..(start + per).min(count))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_are_grouped_with_a_shorter_last_paragraph() {
        assert_eq!(paragraphs(10, 4), vec![0..4, 4..8, 8..10]);
        assert_eq!(paragraphs(4, 4), vec![0..4]);
        assert_eq!(paragraphs(0, 4), Vec::<Range<usize>>::new());
        assert_eq!(paragraphs(2, 0), vec![0..1, 1..2]);
    }
}
