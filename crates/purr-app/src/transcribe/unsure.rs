//! Picking the words the recogniser was least sure of.

use purr_common::{platform::TranscriptSegment, transcript_view::UnsureShare};

/// For each segment, the indices of its words among the least sure `share` of the scored words.
///
/// The share is taken of all the scored words so far, rounded to the nearest word, with no
/// absolute threshold; a word without a score is never picked. Words with equal scores are
/// picked in reading order.
#[must_use]
pub fn unsure_words(segments: &[TranscriptSegment], share: UnsureShare) -> Vec<Vec<usize>> {
    let mut scored: Vec<(u8, usize, usize)> = segments
        .iter()
        .enumerate()
        .flat_map(|(segment, found)| {
            found
                .words
                .iter()
                .enumerate()
                .filter_map(move |(word, entry)| entry.score.map(|score| (score, segment, word)))
        })
        .collect();
    let count = (scored.len() * usize::from(share.percent()) + 50) / 100;
    scored.sort_unstable();
    let mut picked = vec![Vec::new(); segments.len()];
    for &(_, segment, word) in scored.iter().take(count) {
        if let Some(words) = picked.get_mut(segment) {
            words.push(word);
        }
    }
    for words in &mut picked {
        words.sort_unstable();
    }
    picked
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::word::Word;

    fn segment(scores: &[Option<u8>]) -> TranscriptSegment {
        TranscriptSegment {
            text: String::new(),
            start_time: 0.0,
            end_time: 0.0,
            words: scores
                .iter()
                .map(|score| Word {
                    range: 0..0,
                    start_cs: None,
                    end_cs: None,
                    score: *score,
                })
                .collect(),
        }
    }

    #[test]
    fn the_lowest_share_of_the_scored_words_is_picked() {
        let segments = [
            segment(&[Some(200), Some(10), Some(250), Some(240), Some(230)]),
            segment(&[Some(220), Some(210), Some(30), Some(255), Some(245)]),
        ];
        assert_eq!(
            unsure_words(&segments, UnsureShare::new(20)),
            vec![vec![1], vec![2]]
        );
        assert_eq!(
            unsure_words(&segments, UnsureShare::new(30)),
            vec![vec![0, 1], vec![2]]
        );
    }

    #[test]
    fn words_without_a_score_are_neither_ranked_nor_picked() {
        let segments = [segment(&[None, Some(90), None, Some(80), Some(250)])];
        assert_eq!(unsure_words(&segments, UnsureShare::new(30)), vec![vec![3]]);
    }

    #[test]
    fn ties_are_picked_in_reading_order() {
        let segments = [
            segment(&[Some(50), Some(50)]),
            segment(&[Some(50), Some(50)]),
        ];
        assert_eq!(
            unsure_words(&segments, UnsureShare::new(25)),
            vec![vec![0], vec![]]
        );
        assert_eq!(
            unsure_words(&segments, UnsureShare::new(5)),
            vec![Vec::<usize>::new(), Vec::new()]
        );
    }

    #[test]
    fn the_count_rounds_to_the_nearest_word() {
        let scores: Vec<Option<u8>> = (0..10).map(Some).collect();
        let segments = [segment(&scores)];
        assert_eq!(unsure_words(&segments, UnsureShare::new(5)), vec![vec![0]]);
        assert_eq!(unsure_words(&segments, UnsureShare::new(10)), vec![vec![0]]);
        assert_eq!(
            unsure_words(&segments, UnsureShare::new(15)),
            vec![vec![0, 1]]
        );
        let four: Vec<Option<u8>> = (0..4).map(Some).collect();
        assert_eq!(
            unsure_words(&[segment(&four)], UnsureShare::new(10)),
            vec![Vec::<usize>::new()]
        );
    }

    #[test]
    fn a_transcript_without_words_picks_nothing() {
        assert_eq!(
            unsure_words(&[segment(&[]), segment(&[])], UnsureShare::new(30)),
            vec![Vec::<usize>::new(), Vec::new()]
        );
        assert_eq!(
            unsure_words(&[], UnsureShare::new(30)),
            Vec::<Vec<usize>>::new()
        );
    }
}
