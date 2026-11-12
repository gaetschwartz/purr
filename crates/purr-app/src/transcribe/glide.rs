//! The gliding progress fill: a fake-linear motion towards the transcribed position.
//!
//! The fill never passes the truth and never goes back. It moves at the measured rate, catches up
//! when it trails by more than one expected update, and eases into the target when the next
//! update is late. All times are in seconds.

use std::collections::VecDeque;

/// Below this many seconds before an overdue update the fill slows into the target.
pub const TAU_NEAR: f64 = 0.25;
/// The smallest expected update, in seconds of audio, so dense updates keep a usable catch-up.
pub const MIN_GAP: f64 = 0.5;
/// Aims slightly slow so the fill rarely reaches the target and stalls.
pub const MARGIN: f64 = 0.97;
/// How many updates the rate is measured over.
pub const HISTORY: usize = 6;
/// The final glide to the end never takes longer.
pub const FINISH: f64 = 0.6;
/// After an error the fill reaches the true value within this time, then freezes.
pub const STOP: f64 = 0.3;

/// The speed the fill moves at and the size of one expected update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    /// Seconds of audio per wall second.
    pub rate: f64,
    /// Seconds of audio one update is expected to add.
    pub gap_ref: f64,
}

/// One frame: where the fill is after `dt` seconds, never above `target`, never below `shown`.
///
/// `time_left` is the deadline to arrive by, once the run has ended.
#[must_use]
pub fn glide_step(
    shown: f64,
    target: f64,
    rate: f64,
    gap_ref: f64,
    dt: f64,
    time_left: Option<f64>,
) -> f64 {
    let gap = target - shown;
    if gap <= 0.0 {
        return shown;
    }
    if time_left.is_some_and(|left| left <= dt) {
        return target;
    }
    let catch_up = rate * (1.0 + (gap - gap_ref).max(0.0) / gap_ref);
    let eased = catch_up.min(gap / TAU_NEAR);
    let speed = time_left.map_or(eased, |left| eased.max(gap / left.max(0.05)));
    (shown + speed * dt).min(target)
}

/// The rate and expected update size from `(wall, target)` points, oldest first.
#[must_use]
pub fn estimate(history: &[(f64, f64)]) -> Option<Estimate> {
    let (&(first_at, first), &(last_at, last)) = (history.first()?, history.last()?);
    let span = last_at - first_at;
    if history.len() < 2 || span <= 0.0 {
        return None;
    }
    let rate = (last - first) / span;
    let interval = span / (history.len() - 1) as f64;
    Some(Estimate {
        rate: rate * MARGIN,
        gap_ref: MIN_GAP.max(rate * interval),
    })
}

/// The fill's state across frames.
#[derive(Debug, Clone, PartialEq)]
pub struct Glide {
    shown: f64,
    target: f64,
    history: VecDeque<(f64, f64)>,
    estimate: Option<Estimate>,
    deadline: Option<f64>,
    last_frame: Option<f64>,
}

impl Glide {
    /// A fill at `target` at `now`, as when a view opens on a run in progress.
    #[must_use]
    pub fn at(now: f64, target: f64) -> Self {
        Self {
            shown: target,
            target,
            history: VecDeque::from([(now, target)]),
            estimate: None,
            deadline: None,
            last_frame: None,
        }
    }

    /// Where the fill is drawn.
    #[must_use]
    pub const fn shown(&self) -> f64 {
        self.shown
    }

    /// The latest true position.
    #[must_use]
    pub const fn target(&self) -> f64 {
        self.target
    }

    /// Whether the run ended and the fill has arrived at the end value.
    #[must_use]
    pub fn arrived(&self) -> bool {
        self.deadline.is_some() && self.shown >= self.target
    }

    /// A new true position at `now`; positions that do not move forward are ignored.
    pub fn retarget(&mut self, now: f64, target: f64) {
        if target <= self.target {
            return;
        }
        self.target = target;
        self.history.push_back((now, target));
        while self.history.len() > HISTORY {
            self.history.pop_front();
        }
        if let Some(found) = estimate(self.history.make_contiguous()) {
            self.estimate = Some(found);
        }
    }

    /// The run ended at `target`: arrive there within `within` seconds of `now`.
    pub fn finish(&mut self, now: f64, target: f64, within: f64) {
        if self.deadline.is_some() {
            return;
        }
        self.target = self.target.max(target);
        self.deadline = Some(now + within);
    }

    /// Jumps to the target, for reduced motion.
    pub const fn settle(&mut self) {
        self.shown = self.target;
    }

    /// Advances to `now` and returns where the fill is drawn.
    pub fn step(&mut self, now: f64) -> f64 {
        let dt = self.last_frame.map_or(0.0, |last| (now - last).max(0.0));
        self.last_frame = Some(now);
        let Estimate { rate, gap_ref } = self.estimate.unwrap_or(Estimate {
            rate: 0.0,
            gap_ref: MIN_GAP,
        });
        let time_left = self.deadline.map(|deadline| deadline - now);
        self.shown = glide_step(self.shown, self.target, rate, gap_ref, dt, time_left);
        self.shown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: f64 = 1.0 / 30.0;

    #[test]
    fn the_fill_never_passes_the_target_nor_goes_back() {
        let mut shown = 0.0;
        for _ in 0..200 {
            let next = glide_step(shown, 10.0, 3.0, 1.0, FRAME, None);
            assert!(next >= shown);
            assert!(next <= 10.0);
            shown = next;
        }
        assert_eq!(glide_step(12.0, 10.0, 3.0, 1.0, FRAME, None), 12.0);
    }

    #[test]
    fn a_steady_staircase_moves_at_a_constant_speed() {
        let mut glide = Glide::at(0.0, 0.0);
        let mut now = 0.0;
        let mut target = 0.0;
        let mut speeds = Vec::new();
        let mut means = Vec::new();
        for update in 1..=10 {
            let next_update = f64::from(update) * 3.0;
            let (start_at, start) = (now, glide.shown());
            while now + FRAME < next_update {
                let before = glide.shown();
                now += FRAME;
                let after = glide.step(now);
                let overdue_soon = next_update - now < TAU_NEAR;
                if update > 3 && !overdue_soon {
                    speeds.push((after - before) / FRAME);
                }
            }
            if update > 3 {
                means.push((glide.shown() - start) / (now - start_at));
            }
            target += 4.0;
            now = next_update;
            glide.retarget(now, target);
            glide.step(now);
        }
        let expected = 4.0 / 3.0;
        assert!(!speeds.is_empty());
        for mean in means {
            assert!(
                (mean - expected).abs() <= expected * 0.05,
                "a mean of {mean} is not within 5 % of {expected}"
            );
        }
        for speed in speeds {
            assert!(
                (speed - expected).abs() <= expected * 0.2,
                "{speed} is not within 20 % of {expected}"
            );
        }
    }

    #[test]
    fn a_large_gap_closes_within_about_two_updates() {
        let gap_ref = 4.0;
        let mut shown = 0.0;
        let target = 10.0 * gap_ref;
        let mut elapsed = 0.0;
        while target - shown >= gap_ref {
            shown = glide_step(shown, target, 4.0 / 3.0, gap_ref, FRAME, None);
            elapsed += FRAME;
        }
        assert!(elapsed <= 2.5 * 3.0, "took {elapsed} s");
    }

    #[test]
    fn a_deadline_is_met() {
        let mut glide = Glide::at(0.0, 10.0);
        glide.step(0.0);
        glide.finish(0.0, 100.0, FINISH);
        let mut now = 0.0;
        while now < FINISH + FRAME {
            now += FRAME;
            glide.step(now);
        }
        assert_eq!(glide.shown(), 100.0);
        assert!(glide.arrived());
    }

    #[test]
    fn a_huge_frame_lands_exactly_on_the_target() {
        assert_eq!(glide_step(5.0, 40.0, 1.3, 4.0, 10.0, None), 40.0);
    }

    #[test]
    fn the_rate_needs_two_points_and_counts_from_the_start() {
        assert_eq!(estimate(&[(1.0, 0.0)]), None);
        let found = estimate(&[(0.0, 0.0), (3.0, 4.0)]).unwrap();
        assert!((found.rate - 4.0 / 3.0 * MARGIN).abs() < 1e-12);
        assert!((found.gap_ref - 4.0).abs() < 1e-12);
    }

    #[test]
    fn dense_updates_keep_a_minimum_expected_gap() {
        let found = estimate(&[(0.0, 0.0), (0.1, 0.01), (0.2, 0.02)]).unwrap();
        assert_eq!(found.gap_ref, MIN_GAP);
    }

    #[test]
    fn opening_mid_run_starts_where_the_run_is() {
        let mut glide = Glide::at(50.0, 120.0);
        assert_eq!(glide.step(50.5), 120.0);
        glide.retarget(51.0, 100.0);
        assert_eq!(glide.target(), 120.0);
    }
}
