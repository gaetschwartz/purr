/// How often the page checks for new entries, in milliseconds.
pub const POLL_MS: u64 = 500;

/// How many checks pass between two announcements to screen readers.
const TICKS_PER_ANNOUNCEMENT: u32 = 20;

/// Sums up the entries that arrive and releases a short message now and then, so a screen reader
/// hears about the flow without reading every row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Announcer {
    pending: usize,
    ticks: u32,
}

impl Announcer {
    /// Records one check that found `new_entries` entries; returns a message when it is time.
    pub fn tick(&mut self, new_entries: usize) -> Option<String> {
        self.pending = self.pending.saturating_add(new_entries);
        self.ticks = self.ticks.saturating_add(1);
        if self.ticks < TICKS_PER_ANNOUNCEMENT {
            return None;
        }
        self.ticks = 0;
        let count = std::mem::take(&mut self.pending);
        match count {
            0 => None,
            1 => Some("1 new entry".to_owned()),
            many => Some(format!("{many} new entries")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_are_summed_up_until_the_interval_has_passed() {
        let mut announcer = Announcer::default();
        for _ in 0..TICKS_PER_ANNOUNCEMENT - 1 {
            assert_eq!(announcer.tick(3), None);
        }
        assert_eq!(
            announcer.tick(3),
            Some(format!("{} new entries", 3 * TICKS_PER_ANNOUNCEMENT))
        );
    }

    #[test]
    fn a_quiet_interval_announces_nothing() {
        let mut announcer = Announcer::default();
        for _ in 0..TICKS_PER_ANNOUNCEMENT * 3 {
            assert_eq!(announcer.tick(0), None);
        }
    }

    #[test]
    fn the_count_starts_over_after_each_announcement() {
        let mut announcer = Announcer::default();
        assert_eq!(announcer.tick(1), None);
        for _ in 0..TICKS_PER_ANNOUNCEMENT - 2 {
            assert_eq!(announcer.tick(0), None);
        }
        assert_eq!(announcer.tick(0), Some("1 new entry".to_owned()));
        for _ in 0..TICKS_PER_ANNOUNCEMENT - 1 {
            assert_eq!(announcer.tick(0), None);
        }
        assert_eq!(announcer.tick(2), Some("2 new entries".to_owned()));
    }
}
