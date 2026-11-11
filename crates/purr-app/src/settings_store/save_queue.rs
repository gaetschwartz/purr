//! The coalescing save queue: which settings to write next, and what the last write did.

use super::status::SaveStatus;
use futures::channel::oneshot;
use purr_common::settings::Settings;

/// A settings value with the change number it was submitted under.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    pub revision: u64,
    pub settings: Settings,
}

/// Whether writes may reach the platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteAccess {
    Open,
    /// The stored settings could not be read; writing would destroy them.
    Locked,
}

/// Why a write is being started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartReason {
    /// The writer loop: a snapshot whose write failed waits for a new change or a retry.
    Routine,
    /// The app is closing: one more attempt at whatever is unsaved.
    Closing,
}

/// What finishing a write changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finished {
    Saved,
    /// The write failed; `first` is true when the queue was not already failing.
    Failed {
        error: String,
        first: bool,
    },
}

/// At most one write in flight; while it runs, newer changes collapse into the newest one.
#[derive(Debug)]
pub struct SaveQueue {
    latest: u64,
    saved: u64,
    saved_at: Option<u64>,
    pending: Option<Snapshot>,
    in_flight: Option<u64>,
    held: bool,
    failure: Option<String>,
    access: WriteAccess,
    idle_waiters: Vec<oneshot::Sender<()>>,
}

impl SaveQueue {
    pub const fn new(access: WriteAccess) -> Self {
        Self {
            latest: 0,
            saved: 0,
            saved_at: None,
            pending: None,
            in_flight: None,
            held: false,
            failure: None,
            access,
            idle_waiters: Vec::new(),
        }
    }

    /// Records a change; it replaces any change that has not started writing yet.
    pub fn submit(&mut self, settings: Settings) -> u64 {
        self.latest += 1;
        self.pending = Some(Snapshot {
            revision: self.latest,
            settings,
        });
        self.held = false;
        self.latest
    }

    pub const fn access(&self) -> WriteAccess {
        self.access
    }

    pub const fn set_access(&mut self, access: WriteAccess) {
        self.access = access;
    }

    /// Releases a failed snapshot so the writer loop tries it again.
    pub const fn release_held(&mut self) -> bool {
        let was_held = self.held && self.pending.is_some();
        self.held = false;
        was_held
    }

    /// Takes the snapshot to write now, if a write may start.
    pub fn start(&mut self, reason: StartReason) -> Option<Snapshot> {
        let allowed = match (self.access, reason) {
            (WriteAccess::Locked, StartReason::Routine | StartReason::Closing) => false,
            (WriteAccess::Open, StartReason::Routine) => !self.held,
            (WriteAccess::Open, StartReason::Closing) => true,
        };
        if !allowed || self.in_flight.is_some() {
            return None;
        }
        let snapshot = self.pending.take()?;
        self.in_flight = Some(snapshot.revision);
        Some(snapshot)
    }

    /// Ends the write of `snapshot`, started by [`Self::start`].
    pub fn finish(&mut self, snapshot: Snapshot, result: Result<(), String>, now: u64) -> Finished {
        self.in_flight = None;
        for waiter in self.idle_waiters.drain(..) {
            if waiter.send(()).is_err() {
                tracing::debug!("A settings flush stopped waiting before the write ended");
            }
        }
        match result {
            Ok(()) => {
                self.saved = self.saved.max(snapshot.revision);
                self.saved_at = Some(now);
                self.failure = None;
                Finished::Saved
            }
            Err(error) => {
                let first = self.failure.is_none();
                self.failure = Some(error.clone());
                if self.pending.is_none() {
                    self.pending = Some(snapshot);
                    self.held = true;
                }
                Finished::Failed { error, first }
            }
        }
    }

    /// A receiver that fires when the write in flight ends, or `None` when nothing is writing.
    pub fn wait_for_write(&mut self) -> Option<oneshot::Receiver<()>> {
        self.in_flight?;
        let (sender, receiver) = oneshot::channel();
        self.idle_waiters.push(sender);
        Some(receiver)
    }

    pub const fn is_writing(&self) -> bool {
        self.in_flight.is_some()
    }

    /// Whether a change exists that no write has stored.
    pub const fn has_unsaved(&self) -> bool {
        self.saved < self.latest
    }

    pub fn status(&self) -> SaveStatus {
        let writing = self.in_flight.is_some() || (self.pending.is_some() && !self.held);
        match (self.access, &self.failure, self.saved_at) {
            (WriteAccess::Locked, _, _) => SaveStatus::Paused,
            (WriteAccess::Open, _, _) if writing => SaveStatus::Saving,
            (WriteAccess::Open, Some(error), _) => SaveStatus::Failed {
                error: error.clone(),
            },
            (WriteAccess::Open, None, Some(at_millis)) if !self.has_unsaved() => {
                SaveStatus::Saved { at_millis }
            }
            (WriteAccess::Open, None, Some(_) | None) => SaveStatus::Idle,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_model(name: &str) -> Settings {
        let mut settings = Settings::default();
        name.clone_into(&mut settings.model.model_name);
        settings
    }

    #[test]
    fn changes_made_during_a_write_collapse_into_the_newest() {
        let mut queue = SaveQueue::new(WriteAccess::Open);
        queue.submit(with_model("a"));
        let Some(first) = queue.start(StartReason::Routine) else {
            unreachable!("an idle queue starts the first write")
        };
        assert_eq!(first.revision, 1);
        queue.submit(with_model("b"));
        queue.submit(with_model("c"));
        assert_eq!(queue.start(StartReason::Routine), None);
        assert_eq!(queue.finish(first, Ok(()), 10), Finished::Saved);
        assert_eq!(
            queue.start(StartReason::Routine),
            Some(Snapshot {
                revision: 3,
                settings: with_model("c")
            })
        );
        assert_eq!(queue.status(), SaveStatus::Saving);
    }

    #[test]
    fn saved_shows_only_once_the_newest_change_is_stored() {
        let mut queue = SaveQueue::new(WriteAccess::Open);
        assert_eq!(queue.status(), SaveStatus::Idle);
        queue.submit(with_model("a"));
        let Some(first) = queue.start(StartReason::Routine) else {
            unreachable!("an idle queue starts the first write")
        };
        queue.submit(with_model("b"));
        queue.finish(first, Ok(()), 10);
        assert_eq!(queue.status(), SaveStatus::Saving);
        let Some(second) = queue.start(StartReason::Routine) else {
            unreachable!("the newer change is waiting")
        };
        queue.finish(second, Ok(()), 20);
        assert_eq!(queue.status(), SaveStatus::Saved { at_millis: 20 });
        assert!(!queue.has_unsaved());
    }

    #[test]
    fn a_failed_write_is_held_until_a_change_or_a_retry() {
        let mut queue = SaveQueue::new(WriteAccess::Open);
        queue.submit(with_model("a"));
        let Some(first) = queue.start(StartReason::Routine) else {
            unreachable!("an idle queue starts the first write")
        };
        assert_eq!(
            queue.finish(first, Err("disk full".to_owned()), 10),
            Finished::Failed {
                error: "disk full".to_owned(),
                first: true
            }
        );
        assert_eq!(
            queue.status(),
            SaveStatus::Failed {
                error: "disk full".to_owned()
            }
        );
        assert_eq!(queue.start(StartReason::Routine), None);
        assert!(queue.release_held());
        let Some(retried) = queue.start(StartReason::Routine) else {
            unreachable!("the retry released the held write")
        };
        assert_eq!(retried.revision, 1);
        assert_eq!(
            queue.finish(retried, Err("still full".to_owned()), 20),
            Finished::Failed {
                error: "still full".to_owned(),
                first: false
            }
        );
    }

    #[test]
    fn a_newer_change_replaces_a_failed_one() {
        let mut queue = SaveQueue::new(WriteAccess::Open);
        queue.submit(with_model("a"));
        let Some(first) = queue.start(StartReason::Routine) else {
            unreachable!("an idle queue starts the first write")
        };
        queue.submit(with_model("b"));
        queue.finish(first, Err("busy".to_owned()), 10);
        let next = queue.start(StartReason::Routine);
        assert_eq!(next.map(|s| s.settings), Some(with_model("b")));
    }

    #[test]
    fn closing_retries_a_held_write_once_more() {
        let mut queue = SaveQueue::new(WriteAccess::Open);
        queue.submit(with_model("a"));
        let Some(first) = queue.start(StartReason::Routine) else {
            unreachable!("an idle queue starts the first write")
        };
        queue.finish(first, Err("busy".to_owned()), 10);
        assert_eq!(
            queue.start(StartReason::Closing).map(|s| s.revision),
            Some(1)
        );
    }

    #[test]
    fn a_locked_queue_never_writes() {
        let mut queue = SaveQueue::new(WriteAccess::Locked);
        queue.submit(with_model("a"));
        assert_eq!(queue.start(StartReason::Routine), None);
        assert_eq!(queue.start(StartReason::Closing), None);
        assert_eq!(queue.status(), SaveStatus::Paused);
        queue.set_access(WriteAccess::Open);
        assert_eq!(
            queue.start(StartReason::Routine).map(|s| s.settings),
            Some(with_model("a"))
        );
    }

    #[test]
    fn waiters_wake_when_the_write_ends() {
        let mut queue = SaveQueue::new(WriteAccess::Open);
        assert!(queue.wait_for_write().is_none());
        queue.submit(with_model("a"));
        let Some(first) = queue.start(StartReason::Routine) else {
            unreachable!("an idle queue starts the first write")
        };
        let Some(mut waiter) = queue.wait_for_write() else {
            unreachable!("a write is in flight")
        };
        assert_eq!(waiter.try_recv(), Ok(None));
        queue.finish(first, Ok(()), 10);
        assert_eq!(waiter.try_recv(), Ok(Some(())));
    }
}
