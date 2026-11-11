//! Running the writes the [`SaveQueue`] hands out, one at a time.

use super::save_queue::{Finished, SaveQueue, StartReason, WriteAccess};
use futures::FutureExt as _;
use purr_common::settings::Settings;
use std::{cell::RefCell, future::Future};

/// How a flush before the app closes ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlushOutcome {
    NothingToSave,
    Saved,
    Failed(String),
    /// The stored settings were unreadable, so the in-memory changes were not written over them.
    Locked,
    /// A write was still running and cannot be waited for here.
    Busy,
}

/// Writes queued changes until none is left, telling `on_change` after each start and finish.
pub async fn drain<S, F>(
    queue: &RefCell<SaveQueue>,
    save: &S,
    now: impl Fn() -> u64,
    mut on_change: impl FnMut(Option<Finished>),
) where
    S: Fn(Settings) -> F,
    F: Future<Output = Result<(), String>>,
{
    loop {
        let Some(snapshot) = queue.borrow_mut().start(StartReason::Routine) else {
            return;
        };
        on_change(None);
        let result = save(snapshot.settings.clone()).await;
        let finished = queue.borrow_mut().finish(snapshot, result, now());
        on_change(Some(finished));
    }
}

/// Waits for the write in flight, then writes whatever is still unsaved, failed writes included.
pub async fn flush<S, F>(
    queue: &RefCell<SaveQueue>,
    save: &S,
    now: impl Fn() -> u64,
) -> FlushOutcome
where
    S: Fn(Settings) -> F,
    F: Future<Output = Result<(), String>>,
{
    let mut wrote = false;
    loop {
        let started = queue.borrow_mut().start(StartReason::Closing);
        if let Some(snapshot) = started {
            let result = save(snapshot.settings.clone()).await;
            let finished = queue.borrow_mut().finish(snapshot, result, now());
            match finished {
                Finished::Saved => wrote = true,
                Finished::Failed { error, .. } => return FlushOutcome::Failed(error),
            }
            continue;
        }
        let waiter = queue.borrow_mut().wait_for_write();
        match waiter {
            Some(waiter) => {
                if waiter.await.is_err() {
                    tracing::debug!("The settings write ended without signalling the flush");
                }
            }
            None => return settled(&queue.borrow(), wrote),
        }
    }
}

/// [`flush`] for when nothing can be awaited, such as a page being hidden: the save must finish
/// on its first poll, which a synchronous storage write does.
pub fn flush_now<S, F>(queue: &RefCell<SaveQueue>, save: &S, now: impl Fn() -> u64) -> FlushOutcome
where
    S: Fn(Settings) -> F,
    F: Future<Output = Result<(), String>>,
{
    let started = queue.borrow_mut().start(StartReason::Closing);
    let Some(snapshot) = started else {
        let queue = queue.borrow();
        return if queue.is_writing() {
            FlushOutcome::Busy
        } else {
            settled(&queue, false)
        };
    };
    let result = save(snapshot.settings.clone())
        .now_or_never()
        .unwrap_or_else(|| Err("The save did not finish before the page closed".to_owned()));
    match queue.borrow_mut().finish(snapshot, result, now()) {
        Finished::Saved => FlushOutcome::Saved,
        Finished::Failed { error, .. } => FlushOutcome::Failed(error),
    }
}

fn settled(queue: &SaveQueue, wrote: bool) -> FlushOutcome {
    match (queue.access(), queue.has_unsaved(), wrote) {
        (WriteAccess::Locked, true, _) => FlushOutcome::Locked,
        (WriteAccess::Open, true, _) => {
            FlushOutcome::Failed("A change is still waiting to be written".to_owned())
        }
        (WriteAccess::Locked | WriteAccess::Open, false, true) => FlushOutcome::Saved,
        (WriteAccess::Locked | WriteAccess::Open, false, false) => FlushOutcome::NothingToSave,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings_store::status::SaveStatus;
    use futures::channel::oneshot;
    use std::{
        cell::Cell,
        pin::pin,
        rc::Rc,
        task::{Context, Poll},
    };

    fn with_model(name: &str) -> Settings {
        let mut settings = Settings::default();
        name.clone_into(&mut settings.model.model_name);
        settings
    }

    fn model_names(writes: &RefCell<Vec<Settings>>) -> Vec<String> {
        writes
            .borrow()
            .iter()
            .map(|settings| settings.model.model_name.clone())
            .collect()
    }

    /// Records every write; the first one waits for `gate` when there is one, and the writes
    /// listed in `failures` (by order) fail.
    struct FakeStorage {
        writes: Rc<RefCell<Vec<Settings>>>,
        gate: RefCell<Option<oneshot::Receiver<()>>>,
        failures: Vec<usize>,
        calls: Cell<usize>,
    }

    impl FakeStorage {
        fn new() -> Self {
            Self {
                writes: Rc::default(),
                gate: RefCell::new(None),
                failures: Vec::new(),
                calls: Cell::new(0),
            }
        }

        fn save(&self, settings: Settings) -> impl Future<Output = Result<(), String>> + use<> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            let gate = self.gate.borrow_mut().take();
            let fails = self.failures.contains(&call);
            let writes = Rc::clone(&self.writes);
            async move {
                if let Some(gate) = gate {
                    gate.await.map_err(|error| error.to_string())?;
                }
                if fails {
                    return Err(format!("write {call} failed"));
                }
                writes.borrow_mut().push(settings);
                Ok(())
            }
        }
    }

    fn poll_once<F: Future>(future: std::pin::Pin<&mut F>) -> Poll<F::Output> {
        future.poll(&mut Context::from_waker(futures::task::noop_waker_ref()))
    }

    #[test]
    fn a_burst_while_writing_ends_in_two_writes_of_the_first_and_newest() {
        let queue = RefCell::new(SaveQueue::new(WriteAccess::Open));
        let storage = FakeStorage::new();
        let (release, gate) = oneshot::channel();
        *storage.gate.borrow_mut() = Some(gate);
        let save = |settings| storage.save(settings);

        queue.borrow_mut().submit(with_model("a"));
        let events = RefCell::new(Vec::new());
        let mut running = pin!(drain(
            &queue,
            &save,
            || 5,
            |event| events.borrow_mut().push(event)
        ));
        assert_eq!(poll_once(running.as_mut()), Poll::Pending);
        for name in ["b", "c", "d"] {
            queue.borrow_mut().submit(with_model(name));
        }
        assert_eq!(queue.borrow().status(), SaveStatus::Saving);
        assert_eq!(release.send(()), Ok(()));
        assert_eq!(poll_once(running.as_mut()), Poll::Ready(()));

        assert_eq!(model_names(&storage.writes), ["a", "d"]);
        assert_eq!(
            *events.borrow(),
            [None, Some(Finished::Saved), None, Some(Finished::Saved)]
        );
        assert_eq!(queue.borrow().status(), SaveStatus::Saved { at_millis: 5 });
    }

    #[test]
    fn a_failed_write_is_reported_and_not_retried_in_a_loop() {
        let queue = RefCell::new(SaveQueue::new(WriteAccess::Open));
        let mut storage = FakeStorage::new();
        storage.failures = vec![0];
        let save = |settings| storage.save(settings);

        queue.borrow_mut().submit(with_model("a"));
        let mut events = Vec::new();
        let finished = drain(&queue, &save, || 5, |event| events.push(event)).now_or_never();
        assert_eq!(finished, Some(()));
        assert_eq!(
            events,
            [
                None,
                Some(Finished::Failed {
                    error: "write 0 failed".to_owned(),
                    first: true
                })
            ]
        );
        assert_eq!(storage.calls.get(), 1);

        queue.borrow_mut().release_held();
        drain(&queue, &save, || 9, |_| ()).now_or_never();
        assert_eq!(model_names(&storage.writes), ["a"]);
        assert_eq!(queue.borrow().status(), SaveStatus::Saved { at_millis: 9 });
    }

    #[test]
    fn flush_waits_until_the_newest_change_is_written() {
        let queue = RefCell::new(SaveQueue::new(WriteAccess::Open));
        let storage = FakeStorage::new();
        let (release, gate) = oneshot::channel();
        *storage.gate.borrow_mut() = Some(gate);
        let save = |settings| storage.save(settings);

        queue.borrow_mut().submit(with_model("a"));
        let mut running = pin!(drain(&queue, &save, || 5, |_| ()));
        assert_eq!(poll_once(running.as_mut()), Poll::Pending);
        queue.borrow_mut().submit(with_model("b"));

        let mut flushing = pin!(flush(&queue, &save, || 6));
        assert_eq!(poll_once(flushing.as_mut()), Poll::Pending);
        assert_eq!(release.send(()), Ok(()));
        assert_eq!(poll_once(flushing.as_mut()), Poll::Pending);
        assert_eq!(poll_once(running.as_mut()), Poll::Ready(()));
        assert_eq!(model_names(&storage.writes), ["a", "b"]);
        assert_eq!(
            poll_once(flushing.as_mut()),
            Poll::Ready(FlushOutcome::NothingToSave)
        );
        assert_eq!(storage.calls.get(), 2);
    }

    #[test]
    fn flush_retries_a_failed_write() {
        let queue = RefCell::new(SaveQueue::new(WriteAccess::Open));
        let mut storage = FakeStorage::new();
        storage.failures = vec![0];
        let save = |settings| storage.save(settings);
        queue.borrow_mut().submit(with_model("a"));
        drain(&queue, &save, || 5, |_| ()).now_or_never();

        assert_eq!(
            flush(&queue, &save, || 6).now_or_never(),
            Some(FlushOutcome::Saved)
        );
        assert_eq!(model_names(&storage.writes), ["a"]);
        assert_eq!(
            flush(&queue, &save, || 7).now_or_never(),
            Some(FlushOutcome::NothingToSave)
        );
    }

    #[test]
    fn nothing_is_written_over_unreadable_settings() {
        let queue = RefCell::new(SaveQueue::new(WriteAccess::Locked));
        let storage = FakeStorage::new();
        let save = |settings| storage.save(settings);
        queue.borrow_mut().submit(with_model("a"));

        drain(&queue, &save, || 5, |_| ()).now_or_never();
        assert_eq!(
            flush(&queue, &save, || 6).now_or_never(),
            Some(FlushOutcome::Locked)
        );
        assert_eq!(flush_now(&queue, &save, || 6), FlushOutcome::Locked);
        assert_eq!(storage.calls.get(), 0);
    }

    #[test]
    fn flush_now_writes_a_change_the_writer_has_not_picked_up() {
        let queue = RefCell::new(SaveQueue::new(WriteAccess::Open));
        let storage = FakeStorage::new();
        let save = |settings| storage.save(settings);
        queue.borrow_mut().submit(with_model("a"));
        queue.borrow_mut().submit(with_model("b"));

        assert_eq!(flush_now(&queue, &save, || 6), FlushOutcome::Saved);
        assert_eq!(model_names(&storage.writes), ["b"]);
        assert_eq!(flush_now(&queue, &save, || 7), FlushOutcome::NothingToSave);
    }

    #[test]
    fn flush_now_reports_a_save_that_cannot_finish_at_once() {
        let queue = RefCell::new(SaveQueue::new(WriteAccess::Open));
        let storage = FakeStorage::new();
        let (_release, gate) = oneshot::channel::<()>();
        *storage.gate.borrow_mut() = Some(gate);
        let save = |settings| storage.save(settings);
        queue.borrow_mut().submit(with_model("a"));

        assert_eq!(
            flush_now(&queue, &save, || 6),
            FlushOutcome::Failed("The save did not finish before the page closed".to_owned())
        );
        assert!(queue.borrow().has_unsaved());
    }
}
