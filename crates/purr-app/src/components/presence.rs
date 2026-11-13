use crate::utils::delay::sleep;
use dioxus::{core::Task, prelude::*};
use std::{cell::RefCell, rc::Rc, time::Duration};

/// How long an overlay stays mounted after it is asked to close, which covers its exit animation.
pub const EXIT: Duration = Duration::from_millis(240);

/// Where a [`Presence`] stands between mounted and gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Open,
    Closing,
    Closed,
}

impl Phase {
    /// The phase after the caller asks for `present`.
    const fn requested(self, present: bool) -> Self {
        match (present, self) {
            (true, Self::Open | Self::Closing | Self::Closed) => Self::Open,
            (false, Self::Open | Self::Closing) => Self::Closing,
            (false, Self::Closed) => Self::Closed,
        }
    }

    /// The phase once the exit time has run out.
    const fn settled(self) -> Self {
        match self {
            Self::Open => Self::Open,
            Self::Closing | Self::Closed => Self::Closed,
        }
    }
}

#[derive(Clone, Copy)]
struct Shown(Signal<Phase>);

/// Follows `present`, holding the phase at [`Phase::Closing`] for `exit` after it turns false.
fn use_phase(present: bool, exit: Duration) -> Signal<Phase> {
    let mut phase = use_signal(|| if present { Phase::Open } else { Phase::Closed });
    let mut timer = use_signal(|| None::<Task>);
    use_effect(use_reactive!(|present| {
        if let Some(task) = timer.write().take() {
            task.cancel();
        }
        let current = *phase.peek();
        let next = current.requested(present);
        if next != current {
            phase.set(next);
        }
        if next == Phase::Closing {
            let task = spawn(async move {
                sleep(exit).await;
                let settled = phase.peek().settled();
                phase.set(settled);
                timer.set(None);
            });
            timer.set(Some(task));
        }
    }));
    phase
}

/// Keeps `children` mounted while `present` holds and for [`EXIT`] after it turns false, so the
/// overlay inside can play its exit animation; reopening during the exit cancels it.
///
/// While it exits, `children` stay as they were when `present` was last true, so state that
/// the opener resets on close does not flash. [`Popover`](super::popover::Popover),
/// [`Sheet`](super::sheet::Sheet) and [`PagesDialog`](super::pages_dialog::PagesDialog) read the
/// phase through [`use_closing`] and mark their layer.
#[component]
pub fn Presence(present: bool, children: Element) -> Element {
    let phase = use_phase(present, EXIT);
    use_context_provider(|| Shown(phase));
    let held = use_hook(|| Rc::new(RefCell::new(None::<Element>)));
    if present {
        *held.borrow_mut() = Some(children.clone());
        return children;
    }
    if phase() == Phase::Closed {
        held.borrow_mut().take();
        return rsx! {};
    }
    held.borrow().clone().unwrap_or(children)
}

/// Whether the nearest [`Presence`] above the caller is on its way out.
pub fn use_closing() -> bool {
    try_use_context::<Shown>().is_some_and(|shown| (shown.0)() == Phase::Closing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{
        layer::{LayerOutlet, use_layer_host_provider},
        sheet::Sheet,
    };
    use dioxus::core::NoOpMutations;
    use std::cell::Cell;

    const TEST_EXIT: Duration = Duration::from_millis(60);

    #[test]
    fn asking_to_close_an_open_overlay_starts_the_exit() {
        assert_eq!(Phase::Open.requested(false), Phase::Closing);
    }

    #[test]
    fn asking_again_while_closing_keeps_the_exit_running() {
        assert_eq!(Phase::Closing.requested(false), Phase::Closing);
    }

    #[test]
    fn reopening_during_the_exit_cancels_it() {
        assert_eq!(Phase::Closing.requested(true), Phase::Open);
    }

    #[test]
    fn a_closed_overlay_stays_closed_until_asked() {
        assert_eq!(Phase::Closed.requested(false), Phase::Closed);
        assert_eq!(Phase::Closed.requested(true), Phase::Open);
    }

    #[test]
    fn the_exit_ends_closed_and_never_closes_an_open_overlay() {
        assert_eq!(Phase::Closing.settled(), Phase::Closed);
        assert_eq!(Phase::Open.settled(), Phase::Open);
    }

    #[derive(Clone)]
    struct Rig {
        wanted: Rc<Cell<Option<Signal<bool>>>>,
        seen: Rc<Cell<Phase>>,
        start_open: bool,
    }

    impl PartialEq for Rig {
        fn eq(&self, _: &Self) -> bool {
            false
        }
    }

    #[allow(non_snake_case, reason = "a component")]
    fn Driver(rig: Rig) -> Element {
        let present = use_signal(|| rig.start_open);
        rig.wanted.set(Some(present));
        rsx! {
            Observer { rig: rig.clone(), present: present() }
        }
    }

    #[allow(non_snake_case, reason = "a component")]
    #[component]
    fn Observer(rig: Rig, present: bool) -> Element {
        let phase = use_phase(present, TEST_EXIT);
        rig.seen.set(phase());
        rsx! {}
    }

    struct Harness {
        dom: VirtualDom,
        wanted: Rc<Cell<Option<Signal<bool>>>>,
        seen: Rc<Cell<Phase>>,
    }

    impl Harness {
        fn new(start_open: bool) -> Self {
            let wanted = Rc::new(Cell::new(None));
            let seen = Rc::new(Cell::new(Phase::Closed));
            let rig = Rig {
                wanted: Rc::clone(&wanted),
                seen: Rc::clone(&seen),
                start_open,
            };
            let mut dom = VirtualDom::new_with_props(Driver, rig);
            dom.rebuild_in_place();
            Self { dom, wanted, seen }
        }

        fn present(&mut self, present: bool) {
            let mut signal = self.wanted.get().expect("the driver rendered");
            self.dom.in_runtime(|| signal.set(present));
        }

        async fn run_for(&mut self, span: Duration) {
            let deadline = tokio::time::Instant::now() + span;
            while let Ok(()) = tokio::time::timeout_at(deadline, self.dom.wait_for_work()).await {
                self.dom.render_immediate(&mut NoOpMutations);
            }
        }
    }

    #[derive(Clone)]
    struct LayerRig {
        wanted: Rc<Cell<Option<Signal<bool>>>>,
        closing: Rc<Cell<bool>>,
        dropped: Rc<Cell<bool>>,
    }

    impl PartialEq for LayerRig {
        fn eq(&self, _: &Self) -> bool {
            false
        }
    }

    #[allow(non_snake_case, reason = "a component")]
    fn LayerDriver(rig: LayerRig) -> Element {
        use_layer_host_provider();
        let present = use_signal(|| true);
        rig.wanted.set(Some(present));
        rsx! {
            Presence { present: present(),
                ClosingProbe {
                    closing: Rc::clone(&rig.closing),
                    dropped: Rc::clone(&rig.dropped),
                }
                Sheet { title: "Probe", on_close: move |()| {}, "Body" }
            }
            LayerOutlet {}
        }
    }

    #[derive(Clone, Props)]
    struct ClosingProbeProps {
        closing: Rc<Cell<bool>>,
        dropped: Rc<Cell<bool>>,
    }

    impl PartialEq for ClosingProbeProps {
        fn eq(&self, _: &Self) -> bool {
            false
        }
    }

    #[allow(non_snake_case, reason = "a component")]
    fn ClosingProbe(props: ClosingProbeProps) -> Element {
        props.closing.set(use_closing());
        let dropped = Rc::clone(&props.dropped);
        use_drop(move || dropped.set(true));
        rsx! {}
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_sheet_inside_a_presence_is_told_it_is_closing_and_is_unmounted_after_the_exit() {
        let wanted = Rc::new(Cell::new(None));
        let closing = Rc::new(Cell::new(false));
        let dropped = Rc::new(Cell::new(false));
        let mut dom = VirtualDom::new_with_props(
            LayerDriver,
            LayerRig {
                wanted: Rc::clone(&wanted),
                closing: Rc::clone(&closing),
                dropped: Rc::clone(&dropped),
            },
        );
        dom.rebuild_in_place();
        let mut present = wanted.get().expect("the driver rendered");
        assert_eq!(closing.get(), false);
        dom.in_runtime(|| present.set(false));
        let until = tokio::time::Instant::now() + EXIT / 2;
        while let Ok(()) = tokio::time::timeout_at(until, dom.wait_for_work()).await {
            dom.render_immediate(&mut NoOpMutations);
        }
        assert_eq!(closing.get(), true);
        assert_eq!(dropped.get(), false);
        let until = tokio::time::Instant::now() + EXIT * 2;
        while let Ok(()) = tokio::time::timeout_at(until, dom.wait_for_work()).await {
            dom.render_immediate(&mut NoOpMutations);
        }
        assert_eq!(dropped.get(), true);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn an_overlay_closed_from_the_start_stays_closed() {
        let mut harness = Harness::new(false);
        harness.run_for(TEST_EXIT * 2).await;
        assert_eq!(harness.seen.get(), Phase::Closed);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn closing_holds_the_overlay_for_the_exit_then_removes_it() {
        let mut harness = Harness::new(true);
        harness.run_for(TEST_EXIT / 4).await;
        assert_eq!(harness.seen.get(), Phase::Open);
        harness.present(false);
        harness.run_for(TEST_EXIT / 4).await;
        assert_eq!(harness.seen.get(), Phase::Closing);
        harness.run_for(TEST_EXIT * 2).await;
        assert_eq!(harness.seen.get(), Phase::Closed);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn reopening_during_the_exit_keeps_the_overlay_past_the_old_deadline() {
        let mut harness = Harness::new(true);
        harness.present(false);
        harness.run_for(TEST_EXIT / 4).await;
        harness.present(true);
        harness.run_for(TEST_EXIT * 2).await;
        assert_eq!(harness.seen.get(), Phase::Open);
        harness.present(false);
        harness.run_for(TEST_EXIT * 2).await;
        assert_eq!(harness.seen.get(), Phase::Closed);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_second_close_restarts_the_exit_instead_of_inheriting_the_first() {
        let mut harness = Harness::new(true);
        harness.present(false);
        harness.run_for(Duration::from_millis(15)).await;
        harness.present(true);
        harness.run_for(Duration::from_millis(25)).await;
        harness.present(false);
        harness.run_for(Duration::from_millis(40)).await;
        assert_eq!(harness.seen.get(), Phase::Closing);
        harness.run_for(Duration::from_millis(90)).await;
        assert_eq!(harness.seen.get(), Phase::Closed);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn an_overlay_opens_again_after_it_closed() {
        let mut harness = Harness::new(true);
        harness.present(false);
        harness.run_for(TEST_EXIT * 2).await;
        assert_eq!(harness.seen.get(), Phase::Closed);
        harness.present(true);
        harness.run_for(TEST_EXIT / 4).await;
        assert_eq!(harness.seen.get(), Phase::Open);
    }
}
