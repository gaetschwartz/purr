use serde::{Deserialize, Serialize};

/// Whether the list keeps showing the newest entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FollowMode {
    /// New entries scroll into view.
    #[default]
    Following,
    /// The user scrolled up; reaching the bottom again resumes following.
    Paused,
    /// The user switched auto-scroll off.
    Off,
}

/// What the page reports about scrolling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScrollEvent {
    /// The user scrolled away from the bottom.
    Up,
    /// The scroll position reached the bottom.
    Bottom,
}

impl FollowMode {
    /// The mode after the auto-scroll switch is set to `on`.
    #[must_use]
    pub const fn switched(on: bool) -> Self {
        if on { Self::Following } else { Self::Off }
    }

    /// The mode after the page reports `event`.
    #[must_use]
    pub const fn after(self, event: ScrollEvent) -> Self {
        match (self, event) {
            (Self::Following, ScrollEvent::Up) => Self::Paused,
            (Self::Paused, ScrollEvent::Bottom) => Self::Following,
            (Self::Following, ScrollEvent::Bottom)
            | (Self::Paused, ScrollEvent::Up)
            | (Self::Off, ScrollEvent::Up | ScrollEvent::Bottom) => self,
        }
    }

    /// Whether the auto-scroll switch reads on.
    #[must_use]
    pub const fn is_following(self) -> bool {
        match self {
            Self::Following => true,
            Self::Paused | Self::Off => false,
        }
    }
}

/// Watches the scroller of the log list (the list itself, or the page where the list grows with
/// its content) and keeps it at the bottom while following.
///
/// A scroll up counts as the user's only when a wheel, touch, key or scrollbar press asking for it
/// came just before, so scrolling done by the script or by the browser adjusting to new rows
/// never pauses following.
pub const WATCHER: &str = r"
const dispose = window.__purrLogFollow;
if (dispose) dispose();
let main = null;
for (let tries = 0; tries < 60 && !main; tries++) {
  main = document.getElementById('main-content');
  if (!main) await new Promise((resolve) => requestAnimationFrame(resolve));
}
if (!main) { return; }
let mode = 'following';
let intentUntil = 0;
let lastTop = null;
let frame = 0;
const list = () => document.getElementById('loglist');
const scroller = () => {
  const el = list();
  if (!el) return null;
  return getComputedStyle(el).maxHeight === 'none' ? main : el;
};
const toBottom = () => { const el = scroller(); if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'instant' }); };
const nearBottom = (el) => el.scrollHeight - el.scrollTop - el.clientHeight <= 24;
const mark = (ms) => { intentUntil = performance.now() + ms; };
const inside = (event) => { const el = scroller(); return !!el && el.contains(event.target); };
let touchY = 0;
const handlers = {
  wheel: (event) => { if (event.deltaY < 0 && inside(event)) mark(400); },
  touchstart: (event) => { touchY = event.touches[0] ? event.touches[0].clientY : 0; },
  touchmove: (event) => {
    const y = event.touches[0] ? event.touches[0].clientY : touchY;
    if (y > touchY + 2 && inside(event)) mark(400);
    touchY = y;
  },
  keydown: (event) => { if (['ArrowUp', 'PageUp', 'Home'].includes(event.key) && inside(event)) mark(400); },
  pointerdown: (event) => { const el = scroller(); if (el && event.target === el && event.offsetX >= el.clientWidth) mark(5000); },
  pointerup: () => { intentUntil = Math.min(intentUntil, performance.now() + 400); },
};
const check = (target) => {
  frame = 0;
  const el = scroller();
  if (!el || el !== target) return;
  const up = lastTop !== null && el.scrollTop < lastTop - 1;
  lastTop = el.scrollTop;
  if (mode === 'following' && up && !nearBottom(el) && performance.now() < intentUntil) {
    mode = 'paused';
    dioxus.send('up');
  } else if (mode === 'paused' && nearBottom(el)) {
    mode = 'following';
    dioxus.send('bottom');
  }
};
const onScroll = (event) => {
  if (frame) return;
  const target = event.target;
  frame = requestAnimationFrame(() => check(target));
};
const observer = new MutationObserver((records) => {
  const el = list();
  if (mode === 'following' && el && records.some((record) => el.contains(record.target))) toBottom();
});
const names = Object.keys(handlers);
names.forEach((name) => main.addEventListener(name, handlers[name], { capture: true, passive: true }));
main.addEventListener('scroll', onScroll, true);
const onResize = () => { if (mode === 'following') toBottom(); };
window.addEventListener('resize', onResize);
observer.observe(main, { childList: true, subtree: true });
window.__purrLogFollow = () => {
  names.forEach((name) => main.removeEventListener(name, handlers[name], true));
  main.removeEventListener('scroll', onScroll, true);
  window.removeEventListener('resize', onResize);
  observer.disconnect();
  if (frame) cancelAnimationFrame(frame);
  window.__purrLogFollow = null;
};
toBottom();
for (;;) {
  mode = await dioxus.recv();
  lastTop = null;
  if (mode === 'following') toBottom();
}
";

/// Stops the watcher started by [`WATCHER`].
pub const DISPOSE_WATCHER: &str = "if (window.__purrLogFollow) { window.__purrLogFollow(); }";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrolling_up_pauses_and_returning_to_the_bottom_resumes() {
        let paused = FollowMode::Following.after(ScrollEvent::Up);
        assert_eq!(paused, FollowMode::Paused);
        assert_eq!(paused.after(ScrollEvent::Bottom), FollowMode::Following);
    }

    #[test]
    fn scrolling_does_not_override_the_switch() {
        assert_eq!(FollowMode::Off.after(ScrollEvent::Bottom), FollowMode::Off);
        assert_eq!(FollowMode::Off.after(ScrollEvent::Up), FollowMode::Off);
    }

    #[test]
    fn reports_that_change_nothing_leave_the_mode_alone() {
        assert_eq!(
            FollowMode::Following.after(ScrollEvent::Bottom),
            FollowMode::Following
        );
        assert_eq!(
            FollowMode::Paused.after(ScrollEvent::Up),
            FollowMode::Paused
        );
    }

    #[test]
    fn the_switch_decides_between_following_and_off() {
        assert_eq!(FollowMode::switched(true), FollowMode::Following);
        assert_eq!(FollowMode::switched(false), FollowMode::Off);
        assert!(FollowMode::Following.is_following());
        assert!(!FollowMode::Paused.is_following());
        assert!(!FollowMode::Off.is_following());
    }
}
