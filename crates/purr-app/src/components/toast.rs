use super::{
    button::{Button, ButtonSize, ButtonVariant},
    icon::{Icon, IconName, IconSize},
};
use crate::utils::delay::sleep;
use dioxus::{core::spawn_forever, prelude::*};
use std::{fmt, rc::Rc, time::Duration};

/// How many toasts stay on screen; older ones leave first.
const MAX_VISIBLE: usize = 4;
const EXIT_ANIMATION: Duration = Duration::from_millis(300);

/// Meaning and colour of a [`Toast`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastKind {
    Info,
    Success,
    Warning,
    Error,
}

impl ToastKind {
    const fn class(self) -> &'static str {
        match self {
            Self::Info => "toast glass g3",
            Self::Success => "toast glass g3 ok",
            Self::Warning => "toast glass g3 warn",
            Self::Error => "toast glass g3 bad",
        }
    }

    const fn icon(self) -> IconName {
        match self {
            Self::Info => IconName::Info,
            Self::Success => IconName::CheckCircle,
            Self::Warning => IconName::Warning,
            Self::Error => IconName::XCircle,
        }
    }
}

/// A button on a toast, such as Undo or Retry; pressing it also dismisses the toast.
///
/// The closure can run after the component that created it is gone, so it must only touch state
/// provided at the app root (contexts, global signals, the platform), never a view's own signals.
#[derive(Clone)]
pub struct ToastAction {
    pub label: String,
    pub run: Rc<dyn Fn()>,
}

impl PartialEq for ToastAction {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label && Rc::ptr_eq(&self.run, &other.run)
    }
}

impl fmt::Debug for ToastAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToastAction")
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

/// A short notice shown at the top of the window (above the tab bar on phones).
#[derive(Debug, Clone, PartialEq)]
pub struct Toast {
    pub kind: ToastKind,
    pub title: String,
    pub detail: Option<String>,
    pub icon: Option<IconName>,
    pub action: Option<ToastAction>,
}

impl Toast {
    /// A toast of `kind` with a title only.
    pub fn new(kind: ToastKind, title: impl Into<String>) -> Self {
        Self {
            kind,
            title: title.into(),
            detail: None,
            icon: None,
            action: None,
        }
    }

    /// Adds a second, quieter line.
    #[must_use]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    /// Replaces the glyph that goes with the kind.
    #[must_use]
    pub const fn with_icon(mut self, icon: IconName) -> Self {
        self.icon = Some(icon);
        self
    }

    /// Adds an action button; see [`ToastAction`] for what the closure may capture.
    #[must_use]
    pub fn with_action(mut self, label: impl Into<String>, run: impl Fn() + 'static) -> Self {
        self.action = Some(ToastAction {
            label: label.into(),
            run: Rc::new(run),
        });
        self
    }

    /// How long the toast stays before it closes itself; longer when it offers an action.
    #[must_use]
    pub const fn lifetime(&self) -> Duration {
        if self.action.is_some() {
            Duration::from_millis(7000)
        } else {
            Duration::from_millis(4200)
        }
    }
}

/// Identifies a toast in the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ToastId(u64);

/// A toast on screen, `leaving` while its exit animation plays.
#[derive(Debug, Clone, PartialEq)]
pub struct ToastEntry {
    pub id: ToastId,
    pub toast: Toast,
    pub leaving: bool,
}

/// The toasts on screen, oldest first.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToastQueue {
    entries: Vec<ToastEntry>,
    next_id: u64,
}

impl ToastQueue {
    /// Adds a toast, dropping the oldest ones beyond the visible limit.
    pub fn push(&mut self, toast: Toast) -> ToastId {
        let id = ToastId(self.next_id);
        self.next_id += 1;
        self.entries.push(ToastEntry {
            id,
            toast,
            leaving: false,
        });
        let excess = self.entries.len().saturating_sub(MAX_VISIBLE);
        self.entries.drain(..excess);
        id
    }

    /// Starts the exit of a toast; false when it is already gone or leaving.
    pub fn begin_dismiss(&mut self, id: ToastId) -> bool {
        match self
            .entries
            .iter_mut()
            .find(|entry| entry.id == id && !entry.leaving)
        {
            Some(entry) => {
                entry.leaving = true;
                true
            }
            None => false,
        }
    }

    /// Takes a toast off the screen.
    pub fn remove(&mut self, id: ToastId) {
        self.entries.retain(|entry| entry.id != id);
    }

    /// The toasts on screen, oldest first.
    #[must_use]
    pub fn entries(&self) -> &[ToastEntry] {
        &self.entries
    }
}

/// Handle to the app's toasts, from [`use_toasts`].
#[derive(Clone, Copy)]
pub struct Toasts {
    queue: Signal<ToastQueue>,
}

impl Toasts {
    /// Shows a toast and schedules its automatic dismissal.
    pub fn show(mut self, toast: Toast) -> ToastId {
        let lifetime = toast.lifetime();
        let id = self.queue.write().push(toast);
        spawn_forever(async move {
            sleep(lifetime).await;
            self.dismiss(id);
        });
        id
    }

    /// Shows a neutral notice.
    pub fn info(self, title: impl Into<String>, detail: impl Into<String>) -> ToastId {
        self.show(Toast::new(ToastKind::Info, title).with_detail(detail))
    }

    /// Shows a success notice.
    pub fn success(self, title: impl Into<String>, detail: impl Into<String>) -> ToastId {
        self.show(Toast::new(ToastKind::Success, title).with_detail(detail))
    }

    /// Shows a warning.
    pub fn warning(self, title: impl Into<String>, detail: impl Into<String>) -> ToastId {
        self.show(Toast::new(ToastKind::Warning, title).with_detail(detail))
    }

    /// Shows an error.
    pub fn error(self, title: impl Into<String>, detail: impl Into<String>) -> ToastId {
        self.show(Toast::new(ToastKind::Error, title).with_detail(detail))
    }

    /// Closes a toast with its exit animation.
    pub fn dismiss(mut self, id: ToastId) {
        if self.queue.write().begin_dismiss(id) {
            spawn_forever(async move {
                sleep(EXIT_ANIMATION).await;
                self.queue.write().remove(id);
            });
        }
    }
}

/// Provides the toast queue; call once at the root, above [`ToastRegion`].
pub fn use_toast_provider() -> Toasts {
    use_context_provider(|| Toasts {
        queue: Signal::new_in_scope(ToastQueue::default(), ScopeId::ROOT),
    })
}

/// The app's toasts: `use_toasts().success("Saved", "To Downloads")`.
pub fn use_toasts() -> Toasts {
    use_context::<Toasts>()
}

/// The polite live region the toasts render in; mount it once inside `.app`.
#[component]
pub fn ToastRegion() -> Element {
    let toasts = use_toasts();
    let entries = toasts.queue.read().entries().to_vec();
    rsx! {
        div { class: "toasts", role: "status", aria_live: "polite",
            for entry in entries {
                ToastView { key: "{entry.id.0}", entry, toasts }
            }
        }
    }
}

#[component]
fn ToastView(entry: ToastEntry, toasts: Toasts) -> Element {
    let ToastEntry { id, toast, leaving } = entry;
    let class = if leaving {
        format!("{} out", toast.kind.class())
    } else {
        toast.kind.class().to_owned()
    };
    rsx! {
        div { class,
            span { class: "tico",
                Icon { name: toast.icon.unwrap_or_else(|| toast.kind.icon()) }
            }
            div { class: "grow",
                b { "{toast.title}" }
                if let Some(detail) = &toast.detail {
                    span { "{detail}" }
                }
            }
            if let Some(action) = toast.action {
                Button {
                    variant: ButtonVariant::Plain,
                    size: ButtonSize::Small,
                    onclick: move |_| {
                        toasts.dismiss(id);
                        (action.run)();
                    },
                    "{action.label}"
                }
            }
            button {
                class: "icon-btn",
                r#type: "button",
                aria_label: "Dismiss",
                onclick: move |_| toasts.dismiss(id),
                Icon { name: IconName::Close, size: IconSize::Small }
            }
        }
    }
}

impl PartialEq for Toasts {
    fn eq(&self, other: &Self) -> bool {
        self.queue == other.queue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn titles(queue: &ToastQueue) -> Vec<&str> {
        queue
            .entries()
            .iter()
            .map(|entry| entry.toast.title.as_str())
            .collect()
    }

    #[test]
    fn the_oldest_toasts_leave_beyond_the_limit() {
        let mut queue = ToastQueue::default();
        for title in ["a", "b", "c", "d", "e"] {
            queue.push(Toast::new(ToastKind::Info, title));
        }
        assert_eq!(titles(&queue), vec!["b", "c", "d", "e"]);
    }

    #[test]
    fn dismissing_marks_then_removes() {
        let mut queue = ToastQueue::default();
        let first = queue.push(Toast::new(ToastKind::Success, "saved"));
        let second = queue.push(Toast::new(ToastKind::Error, "failed"));
        assert!(queue.begin_dismiss(first));
        assert!(!queue.begin_dismiss(first));
        assert_eq!(
            queue
                .entries()
                .iter()
                .map(|entry| entry.leaving)
                .collect::<Vec<_>>(),
            vec![true, false]
        );
        queue.remove(first);
        assert_eq!(titles(&queue), vec!["failed"]);
        queue.remove(second);
        assert!(!queue.begin_dismiss(second));
    }

    #[test]
    fn toasts_with_an_action_stay_longer() {
        let plain = Toast::new(ToastKind::Info, "Copied");
        let undo = Toast::new(ToastKind::Info, "Deleted").with_action("Undo", || {});
        assert_eq!(plain.lifetime(), Duration::from_millis(4200));
        assert_eq!(undo.lifetime(), Duration::from_millis(7000));
    }
}
