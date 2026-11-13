use dioxus::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Choice {
    auto: bool,
    open: bool,
}

/// A manual choice holds only while the automatic state it was made against has not changed.
fn resolve(auto: bool, choice: Option<Choice>) -> bool {
    match choice {
        Some(choice) if choice.auto == auto => choice.open,
        Some(_) | None => auto,
    }
}

/// A collapsible region that opens on its own when `auto` turns true and otherwise follows the user.
#[derive(Clone, Copy, PartialEq)]
pub struct Disclosure {
    auto: bool,
    choice: Signal<Option<Choice>>,
}

impl Disclosure {
    /// Whether the region is showing.
    #[must_use]
    pub fn is_open(&self) -> bool {
        resolve(self.auto, (self.choice)())
    }

    /// Opens a closed region and closes an open one.
    pub fn toggle(&mut self) {
        let open = !self.is_open();
        self.choice.set(Some(Choice {
            auto: self.auto,
            open,
        }));
    }
}

/// Keeps the open state of a region that is open by default whenever `auto` holds.
pub fn use_disclosure(auto: bool) -> Disclosure {
    Disclosure {
        auto,
        choice: use_signal(|| None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_a_choice_the_region_follows_the_automatic_state() {
        assert_eq!((resolve(true, None), resolve(false, None)), (true, false));
    }

    #[test]
    fn a_choice_wins_while_the_automatic_state_is_unchanged() {
        let closed = Some(Choice {
            auto: true,
            open: false,
        });
        let opened = Some(Choice {
            auto: false,
            open: true,
        });
        assert_eq!(
            (resolve(true, closed), resolve(false, opened)),
            (false, true)
        );
    }

    #[test]
    fn a_choice_is_dropped_when_the_automatic_state_changes() {
        let closed = Some(Choice {
            auto: true,
            open: false,
        });
        assert!(!resolve(false, closed));
        let opened = Some(Choice {
            auto: false,
            open: true,
        });
        assert!(resolve(true, opened));
    }
}
