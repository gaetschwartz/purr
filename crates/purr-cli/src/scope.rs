//! Which stream the text being formatted goes to: stdout unless inside `on_stderr`.

use crate::color::Stream;
use std::cell::Cell;

thread_local! {
    static CURRENT: Cell<Stream> = const { Cell::new(Stream::Stdout) };
}

struct Restore(Stream);

impl Drop for Restore {
    fn drop(&mut self) {
        CURRENT.with(|current| current.set(self.0));
    }
}

/// The stream the text formatted on this thread right now is for.
pub fn current() -> Stream {
    CURRENT.with(Cell::get)
}

/// Runs `print`, during which styled text is for stderr. A closure, so the scope cannot be held
/// across an `.await`.
pub fn on_stderr<R>(print: impl FnOnce() -> R) -> R {
    let _restore = Restore(CURRENT.with(|current| current.replace(Stream::Stderr)));
    print()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_is_for_stdout_except_inside_the_stderr_scope() {
        assert_eq!(current(), Stream::Stdout);
        let inside = on_stderr(|| {
            let nested = on_stderr(current);
            (current(), nested)
        });
        assert_eq!(inside, (Stream::Stderr, Stream::Stderr));
        assert_eq!(current(), Stream::Stdout);
    }

    #[test]
    fn the_scope_ends_when_the_closure_panics() {
        let panicked = std::panic::catch_unwind(|| on_stderr(|| panic!("boom")));
        assert_eq!(panicked.map_err(|_| "panicked"), Err("panicked"));
        assert_eq!(current(), Stream::Stdout);
    }
}
