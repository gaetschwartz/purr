use dioxus::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(1);

/// A DOM id unique to this component instance, stable across its renders.
pub fn use_dom_id(prefix: &'static str) -> String {
    use_hook(|| format!("{prefix}-{}", NEXT.fetch_add(1, Ordering::Relaxed)))
}
