/// Platform abstraction layer for client-only architecture
/// This module provides a unified interface for platform-specific functionality
pub use purr_common::platform::*;
use std::sync::{Arc, LazyLock};

/// Re-export the appropriate platform implementation
#[cfg(feature = "desktop")]
#[path = "desktop.rs"]
mod platform_impl;

#[cfg(all(feature = "web", target_arch = "wasm32", not(feature = "desktop")))]
#[path = "web.rs"]
mod platform_impl;

#[cfg(not(any(feature = "desktop", all(feature = "web", target_arch = "wasm32"))))]
#[path = "unimplemented.rs"]
mod platform_impl;

pub static PLATFORM: LazyLock<Arc<dyn Platform>> =
    LazyLock::new(|| Arc::new(platform_impl::PlatformImpl::new()));
