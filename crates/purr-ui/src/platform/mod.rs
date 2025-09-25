/// Platform abstraction layer for client-only architecture
/// This module provides a unified interface for platform-specific functionality
pub use purr_common::platform::*;
use std::sync::Arc;
use tokio::sync::OnceCell;

/// Re-export the appropriate platform implementation
#[cfg(feature = "desktop")]
#[path = "desktop.rs"]
mod platform_impl;

#[cfg(all(feature = "web", target_arch = "wasm32", not(feature = "desktop")))]
pub use purr_wasm::platform as platform_impl;

#[cfg(not(any(feature = "desktop", all(feature = "web", target_arch = "wasm32"))))]
#[path = "unimplemented.rs"]
mod platform_impl;

pub async fn get_platform() -> Result<&'static Arc<dyn Platform>, PlatformError> {
    static PLATFORM: OnceCell<Arc<dyn Platform>> = OnceCell::const_new();
    PLATFORM
        .get_or_try_init(|| async move {
            <platform_impl::PlatformImpl as Platform>::new()
                .await
                .map(|p| Arc::new(p) as Arc<dyn Platform>)
        })
        .await
}
