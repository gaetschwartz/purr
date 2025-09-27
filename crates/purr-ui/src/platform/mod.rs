/// Platform abstraction layer for client-only architecture
/// This module provides a unified interface for platform-specific functionality
pub use purr_common::platform::*;
use std::sync::Arc;
use tokio::sync::OnceCell;

/// Logging infrastructure
pub mod logging;

/// Re-export the appropriate platform implementation
#[cfg(feature = "desktop")]
#[path = "desktop.rs"]
mod platform_impl;

#[cfg(all(feature = "web", target_arch = "wasm32", not(feature = "desktop")))]
pub use purr_wasm::platform as platform_impl;

mod unimplemented;

#[cfg(not(any(feature = "desktop", all(feature = "web", target_arch = "wasm32"))))]
use unimplemented as platform_impl;

pub async fn get_platform() -> Result<&'static Arc<platform_impl::PlatformImpl>, PlatformError> {
    static PLATFORM: OnceCell<Arc<platform_impl::PlatformImpl>> = OnceCell::const_new();
    PLATFORM
        .get_or_try_init(|| async move {
            <platform_impl::PlatformImpl>::new()
                .await
                .map(|p| Arc::new(p))
        })
        .await
}
