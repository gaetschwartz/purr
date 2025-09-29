use ambassador::Delegate;
use bytes::Bytes;
pub use purr_common::platform::*;
use purr_common::settings::Settings;
use std::path::Path;
use std::sync::{Arc, LazyLock};

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

#[derive(Delegate)]
#[delegate(Platform)]
pub struct PlatformImpl {
    inner: platform_impl::PlatformImpl,
}

impl PlatformImpl {
    fn new() -> Result<Self, PlatformError> {
        Ok(Self {
            inner: platform_impl::PlatformImpl::new()?,
        })
    }
}

pub async fn get_platform() -> Result<&'static Arc<PlatformImpl>, &'static PlatformError> {
    static PLATFORM: LazyLock<Result<Arc<PlatformImpl>, PlatformError>> =
        LazyLock::new(|| PlatformImpl::new().map(Arc::new));
    PLATFORM.as_ref()
}
