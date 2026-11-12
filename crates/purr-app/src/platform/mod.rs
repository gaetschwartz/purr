use ambassador::Delegate;
use bytes::Bytes;
pub use purr_common::platform::*;
use purr_common::{
    accel::SystemReport,
    audio_overview::AudioOverview,
    history::{
        entry::{HistoryEntry, HistoryId, HistoryRecord},
        listing::{HistoryListing, Saved},
    },
    hugging_face_token::HuggingFaceToken,
    settings::Settings,
};
use std::path::Path;
use std::sync::{Arc, LazyLock};

/// Saving and copying text
pub mod export;

/// The directory of saved transcriptions on the desktop
#[cfg(feature = "desktop")]
mod history_store;

/// Where the player loads audio from
pub mod playback_source;

/// Logging infrastructure
pub mod logging;

/// Text, JSON and CSV files made of captured logs
pub mod log_export;

/// Settings and machine probes as purr-core configuration and reports
#[cfg(feature = "desktop")]
mod engine_options;

/// Re-export the appropriate platform implementation
#[cfg(feature = "desktop")]
#[path = "desktop.rs"]
mod platform_impl;

#[cfg(purr_web)]
pub use purr_wasm::platform as platform_impl;

mod unimplemented;

#[cfg(not(any(feature = "desktop", all(feature = "web", target_arch = "wasm32"))))]
use unimplemented as platform_impl;

#[derive(Delegate)]
#[delegate(Platform)]
enum Inner {
    Ready(platform_impl::PlatformImpl),
    Unavailable(unimplemented::PlatformImpl),
}

/// The platform the app runs on, or a stand-in that refuses everything when it failed to start.
#[derive(Delegate)]
#[delegate(Platform)]
pub struct PlatformImpl {
    inner: Inner,
}

/// Whether this build runs in a browser, where desktop-only options do not apply.
pub const IS_WEB: bool = cfg!(purr_web);

struct StartedPlatform {
    platform: Arc<PlatformImpl>,
    init_error: Option<PlatformError>,
}

static PLATFORM: LazyLock<StartedPlatform> =
    LazyLock::new(|| match platform_impl::PlatformImpl::new() {
        Ok(inner) => StartedPlatform {
            platform: Arc::new(PlatformImpl {
                inner: Inner::Ready(inner),
            }),
            init_error: None,
        },
        Err(error) => StartedPlatform {
            platform: Arc::new(PlatformImpl {
                inner: Inner::Unavailable(unimplemented::UnimplementedPlatformImpl {}),
            }),
            init_error: Some(error),
        },
    });

/// Why the platform could not start, if it could not. The UI shows this instead of rendering.
pub fn platform_init_error() -> Option<&'static PlatformError> {
    PLATFORM.init_error.as_ref()
}

pub fn get_platform() -> &'static Arc<PlatformImpl> {
    &PLATFORM.platform
}
