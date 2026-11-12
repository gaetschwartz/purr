use crate::platform::{self, IS_WEB};
use dioxus::prelude::*;
use purr_common::{accel::SystemReport, platform::Platform as _};

/// What the machine reported about the hardware transcription can run on.
#[derive(Debug, Clone)]
pub enum SystemProbe {
    Checking,
    /// The platform has no report, or the probe failed.
    Unavailable,
    Ready(SystemReport),
}

/// Probes the machine once; call at the app root.
///
/// The browser has no report: its adapter comes from the models handle.
pub fn use_system_probe_provider() {
    let mut probe = use_signal(|| SystemProbe::Checking);
    use_resource(move || async move {
        if IS_WEB {
            probe.set(SystemProbe::Unavailable);
            return;
        }
        let next = match platform::get_platform().system_report().await {
            Ok(report) => SystemProbe::Ready(report),
            Err(error) => {
                tracing::warn!("Could not probe the hardware: {error}");
                SystemProbe::Unavailable
            }
        };
        probe.set(next);
    });
    use_context_provider(|| ReadSignal::from(probe));
}

/// The probe provided at the app root.
pub fn use_system_probe() -> ReadSignal<SystemProbe> {
    use_context::<ReadSignal<SystemProbe>>()
}
