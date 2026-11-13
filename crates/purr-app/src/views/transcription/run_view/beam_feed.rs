use crate::{
    transcribe::{beam_lanes::refresh_interval, session::TranscribeSession},
    utils::delay::sleep,
};
use dioxus::prelude::*;
use purr_common::beams::BeamSnapshot;
use std::time::Duration;

const IDLE: Duration = Duration::from_millis(500);

/// The decoder's latest beams, copied from the run a few times a second while `wanted`.
///
/// Telemetry arrives faster than the lanes are worth redrawing, so the view reads this instead of
/// the run.
pub fn use_beam_feed(session: &TranscribeSession, wanted: bool) -> Signal<Option<BeamSnapshot>> {
    let session = *session;
    let mut shown = use_signal(|| None::<BeamSnapshot>);
    let mut wanted_now = use_signal(|| wanted);
    use_effect(use_reactive!(|wanted| {
        if *wanted_now.peek() != wanted {
            wanted_now.set(wanted);
        }
    }));
    use_future(move || async move {
        let mut probe =
            document::eval("dioxus.send(matchMedia('(prefers-reduced-motion: reduce)').matches)");
        let reduced = match probe.recv::<bool>().await {
            Ok(reduced) => reduced,
            Err(error) => {
                tracing::debug!("Could not read the motion preference: {error}");
                false
            }
        };
        loop {
            let pace = if *wanted_now.peek() {
                let latest = session.latest_beams();
                if *shown.peek() != latest {
                    shown.set(latest);
                }
                refresh_interval(reduced)
            } else {
                IDLE
            };
            sleep(pace).await;
        }
    });
    shown
}
