use crate::{
    components::audio_player::WAVE_ID,
    transcribe::{
        eta::{EtaText, EtaTracker},
        glide::{FINISH, Glide, STOP},
    },
    utils::{clock::now_ms, delay::sleep},
};
use dioxus::prelude::*;
use std::time::Duration;

const FRAME: Duration = Duration::from_millis(33);
const IDLE: Duration = Duration::from_millis(250);
const ETA_TICK: f64 = 0.5;

/// Where the run stands, for the fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlidePhase {
    /// No speech decoded yet: the fill waits at the start.
    Waiting,
    Transcribing,
    Finished,
    Stopped,
}

/// What the fill follows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlideInput {
    /// Seconds transcribed, as the platform reports them.
    pub target: f64,
    /// The length of the audio, or 0 while unknown.
    pub duration: f64,
    pub phase: GlidePhase,
}

/// Writes the gliding fill into the waveform's `--done` property and keeps the time left.
///
/// The property is written by script, so no bar re-renders. `arrived` turns true once the run
/// has ended and the fill has reached its end value.
#[component]
pub fn GlideDriver(
    input: Memo<GlideInput>,
    eta: Signal<Option<EtaText>>,
    arrived: Signal<bool>,
) -> Element {
    let writer = use_hook(|| document::eval(&writer_script()));
    use_future(move || async move {
        let mut writer = writer;
        let mut eta = eta;
        let mut arrived = arrived;
        let reduced = writer.recv::<bool>().await.unwrap_or(false);
        let first = *input.peek();
        let mut glide = Glide::at(seconds_now(), first.target);
        let mut tracker: Option<EtaTracker> = None;
        let mut fed = first.target;
        let mut last_sent = f64::NAN;
        let mut last_tick = f64::NEG_INFINITY;
        loop {
            let now = seconds_now();
            let GlideInput {
                target,
                duration,
                phase,
            } = *input.peek();
            match phase {
                GlidePhase::Waiting => {}
                GlidePhase::Transcribing => glide.retarget(now, target),
                GlidePhase::Finished => glide.finish(now, target, FINISH),
                GlidePhase::Stopped => glide.finish(now, target, STOP),
            }
            let shown = if reduced {
                glide.settle();
                glide.shown()
            } else {
                glide.step(now)
            };
            let percent = if duration > 0.0 {
                (shown / duration * 100.0).clamp(0.0, 100.0)
            } else {
                0.0
            };
            if (percent - last_sent).abs() > 0.001 || last_sent.is_nan() {
                last_sent = percent;
                if let Err(error) = writer.send(percent) {
                    tracing::debug!("The progress fill stopped: {error}");
                    return;
                }
            }

            if phase == GlidePhase::Transcribing {
                let tracking = tracker.get_or_insert_with(|| EtaTracker::new(now, target));
                if target > fed {
                    tracking.update(now, target);
                }
            }
            fed = fed.max(target);
            if now - last_tick >= ETA_TICK {
                last_tick = now;
                let text = match (phase, tracker.as_mut()) {
                    (GlidePhase::Transcribing, Some(tracking)) => {
                        tracking.tick(now, (duration - target).max(0.0))
                    }
                    (GlidePhase::Transcribing, None)
                    | (GlidePhase::Waiting | GlidePhase::Finished | GlidePhase::Stopped, _) => None,
                };
                if *eta.peek() != text {
                    eta.set(text);
                }
            }

            let done = glide.arrived();
            if *arrived.peek() != done {
                arrived.set(done);
            }
            sleep(if done { IDLE } else { FRAME }).await;
        }
    });
    use_drop(move || {
        if let Err(error) = writer.send(-1.0) {
            tracing::debug!("The progress fill was already gone: {error}");
        }
    });
    rsx! {}
}

fn seconds_now() -> f64 {
    now_ms() / 1000.0
}

fn writer_script() -> String {
    format!(
        "dioxus.send(matchMedia('(prefers-reduced-motion: reduce)').matches);
let done = 0;
const paint = () => document.getElementById({WAVE_ID:?})?.style.setProperty('--done', done.toFixed(3) + '%');
for (;;) {{
  done = await dioxus.recv();
  if (done < 0) break;
  paint();
  requestAnimationFrame(paint);
}}"
    )
}
