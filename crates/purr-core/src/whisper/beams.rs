//! Live beam telemetry, read from whisper.cpp's logits-filter callback.
//!
//! whisper.cpp calls the callback once per active decoder at every decoding step, before it
//! picks the next token, with that decoder's tokens so far. The calls of one step are the
//! step's beams; a call with no tokens starts a new attempt at the window (a temperature retry).

use crate::whisper::StreamEvent;
use parking_lot::Mutex;
use purr_common::beams::{Beam, BeamSnapshot, BeamToken};
use std::{
    ffi::{CStr, c_int, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    time::{Duration, Instant},
};
use tokio::sync::mpsc::UnboundedSender;
use tracing::{debug, error};
use whisper_rs::whisper_rs_sys::{
    whisper_context, whisper_state, whisper_token_data, whisper_token_to_str,
};

/// The most beams a snapshot shows; whisper.cpp's beam search keeps five by default
pub const MAX_BEAMS: usize = 5;
/// The most tokens a beam shows, the newest kept
pub const SHOWN_TOKENS: usize = 24;
/// Snapshots are published at most this often
pub const PUBLISH_EVERY: Duration = Duration::from_millis(100);

/// One token of a decoder's sequence, as the callback sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawToken {
    pub id: i32,
    pub p: f32,
    pub plog: f32,
}

/// A hypothesis with its shown tokens still as ids.
#[derive(Debug, Clone, PartialEq)]
pub struct RawBeam {
    pub tokens: Vec<RawToken>,
    pub log_prob: f32,
}

/// Lets an event through at most once per interval.
#[derive(Debug, Clone)]
pub struct Throttle {
    every: Duration,
    last: Option<Instant>,
}

impl Throttle {
    #[must_use]
    pub const fn new(every: Duration) -> Self {
        Self { every, last: None }
    }

    /// Whether an event may pass at `now`; a passing event starts the next interval.
    pub fn due(&mut self, now: Instant) -> bool {
        match self.last {
            Some(last) if now.saturating_duration_since(last) < self.every => false,
            Some(_) | None => {
                self.last = Some(now);
                true
            }
        }
    }
}

/// A finished decoding step.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub attempt: u32,
    pub step: u32,
    pub lanes: Vec<Vec<RawToken>>,
}

/// Groups the callback's calls into steps, keeping only the steps the throttle lets through.
#[derive(Debug, Clone)]
pub struct Steps {
    throttle: Throttle,
    attempt: Option<u32>,
    step: Option<u32>,
    recording: bool,
    lanes: Vec<Vec<RawToken>>,
}

impl Steps {
    #[must_use]
    pub const fn new(throttle: Throttle) -> Self {
        Self {
            throttle,
            attempt: None,
            step: None,
            recording: false,
            lanes: Vec::new(),
        }
    }

    /// Forgets the window being decoded; the throttle keeps its pace across windows.
    pub fn start_window(&mut self) {
        self.attempt = None;
        self.step = None;
        self.recording = false;
        self.lanes.clear();
    }

    /// Takes one call with a decoder's `tokens` at `now`, and returns the step it completes.
    pub fn observe(&mut self, tokens: &[RawToken], now: Instant) -> Option<Step> {
        let Ok(n) = u32::try_from(tokens.len()) else {
            return None;
        };
        if n == 0 {
            self.attempt = Some(self.attempt.map_or(0, |attempt| attempt + 1));
            self.step = None;
            self.recording = false;
            self.lanes.clear();
            return None;
        }
        if self.step == Some(n) {
            if self.recording && self.lanes.len() < MAX_BEAMS * 2 {
                self.lanes.push(tokens.to_vec());
            }
            return None;
        }
        let finished = match (self.step, self.recording && !self.lanes.is_empty()) {
            (Some(step), true) => Some(Step {
                attempt: self.attempt.unwrap_or(0),
                step,
                lanes: std::mem::take(&mut self.lanes),
            }),
            (Some(_) | None, _) => None,
        };
        self.lanes.clear();
        self.step = Some(n);
        self.recording = self.throttle.due(now);
        if self.recording {
            self.lanes.push(tokens.to_vec());
        }
        finished
    }
}

/// The best distinct hypotheses among `lanes`, each showing its newest text tokens.
///
/// Ids at or above `eot` (special and timestamp tokens) are left out of what a beam shows but
/// still count towards its log probability, as they do in whisper.cpp's beam score.
#[must_use]
pub fn beams(lanes: &[Vec<RawToken>], eot: i32) -> Vec<RawBeam> {
    let mut found: Vec<(&[RawToken], RawBeam)> = Vec::new();
    for lane in lanes {
        if found.iter().any(|(seen, _)| ids_match(seen, lane)) {
            continue;
        }
        let text: Vec<RawToken> = lane
            .iter()
            .filter(|token| token.id < eot)
            .copied()
            .collect();
        let shown: Vec<RawToken> = text
            .iter()
            .skip(text.len().saturating_sub(SHOWN_TOKENS))
            .copied()
            .collect();
        found.push((
            lane,
            RawBeam {
                tokens: shown,
                log_prob: lane.iter().map(|token| token.plog).sum(),
            },
        ));
    }
    let mut beams: Vec<RawBeam> = found.into_iter().map(|(_, beam)| beam).collect();
    beams.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
    beams.truncate(MAX_BEAMS);
    beams
}

fn ids_match(a: &[RawToken], b: &[RawToken]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.id == b.id)
}

/// Collects the beams of the windows a stream decodes and sends them as events.
pub struct BeamCollector {
    steps: Mutex<Steps>,
    window: Mutex<(f64, f64)>,
    eot: i32,
    out: UnboundedSender<crate::Result<StreamEvent>>,
}

impl BeamCollector {
    #[must_use]
    pub fn new(eot: i32, out: UnboundedSender<crate::Result<StreamEvent>>) -> Self {
        Self {
            steps: Mutex::new(Steps::new(Throttle::new(PUBLISH_EVERY))),
            window: Mutex::new((0.0, 0.0)),
            eot,
            out,
        }
    }

    /// Prepares for the window spanning `window` seconds of the source.
    pub fn start_window(&self, window: (f64, f64)) {
        self.steps.lock().start_window();
        *self.window.lock() = window;
    }

    fn observe(&self, ctx: *mut whisper_context, tokens: &[RawToken]) {
        let Some(step) = self.steps.lock().observe(tokens, Instant::now()) else {
            return;
        };
        let beams = beams(&step.lanes, self.eot)
            .into_iter()
            .map(|beam| Beam {
                tokens: beam
                    .tokens
                    .iter()
                    .map(|token| BeamToken {
                        text: token_text(ctx, token.id),
                        p: token.p,
                    })
                    .collect(),
                log_prob: beam.log_prob,
            })
            .collect();
        let snapshot = BeamSnapshot {
            window: *self.window.lock(),
            attempt: step.attempt,
            step: step.step,
            beams,
        };
        if self.out.send(Ok(StreamEvent::Beams(snapshot))).is_err() {
            debug!("Receiver dropped, beam snapshot not delivered");
        }
    }
}

fn token_text(ctx: *mut whisper_context, id: i32) -> String {
    // SAFETY: `ctx` is the live context whisper.cpp passed to the callback, and
    // `whisper_token_to_str` returns null or a NUL-terminated string owned by its vocabulary.
    let text = unsafe { whisper_token_to_str(ctx, id) };
    if text.is_null() {
        return String::new();
    }
    // SAFETY: checked non-null above; the vocabulary outlives this call.
    unsafe { CStr::from_ptr(text) }
        .to_string_lossy()
        .into_owned()
}

/// The logits-filter callback.
///
/// # Safety
///
/// `user_data` must point to a [`BeamCollector`] that outlives the `full()` call the callback
/// is installed on, and `tokens` must hold `n_tokens` entries, as whisper.cpp guarantees.
///
/// whisper.cpp calls it from its worker threads, several at once, so it only takes `&` of the
/// collector. It never reads or changes `logits`, and a panic is caught here because unwinding
/// into C is undefined behaviour.
pub unsafe extern "C" fn on_logits(
    ctx: *mut whisper_context,
    _state: *mut whisper_state,
    tokens: *const whisper_token_data,
    n_tokens: c_int,
    _logits: *mut f32,
    user_data: *mut c_void,
) {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the installer passes a pointer to a `BeamCollector` that stays alive until
        // `full()` returns, and no callback runs after that.
        let Some(collector) = (unsafe { user_data.cast::<BeamCollector>().as_ref() }) else {
            return;
        };
        let count = usize::try_from(n_tokens).unwrap_or(0);
        let data: &[whisper_token_data] = if count == 0 || tokens.is_null() {
            &[]
        } else {
            // SAFETY: whisper.cpp passes `n_tokens` valid entries at `tokens` for this call.
            unsafe { std::slice::from_raw_parts(tokens, count) }
        };
        let raw: Vec<RawToken> = data
            .iter()
            .map(|token| RawToken {
                id: token.id,
                p: token.p,
                plog: token.plog,
            })
            .collect();
        collector.observe(ctx, &raw);
    }));
    if outcome.is_err() {
        error!("The beam telemetry callback panicked; this step's beams are lost");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOT: i32 = 50_257;
    const BEG: i32 = 50_364;

    fn tokens(ids: &[i32], plog: f32) -> Vec<RawToken> {
        ids.iter()
            .map(|&id| RawToken { id, p: 0.5, plog })
            .collect()
    }

    fn at(start: Instant, ms: u64) -> Instant {
        start + Duration::from_millis(ms)
    }

    #[test]
    fn the_throttle_passes_one_event_per_interval() {
        let start = Instant::now();
        let mut throttle = Throttle::new(Duration::from_millis(100));
        let passed: Vec<bool> = [0, 40, 99, 100, 150, 230]
            .into_iter()
            .map(|ms| throttle.due(at(start, ms)))
            .collect();
        assert_eq!(passed, vec![true, false, false, true, false, true]);
    }

    #[test]
    fn a_step_ends_when_the_next_one_starts_and_carries_its_lanes() {
        let start = Instant::now();
        let mut steps = Steps::new(Throttle::new(Duration::from_millis(100)));
        assert_eq!(steps.observe(&[], start), None);
        assert_eq!(steps.observe(&tokens(&[BEG, 1], -0.1), start), None);
        assert_eq!(steps.observe(&tokens(&[BEG, 2], -0.2), start), None);
        assert_eq!(
            steps.observe(&tokens(&[BEG, 1, 3], -0.1), at(start, 10)),
            Some(Step {
                attempt: 0,
                step: 2,
                lanes: vec![tokens(&[BEG, 1], -0.1), tokens(&[BEG, 2], -0.2)],
            })
        );
    }

    #[test]
    fn steps_inside_the_interval_are_skipped() {
        let start = Instant::now();
        let mut steps = Steps::new(Throttle::new(Duration::from_millis(100)));
        steps.observe(&[], start);
        steps.observe(&tokens(&[1], -0.1), start);
        assert!(
            steps
                .observe(&tokens(&[1, 2], -0.1), at(start, 30))
                .is_some()
        );
        assert_eq!(
            steps.observe(&tokens(&[1, 2, 3], -0.1), at(start, 60)),
            None
        );
        assert_eq!(
            steps.observe(&tokens(&[1, 2, 3, 4], -0.1), at(start, 130)),
            None
        );
        assert_eq!(
            steps
                .observe(&tokens(&[1, 2, 3, 4, 5], -0.1), at(start, 140))
                .map(|step| step.step),
            Some(4)
        );
    }

    #[test]
    fn a_retry_restarts_the_lanes_and_counts_the_attempt() {
        let start = Instant::now();
        let mut steps = Steps::new(Throttle::new(Duration::ZERO));
        steps.observe(&[], start);
        steps.observe(&tokens(&[1], -0.1), start);
        steps.observe(&tokens(&[1, 2], -0.1), start);
        assert_eq!(steps.observe(&[], start), None);
        assert_eq!(steps.observe(&tokens(&[7], -0.3), start), None);
        assert_eq!(
            steps.observe(&tokens(&[7, 8], -0.3), start),
            Some(Step {
                attempt: 1,
                step: 1,
                lanes: vec![tokens(&[7], -0.3)],
            })
        );
    }

    #[test]
    fn a_new_window_starts_at_the_first_attempt() {
        let start = Instant::now();
        let mut steps = Steps::new(Throttle::new(Duration::ZERO));
        steps.observe(&[], start);
        steps.observe(&[], start);
        steps.start_window();
        steps.observe(&[], start);
        steps.observe(&tokens(&[1], -0.1), start);
        assert_eq!(
            steps
                .observe(&tokens(&[1, 2], -0.1), start)
                .map(|step| step.attempt),
            Some(0)
        );
    }

    #[test]
    fn beams_are_distinct_best_first_and_show_text_tokens_only() {
        let lanes = vec![
            tokens(&[BEG, 10, 11], -0.5),
            tokens(&[BEG, 10, 12], -0.125),
            tokens(&[BEG, 10, 12], -0.125),
            tokens(&[BEG, 20, BEG + 40], -0.25),
        ];
        assert_eq!(
            beams(&lanes, EOT),
            vec![
                RawBeam {
                    tokens: tokens(&[10, 12], -0.125),
                    log_prob: -0.375,
                },
                RawBeam {
                    tokens: tokens(&[20], -0.25),
                    log_prob: -0.75,
                },
                RawBeam {
                    tokens: tokens(&[10, 11], -0.5),
                    log_prob: -1.5,
                },
            ]
        );
    }

    #[test]
    fn a_beam_shows_only_its_newest_tokens_and_at_most_five_beams_show() {
        let long: Vec<i32> = (0..40).collect();
        let lanes: Vec<Vec<RawToken>> = (0..7)
            .map(|lane| {
                let mut ids = long.clone();
                ids.push(1_000 + lane);
                tokens(&ids, -0.01 * (lane as f32 + 1.0))
            })
            .collect();
        let found = beams(&lanes, EOT);
        assert_eq!(found.len(), MAX_BEAMS);
        assert_eq!(
            found[0].tokens.iter().map(|t| t.id).collect::<Vec<_>>(),
            (17..40).chain([1_000]).collect::<Vec<_>>()
        );
    }
}
