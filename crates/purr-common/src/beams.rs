//! The decoder's competing hypotheses while it decodes, for a live view of its search.

use serde::{Deserialize, Serialize};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// One token of a hypothesis.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BeamToken {
    /// The token's text; a token that holds part of a character shows a replacement character
    pub text: String,
    /// Probability the decoder gave the token
    pub p: f32,
}

/// One hypothesis of the decoder.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Beam {
    /// The latest text tokens of the hypothesis, oldest first
    pub tokens: Vec<BeamToken>,
    /// Sum of the log probabilities of every token of the hypothesis
    pub log_prob: f32,
}

/// The hypotheses the decoder held after one decoding step, best first.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BeamSnapshot {
    /// The seconds of the source the decoded window covers
    pub window: (f64, f64),
    /// How many times the decoder restarted this window at a higher temperature
    pub attempt: u32,
    /// Tokens decoded so far in this attempt
    pub step: u32,
    pub beams: Vec<Beam>,
}

/// Turns the beam view on and off while a transcription runs; a change applies from the next
/// decoded window.
#[derive(Debug, Clone, Default)]
pub struct BeamSwitch(Arc<AtomicBool>);

impl BeamSwitch {
    #[must_use]
    pub fn new(on: bool) -> Self {
        Self(Arc::new(AtomicBool::new(on)))
    }

    pub fn set(&self, on: bool) {
        self.0.store(on, Ordering::Relaxed);
    }

    #[must_use]
    pub fn is_on(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_switch_is_off_until_set_and_shared_by_its_clones() {
        let switch = BeamSwitch::default();
        let copy = switch.clone();
        assert!(!copy.is_on());
        switch.set(true);
        assert!(copy.is_on());
        assert!(BeamSwitch::new(true).is_on());
    }
}
