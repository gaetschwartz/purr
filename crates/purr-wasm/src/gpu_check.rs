//! Whether a browser's WebGPU can be trusted with a model.
//!
//! A known recording is transcribed on the GPU the first time a model runs on an adapter,
//! browser and runtime; when the text comes out wrong the CPU takes over. The verdict is kept
//! per model with a signature of what produced it and checked again when the signature changes
//! or the verdict grows old, so a browser or driver that fixes its WebGPU is picked up again.

use purr_common::accel::{AccelBackend, AccelPolicy};
use purr_common::browser_engine::CpuReason;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

/// What the check recording says.
pub const CANARY_TRANSCRIPT: &str =
    "And so my fellow Americans, ask not what your country can do for you";

/// The share of the expected words a correct transcript contains, and of its own words that are expected.
const MIN_MATCHING_SHARE: f64 = 0.6;

/// How long a verdict holds before the GPU is checked again.
pub const RECHECK_AFTER_MS: f64 = 7.0 * 24.0 * 60.0 * 60.0 * 1000.0;

/// How the GPU handled the check recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CanaryVerdict {
    Correct,
    /// The GPU ran the model and wrote the wrong text.
    Wrong,
    /// WebGPU failed while loading or running the model.
    Failed,
}

/// The WebGPU adapter the worker found, as far as the check cares.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct GpuProbe {
    /// Everything the browser says about the adapter, the browser and the runtime, as one opaque
    /// string; it is only compared, never read.
    pub fingerprint: String,
}

/// What a verdict was given for: the adapter, browser and runtime fingerprint, and the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Signature(String);

impl Signature {
    #[must_use]
    pub fn new(probe: &GpuProbe, model_id: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(probe.fingerprint.as_bytes());
        hasher.update([0]);
        hasher.update(model_id.as_bytes());
        Self(hex::encode(hasher.finalize()))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A stored verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckRecord {
    pub signature: Signature,
    pub verdict: CanaryVerdict,
    pub checked_at_ms: f64,
}

impl CheckRecord {
    /// The verdict, while it was given for `signature` and is recent.
    #[must_use]
    pub fn verdict_for(&self, signature: &Signature, now_ms: f64) -> Option<CanaryVerdict> {
        let age = now_ms - self.checked_at_ms;
        (self.signature == *signature && (0.0..RECHECK_AFTER_MS).contains(&age))
            .then_some(self.verdict)
    }
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Whether `text`, the GPU's transcript of the check recording, says what the recording says.
#[must_use]
pub fn judge(text: &str) -> CanaryVerdict {
    let expected = words(CANARY_TRANSCRIPT);
    let mut unmatched = words(text);
    let written = unmatched.len();
    let mut matching = 0usize;
    for word in &expected {
        if let Some(position) = unmatched.iter().position(|candidate| candidate == word) {
            unmatched.swap_remove(position);
            matching += 1;
        }
    }
    let share = |of: usize| {
        if of == 0 {
            0.0
        } else {
            matching as f64 / of as f64
        }
    };
    if share(expected.len()) >= MIN_MATCHING_SHARE && share(written) >= MIN_MATCHING_SHARE {
        CanaryVerdict::Correct
    } else {
        CanaryVerdict::Wrong
    }
}

/// Where a model should run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plan {
    /// On the GPU; `check` asks for the check recording first.
    Gpu {
        check: bool,
    },
    Cpu(CpuReason),
}

/// Where the model `model_id` should run, given the settings, the adapter the worker found, and
/// the stored verdict for that model.
///
/// Models that need `shader-f16` are never run on the CPU, so they always go to WebGPU, unchecked.
#[must_use]
pub fn plan(
    policy: AccelPolicy,
    probe: Option<&GpuProbe>,
    model_id: &str,
    needs_shader_f16: bool,
    record: Option<&CheckRecord>,
    now_ms: f64,
) -> Plan {
    if needs_shader_f16 {
        return Plan::Gpu { check: false };
    }
    if matches!(
        policy,
        AccelPolicy::Forced {
            backend: AccelBackend::Cpu,
            ..
        }
    ) {
        return Plan::Cpu(CpuReason::Chosen);
    }
    let Some(probe) = probe else {
        return Plan::Cpu(CpuReason::NoWebGpu);
    };
    let signature = Signature::new(probe, model_id);
    match record.and_then(|record| record.verdict_for(&signature, now_ms)) {
        Some(CanaryVerdict::Correct) => Plan::Gpu { check: false },
        Some(CanaryVerdict::Wrong) => Plan::Cpu(CpuReason::WrongOutput),
        Some(CanaryVerdict::Failed) => Plan::Cpu(CpuReason::WebGpuFailed),
        None => Plan::Gpu { check: true },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY_MS: f64 = 24.0 * 60.0 * 60.0 * 1000.0;

    fn probe(fingerprint: &str) -> GpuProbe {
        GpuProbe {
            fingerprint: fingerprint.to_owned(),
        }
    }

    fn record(probe: &GpuProbe, model_id: &str, verdict: CanaryVerdict, at: f64) -> CheckRecord {
        CheckRecord {
            signature: Signature::new(probe, model_id),
            verdict,
            checked_at_ms: at,
        }
    }

    #[test]
    fn transcripts_from_working_gpus_and_cpus_pass() {
        for text in [
            " And so my fellow Americans, ask not what your country can do for you,",
            " And so my fellow Americans ask not what your country can do for you",
            " And so, my fellow Americans, ask not what your country can do for you.",
            " and so my fellow americans ask not what your country can do",
        ] {
            assert_eq!(judge(text), CanaryVerdict::Correct, "{text}");
        }
    }

    #[test]
    fn transcripts_from_the_broken_firefox_runtime_fail() {
        for text in [
            " >>",
            ",",
            " [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [",
            "",
            " ? ? ? ? ? ?",
            "\u{fffd}\u{fffd}\u{fffd}",
        ] {
            assert_eq!(judge(text), CanaryVerdict::Wrong, "{text:?}");
        }
    }

    #[test]
    fn a_repeated_word_or_a_runaway_transcript_fails() {
        assert_eq!(judge(&" you".repeat(40)), CanaryVerdict::Wrong);
        let runaway = format!("{CANARY_TRANSCRIPT}{}", " fellow Americans".repeat(20));
        assert_eq!(judge(&runaway), CanaryVerdict::Wrong);
    }

    #[test]
    fn choosing_the_cpu_wins_over_everything() {
        let gpu = probe("adapter");
        assert_eq!(
            plan(AccelPolicy::CPU_ONLY, Some(&gpu), "base", false, None, 0.0),
            Plan::Cpu(CpuReason::Chosen)
        );
    }

    #[test]
    fn no_adapter_means_the_cpu() {
        assert_eq!(
            plan(AccelPolicy::Auto, None, "base", false, None, 0.0),
            Plan::Cpu(CpuReason::NoWebGpu)
        );
        let gpu = probe("adapter");
        assert_eq!(
            plan(AccelPolicy::Auto, Some(&gpu), "base", false, None, 0.0),
            Plan::Gpu { check: true }
        );
    }

    #[test]
    fn half_precision_models_stay_on_webgpu_whatever_happens() {
        let gpu = probe("adapter");
        let wrong = record(&gpu, "turbo", CanaryVerdict::Wrong, 0.0);
        for (policy, found, stored) in [
            (AccelPolicy::CPU_ONLY, Some(&gpu), None),
            (AccelPolicy::Auto, None, None),
            (AccelPolicy::Auto, Some(&gpu), Some(&wrong)),
        ] {
            assert_eq!(
                plan(policy, found, "turbo", true, stored, 1.0),
                Plan::Gpu { check: false }
            );
        }
    }

    #[test]
    fn a_fresh_verdict_for_the_same_signature_is_reused() {
        let gpu = probe("adapter");
        let now = 10.0 * DAY_MS;
        for (verdict, expected) in [
            (CanaryVerdict::Correct, Plan::Gpu { check: false }),
            (CanaryVerdict::Wrong, Plan::Cpu(CpuReason::WrongOutput)),
            (CanaryVerdict::Failed, Plan::Cpu(CpuReason::WebGpuFailed)),
        ] {
            let stored = record(&gpu, "base", verdict, now - DAY_MS);
            assert_eq!(
                plan(
                    AccelPolicy::Auto,
                    Some(&gpu),
                    "base",
                    false,
                    Some(&stored),
                    now
                ),
                expected
            );
        }
    }

    #[test]
    fn a_new_browser_adapter_runtime_or_model_checks_again() {
        let before = probe("firefox 160, runtime 1.22");
        let after = probe("firefox 161, runtime 1.22");
        let stored = record(&before, "base", CanaryVerdict::Wrong, 0.0);
        assert_eq!(
            plan(
                AccelPolicy::Auto,
                Some(&after),
                "base",
                false,
                Some(&stored),
                1.0
            ),
            Plan::Gpu { check: true }
        );
        assert_eq!(
            plan(
                AccelPolicy::Auto,
                Some(&before),
                "tiny",
                false,
                Some(&stored),
                1.0
            ),
            Plan::Gpu { check: true }
        );
    }

    #[test]
    fn an_old_verdict_or_one_from_the_future_is_checked_again() {
        let gpu = probe("adapter");
        let stored = record(&gpu, "base", CanaryVerdict::Wrong, 100.0 * DAY_MS);
        for now in [100.0f64.mul_add(DAY_MS, RECHECK_AFTER_MS), 99.0 * DAY_MS] {
            assert_eq!(
                plan(
                    AccelPolicy::Auto,
                    Some(&gpu),
                    "base",
                    false,
                    Some(&stored),
                    now
                ),
                Plan::Gpu { check: true }
            );
        }
    }

    #[test]
    fn records_round_trip_through_storage() -> Result<(), serde_json::Error> {
        let stored = record(&probe("adapter"), "base", CanaryVerdict::Wrong, 5.0);
        let decoded: CheckRecord = serde_json::from_str(&serde_json::to_string(&stored)?)?;
        assert_eq!(decoded, stored);
        Ok(())
    }
}
