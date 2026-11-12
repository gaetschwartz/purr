//! Whether a browser's WebGPU can be trusted with a model.
//!
//! A known recording is transcribed on the GPU the first time a model runs on an adapter,
//! browser and runtime; when the text comes out wrong the CPU takes over. The verdict is kept
//! per model with a signature of what produced it and checked again when the signature changes
//! or the verdict grows old, so a browser or driver that fixes its WebGPU is picked up again.
//! When Purr picks the engine, the recording is also timed on both engines and a model both can
//! run goes to the faster one.

use purr_common::accel::{AccelBackend, AccelPolicy};
use purr_common::browser_engine::{CpuReason, EnginePlan};
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
    pub shader_f16: bool,
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

/// How the CPU handled the check recording.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CpuCanary {
    Took {
        ms: f64,
    },
    /// The CPU could not load or run the model.
    Failed,
    /// The CPU was not timed: the model does not run on it, or the settings did not let Purr choose.
    Skipped,
}

/// How long each engine took to transcribe the check recording, with the models loaded.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CanaryTimes {
    pub gpu_ms: f64,
    pub cpu: CpuCanary,
}

impl CanaryTimes {
    #[must_use]
    pub fn cpu_is_faster(&self) -> bool {
        match self.cpu {
            CpuCanary::Took { ms } => ms < self.gpu_ms,
            CpuCanary::Failed | CpuCanary::Skipped => false,
        }
    }
}

/// A stored verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckRecord {
    pub signature: Signature,
    pub verdict: CanaryVerdict,
    pub checked_at_ms: f64,
    #[serde(default)]
    pub times: Option<CanaryTimes>,
}

impl CheckRecord {
    /// Whether the record was made for `signature` and is recent.
    #[must_use]
    pub fn holds_for(&self, signature: &Signature, now_ms: f64) -> bool {
        let age = now_ms - self.checked_at_ms;
        self.signature == *signature && (0.0..RECHECK_AFTER_MS).contains(&age)
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

/// What `plan` needs to know about a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelNeeds<'a> {
    pub id: &'a str,
    pub shader_f16: bool,
    pub runs_on_cpu: bool,
}

/// Where the next run of a model goes, given the settings, the adapter the worker found, and
/// the stored verdict for that model.
#[must_use]
pub fn plan(
    policy: AccelPolicy,
    probe: Option<&GpuProbe>,
    model: ModelNeeds<'_>,
    record: Option<&CheckRecord>,
    now_ms: f64,
) -> EnginePlan {
    let cpu_chosen = matches!(
        policy,
        AccelPolicy::Forced {
            backend: AccelBackend::Cpu,
            ..
        }
    );
    let fresh = probe.and_then(|probe| {
        record.filter(|record| record.holds_for(&Signature::new(probe, model.id), now_ms))
    });
    let verdict = fresh.map(|record| record.verdict);
    let times = fresh.and_then(|record| record.times);
    let cpu_reason = match (cpu_chosen, probe, verdict) {
        (true, _, _) => Some(CpuReason::Chosen),
        (false, None, _) => Some(CpuReason::NoWebGpu),
        (false, Some(probe), _) if model.shader_f16 && !probe.shader_f16 => {
            Some(CpuReason::NoShaderF16)
        }
        (false, Some(_), Some(CanaryVerdict::Wrong)) => Some(CpuReason::WrongOutput),
        (false, Some(_), Some(CanaryVerdict::Failed)) => Some(CpuReason::WebGpuFailed),
        (false, Some(_), Some(CanaryVerdict::Correct))
            if times_the_cpu(policy, model) && times.is_some_and(|times| times.cpu_is_faster()) =>
        {
            Some(CpuReason::Faster)
        }
        (false, Some(_), Some(CanaryVerdict::Correct) | None) => None,
    };
    match (cpu_reason, verdict) {
        (Some(reason), _) if model.runs_on_cpu => EnginePlan::Cpu(reason),
        (Some(reason), _) => EnginePlan::CannotRun(reason),
        (None, Some(CanaryVerdict::Correct)) if speed_is_known(policy, model, times) => {
            EnginePlan::WebGpu
        }
        (
            None,
            Some(CanaryVerdict::Correct | CanaryVerdict::Wrong | CanaryVerdict::Failed) | None,
        ) => EnginePlan::WebGpuAfterCheck,
    }
}

/// Whether the stored times settle which engine is faster, as far as `policy` lets Purr choose.
fn speed_is_known(policy: AccelPolicy, model: ModelNeeds<'_>, times: Option<CanaryTimes>) -> bool {
    match times.map(|times| times.cpu) {
        None => false,
        Some(CpuCanary::Skipped) => !times_the_cpu(policy, model),
        Some(CpuCanary::Took { .. } | CpuCanary::Failed) => true,
    }
}

/// Whether the check should also time the CPU, so the faster engine can be chosen.
#[must_use]
pub fn times_the_cpu(policy: AccelPolicy, model: ModelNeeds<'_>) -> bool {
    policy == AccelPolicy::Auto && model.runs_on_cpu
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY_MS: f64 = 24.0 * 60.0 * 60.0 * 1000.0;

    const BASE: ModelNeeds<'static> = ModelNeeds {
        id: "base",
        shader_f16: false,
        runs_on_cpu: true,
    };
    const TURBO_Q4F16: ModelNeeds<'static> = ModelNeeds {
        id: "turbo-q4f16",
        shader_f16: true,
        runs_on_cpu: true,
    };
    const TURBO_FP16: ModelNeeds<'static> = ModelNeeds {
        id: "turbo",
        shader_f16: true,
        runs_on_cpu: false,
    };

    fn probe(fingerprint: &str) -> GpuProbe {
        GpuProbe {
            fingerprint: fingerprint.to_owned(),
            shader_f16: true,
        }
    }

    fn without_f16(fingerprint: &str) -> GpuProbe {
        GpuProbe {
            shader_f16: false,
            ..probe(fingerprint)
        }
    }

    const GPU_FASTER: CanaryTimes = CanaryTimes {
        gpu_ms: 400.0,
        cpu: CpuCanary::Took { ms: 900.0 },
    };

    fn record(probe: &GpuProbe, model_id: &str, verdict: CanaryVerdict, at: f64) -> CheckRecord {
        CheckRecord {
            signature: Signature::new(probe, model_id),
            verdict,
            checked_at_ms: at,
            times: (verdict == CanaryVerdict::Correct).then_some(GPU_FASTER),
        }
    }

    fn timed(probe: &GpuProbe, model_id: &str, times: Option<CanaryTimes>) -> CheckRecord {
        CheckRecord {
            times,
            ..record(probe, model_id, CanaryVerdict::Correct, 0.0)
        }
    }

    #[test]
    fn the_faster_engine_runs_a_model_both_can_run() {
        let gpu = probe("adapter");
        for (cpu, expected) in [
            (
                CpuCanary::Took { ms: 300.0 },
                EnginePlan::Cpu(CpuReason::Faster),
            ),
            (CpuCanary::Took { ms: 900.0 }, EnginePlan::WebGpu),
            (CpuCanary::Took { ms: 400.0 }, EnginePlan::WebGpu),
            (CpuCanary::Failed, EnginePlan::WebGpu),
        ] {
            let stored = timed(&gpu, "base", Some(CanaryTimes { gpu_ms: 400.0, cpu }));
            assert_eq!(
                plan(AccelPolicy::Auto, Some(&gpu), BASE, Some(&stored), 1.0),
                expected,
                "{cpu:?}"
            );
        }
    }

    #[test]
    fn a_correct_verdict_without_the_cpu_time_is_measured_again() {
        let gpu = probe("adapter");
        let skipped = Some(CanaryTimes {
            gpu_ms: 400.0,
            cpu: CpuCanary::Skipped,
        });
        for times in [None, skipped] {
            let stored = timed(&gpu, "base", times);
            assert_eq!(
                plan(AccelPolicy::Auto, Some(&gpu), BASE, Some(&stored), 1.0),
                EnginePlan::WebGpuAfterCheck,
                "{times:?}"
            );
        }
        let gpu_only = timed(&gpu, "turbo", skipped);
        assert_eq!(
            plan(
                AccelPolicy::Auto,
                Some(&gpu),
                TURBO_FP16,
                Some(&gpu_only),
                1.0
            ),
            EnginePlan::WebGpu
        );
    }

    #[test]
    fn speed_only_decides_when_purr_picks_the_engine() {
        let gpu = probe("adapter");
        let cpu_faster = timed(
            &gpu,
            "base",
            Some(CanaryTimes {
                gpu_ms: 400.0,
                cpu: CpuCanary::Took { ms: 100.0 },
            }),
        );
        let gpu_forced = AccelPolicy::Forced {
            backend: AccelBackend::Vulkan,
            allow_fallback: true,
        };
        assert_eq!(
            plan(gpu_forced, Some(&gpu), BASE, Some(&cpu_faster), 1.0),
            EnginePlan::WebGpu
        );
        assert!(!times_the_cpu(gpu_forced, BASE));
        assert!(!times_the_cpu(AccelPolicy::Auto, TURBO_FP16));
        assert!(times_the_cpu(AccelPolicy::Auto, BASE));
    }

    #[test]
    fn records_kept_before_timing_existed_still_load() -> Result<(), serde_json::Error> {
        let gpu = probe("adapter");
        let json = format!(
            r#"{{"signature":"{}","verdict":"Correct","checked_at_ms":5.0}}"#,
            Signature::new(&gpu, "base").as_str()
        );
        let decoded: CheckRecord = serde_json::from_str(&json)?;
        assert_eq!(
            decoded,
            CheckRecord {
                checked_at_ms: 5.0,
                ..timed(&gpu, "base", None)
            }
        );
        Ok(())
    }

    #[test]
    fn choosing_the_cpu_wins_over_everything() {
        let gpu = probe("adapter");
        let correct = record(&gpu, "base", CanaryVerdict::Correct, 0.0);
        assert_eq!(
            plan(AccelPolicy::CPU_ONLY, Some(&gpu), BASE, Some(&correct), 1.0),
            EnginePlan::Cpu(CpuReason::Chosen)
        );
    }

    #[test]
    fn no_adapter_or_a_missing_feature_means_the_cpu() {
        assert_eq!(
            plan(AccelPolicy::Auto, None, BASE, None, 0.0),
            EnginePlan::Cpu(CpuReason::NoWebGpu)
        );
        assert_eq!(
            plan(
                AccelPolicy::Auto,
                Some(&without_f16("a")),
                TURBO_Q4F16,
                None,
                0.0
            ),
            EnginePlan::Cpu(CpuReason::NoShaderF16)
        );
        assert_eq!(
            plan(AccelPolicy::Auto, Some(&without_f16("a")), BASE, None, 0.0),
            EnginePlan::WebGpuAfterCheck
        );
    }

    #[test]
    fn a_model_the_cpu_cannot_run_is_refused_instead_of_moved() {
        let gpu = probe("adapter");
        let wrong = record(&gpu, "turbo", CanaryVerdict::Wrong, 0.0);
        for (policy, found, stored, reason) in [
            (AccelPolicy::CPU_ONLY, Some(&gpu), None, CpuReason::Chosen),
            (AccelPolicy::Auto, None, None, CpuReason::NoWebGpu),
            (
                AccelPolicy::Auto,
                Some(&gpu),
                Some(&wrong),
                CpuReason::WrongOutput,
            ),
        ] {
            assert_eq!(
                plan(policy, found, TURBO_FP16, stored, 1.0),
                EnginePlan::CannotRun(reason)
            );
        }
        assert_eq!(
            plan(AccelPolicy::Auto, Some(&gpu), TURBO_FP16, None, 1.0),
            EnginePlan::WebGpuAfterCheck
        );
    }

    #[test]
    fn a_fresh_verdict_for_the_same_signature_is_reused() {
        let gpu = probe("adapter");
        let now = 10.0 * DAY_MS;
        for (verdict, expected) in [
            (CanaryVerdict::Correct, EnginePlan::WebGpu),
            (
                CanaryVerdict::Wrong,
                EnginePlan::Cpu(CpuReason::WrongOutput),
            ),
            (
                CanaryVerdict::Failed,
                EnginePlan::Cpu(CpuReason::WebGpuFailed),
            ),
        ] {
            let stored = record(&gpu, "base", verdict, now - DAY_MS);
            assert_eq!(
                plan(AccelPolicy::Auto, Some(&gpu), BASE, Some(&stored), now),
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
            plan(AccelPolicy::Auto, Some(&after), BASE, Some(&stored), 1.0),
            EnginePlan::WebGpuAfterCheck
        );
        let tiny = ModelNeeds { id: "tiny", ..BASE };
        assert_eq!(
            plan(AccelPolicy::Auto, Some(&before), tiny, Some(&stored), 1.0),
            EnginePlan::WebGpuAfterCheck
        );
    }

    #[test]
    fn an_old_verdict_or_one_from_the_future_is_checked_again() {
        let gpu = probe("adapter");
        let stored = record(&gpu, "base", CanaryVerdict::Wrong, 100.0 * DAY_MS);
        for now in [100.0f64.mul_add(DAY_MS, RECHECK_AFTER_MS), 99.0 * DAY_MS] {
            assert_eq!(
                plan(AccelPolicy::Auto, Some(&gpu), BASE, Some(&stored), now),
                EnginePlan::WebGpuAfterCheck
            );
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
    fn records_round_trip_through_storage() -> Result<(), serde_json::Error> {
        let stored = record(&probe("adapter"), "base", CanaryVerdict::Wrong, 5.0);
        let decoded: CheckRecord = serde_json::from_str(&serde_json::to_string(&stored)?)?;
        assert_eq!(decoded, stored);
        Ok(())
    }
}
