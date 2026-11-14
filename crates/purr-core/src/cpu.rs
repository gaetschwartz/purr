//! The instruction sets the running processor has, next to the ones the engine was compiled with.

use purr_common::accel::{CpuFeature, CpuReport};

use crate::accel::inventory::Inventory;
use crate::dev::Device;
use purr_common::platform::DeviceType;

fn present(has: impl Fn(CpuFeature) -> bool) -> Vec<CpuFeature> {
    CpuFeature::ALL
        .into_iter()
        .filter(|feature| has(*feature))
        .collect()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpu_has(feature: CpuFeature) -> bool {
    use std::arch::is_x86_feature_detected as detected;
    match feature {
        CpuFeature::Avx => detected!("avx"),
        CpuFeature::AvxVnni => detected!("avxvnni"),
        CpuFeature::Avx2 => detected!("avx2"),
        CpuFeature::Bmi2 => detected!("bmi2"),
        CpuFeature::Fma => detected!("fma"),
        CpuFeature::F16c => detected!("f16c"),
        CpuFeature::Avx512F => detected!("avx512f"),
        CpuFeature::Avx512Bw => detected!("avx512bw"),
        CpuFeature::Avx512Vl => detected!("avx512vl"),
        CpuFeature::Avx512Vbmi => detected!("avx512vbmi"),
        CpuFeature::Avx512Vnni => detected!("avx512vnni"),
        CpuFeature::Avx512Bf16 => detected!("avx512bf16"),
        CpuFeature::AmxInt8 => false,
    }
}

/// Instruction sets the running processor reports; `None` where it cannot be asked.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[must_use]
pub fn detected() -> Option<Vec<CpuFeature>> {
    Some(present(|feature| {
        feature.probed_at_runtime() && cpu_has(feature)
    }))
}

/// Instruction sets the running processor reports; `None` where it cannot be asked.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[must_use]
pub fn detected() -> Option<Vec<CpuFeature>> {
    None
}

fn cpu_name(devices: &[Device]) -> Option<String> {
    devices
        .iter()
        .find(|device| device.tpe == DeviceType::Cpu)
        .map(|device| device.description.trim())
        .filter(|name| !name.is_empty() && !name.eq_ignore_ascii_case("cpu"))
        .map(str::to_owned)
}

/// The processor's name, thread count and instruction sets, compiled in and available.
#[must_use]
pub fn report(inventory: &Inventory) -> CpuReport {
    CpuReport {
        name: cpu_name(&inventory.devices),
        logical_threads: std::thread::available_parallelism()
            .ok()
            .and_then(|threads| u32::try_from(threads.get()).ok()),
        built_with: inventory.build.cpu.clone(),
        detected: detected(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accel::{backend::AccelBackend, build_info::BuildInfo};

    fn cpu_device(description: &str) -> Device {
        Device {
            id: 0,
            name: "CPU".to_owned(),
            description: description.to_owned(),
            device_id: None,
            vram_free: 0,
            vram_total: 0,
            tpe: DeviceType::Cpu,
            backend: Some(AccelBackend::Cpu),
            caps: None,
            vulkan_driver: None,
        }
    }

    #[test]
    fn instruction_sets_are_listed_in_a_stable_order() {
        let features = present(|feature| {
            matches!(
                feature,
                CpuFeature::F16c | CpuFeature::Avx | CpuFeature::Avx2
            )
        });
        assert_eq!(
            features,
            [CpuFeature::Avx, CpuFeature::Avx2, CpuFeature::F16c]
        );
    }

    #[test]
    fn nothing_present_lists_nothing() {
        assert_eq!(present(|_| false), []);
    }

    #[test]
    fn the_report_takes_the_compiled_in_sets_from_the_engine_build() {
        let inventory = Inventory {
            devices: vec![cpu_device("A processor")],
            build: BuildInfo {
                cpu: vec![CpuFeature::Avx, CpuFeature::Fma],
                ..BuildInfo::default()
            },
            ..Inventory::default()
        };
        let report = report(&inventory);
        assert_eq!(report.name, Some("A processor".to_owned()));
        assert_eq!(report.built_with, [CpuFeature::Avx, CpuFeature::Fma]);
    }

    #[test]
    fn features_std_cannot_probe_are_never_detected() {
        assert_eq!(
            detected()
                .into_iter()
                .flatten()
                .filter(|feature| !feature.probed_at_runtime())
                .collect::<Vec<_>>(),
            []
        );
    }

    #[test]
    fn the_processor_is_named_by_its_device_description() {
        assert_eq!(
            cpu_name(&[cpu_device("  AMD Ryzen 9 7950X 16-Core Processor ")]),
            Some("AMD Ryzen 9 7950X 16-Core Processor".to_owned())
        );
    }

    #[test]
    fn a_processor_the_system_does_not_name_stays_unnamed() {
        assert_eq!(cpu_name(&[cpu_device("CPU")]), None);
        assert_eq!(cpu_name(&[cpu_device("  ")]), None);
        assert_eq!(cpu_name(&[]), None);
    }
}
