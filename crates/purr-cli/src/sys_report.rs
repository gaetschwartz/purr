//! The text of `purr sys` for the processor and the devices, as plain lines.

use purr_common::accel::{CpuFeature, CpuReport};
use purr_core::{
    accel::backend::AccelBackend,
    dev::{Device, DeviceType},
};

/// Format file size in human readable format
pub fn format_file_size(size: u64) -> String {
    let mut value = size as f64;
    let mut scaled_unit = None;
    for unit in ["KB", "MB", "GB"] {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        scaled_unit = Some(unit);
    }
    match scaled_unit {
        Some(unit) => format!("{value:.1} {unit}"),
        None => format!("{size} B"),
    }
}

const fn answer(answer: Option<bool>) -> &'static str {
    match answer {
        Some(true) => "yes",
        Some(false) => "no",
        None => "n/a",
    }
}

fn compiled_in(cpu: &CpuReport, feature: CpuFeature) -> Option<bool> {
    feature
        .reported_by_build()
        .then(|| cpu.built_with.contains(&feature))
}

fn on_this_cpu(cpu: &CpuReport, feature: CpuFeature) -> Option<bool> {
    let detected = cpu.detected.as_ref()?;
    feature
        .probed_at_runtime()
        .then(|| detected.contains(&feature))
}

/// The processor and a table of the instruction sets this binary was compiled with next to the
/// ones the processor reports.
pub fn cpu_lines(cpu: &CpuReport) -> Vec<String> {
    let name = cpu.name.as_deref().unwrap_or("name not reported");
    let threads = cpu.logical_threads.map_or_else(
        || "threads not reported".to_owned(),
        |n| format!("{n} threads"),
    );
    let mut lines = vec![
        format!("{name} ({threads})"),
        format!("  {:<12}{:<10}{}", "Instruction", "Compiled", "This CPU"),
    ];
    for feature in CpuFeature::ALL {
        lines.push(format!(
            "  {:<12}{:<10}{}",
            feature.label(),
            answer(compiled_in(cpu, feature)),
            answer(on_this_cpu(cpu, feature)),
        ));
    }
    let missing = cpu.missing_on_cpu();
    if !missing.is_empty() {
        lines.push(format!(
            "  Compiled in but not supported by this CPU: {}",
            CpuFeature::join_labels(&missing)
        ));
    }
    let unused = cpu.unused_by_build();
    if !unused.is_empty() {
        lines.push(format!(
            "  Supported by this CPU but not compiled in: {}",
            CpuFeature::join_labels(&unused)
        ));
    }
    if cpu.detected.is_none() {
        lines.push("  This CPU's instruction sets cannot be queried on this platform".to_owned());
    }
    lines
}

fn memory_line(device: &Device) -> String {
    let shared = match device.tpe {
        DeviceType::Igpu => " (integrated GPUs share system RAM)",
        DeviceType::Cpu | DeviceType::Gpu | DeviceType::Accel | DeviceType::Unknown => "",
    };
    let memory = if device.vram_total == 0 {
        "not reported".to_owned()
    } else if device.vram_free != 0 && device.vram_free <= device.vram_total {
        format!(
            "{} total, {} free",
            format_file_size(device.vram_total as u64),
            format_file_size(device.vram_free as u64)
        )
    } else {
        format!("{} total", format_file_size(device.vram_total as u64))
    };
    format!("  Memory: {memory}{shared}")
}

fn capabilities_line(device: &Device) -> String {
    match device.capability_names() {
        None => "  Capabilities: not reported".to_owned(),
        Some(names) if names.is_empty() => "  Capabilities: none".to_owned(),
        Some(names) => format!("  Capabilities: {}", names.join(", ")),
    }
}

/// One ggml device as ggml reports it, with the backends that would pick it automatically.
pub fn device_lines(device: &Device, picked_for: &[AccelBackend]) -> Vec<String> {
    let mut lines = vec![format!("Device {} - {}", device.id, device.name)];
    if !device.description.is_empty() {
        lines.push(format!("  Description: {}", device.description));
    }
    lines.push(format!("  Type: {}", device.tpe.label()));
    lines.push(format!(
        "  Backend: {}",
        device
            .backend
            .map_or("not selectable by Purr", AccelBackend::pretty_name)
    ));
    lines.push(format!(
        "  Device id: {}",
        device
            .device_id
            .as_deref()
            .unwrap_or("not reported by ggml")
    ));
    lines.push(memory_line(device));
    lines.push(capabilities_line(device));
    if device.backend == Some(AccelBackend::Vulkan) {
        lines.push(
            "  Driver: not reported by ggml (`vulkaninfo --summary` lists the driver and Vulkan version)"
                .to_owned(),
        );
    }
    if !picked_for.is_empty() {
        let names: Vec<_> = picked_for.iter().map(|b| b.pretty_name()).collect();
        lines.push(format!(
            "  Picked by automatic selection for: {}",
            names.join(", ")
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu(built_with: &[CpuFeature], detected: Option<&[CpuFeature]>) -> CpuReport {
        CpuReport {
            name: Some("Some Processor".to_owned()),
            logical_threads: Some(16),
            built_with: built_with.to_vec(),
            detected: detected.map(<[CpuFeature]>::to_vec),
        }
    }

    #[test]
    fn the_cpu_table_puts_the_compiled_instruction_sets_next_to_the_detected_ones() {
        use CpuFeature::{Avx, Avx2, Avx512Bw, Avx512F, Avx512Vl, Bmi2, F16c, Fma};
        let lines = cpu_lines(&cpu(
            &[Avx, Avx2, Fma, F16c],
            Some(&[Avx, Avx2, Fma, F16c, Bmi2, Avx512F, Avx512Bw, Avx512Vl]),
        ));
        assert_eq!(
            lines,
            [
                "Some Processor (16 threads)",
                "  Instruction Compiled  This CPU",
                "  AVX         yes       yes",
                "  AVX-VNNI    no        no",
                "  AVX2        yes       yes",
                "  BMI2        no        yes",
                "  FMA         yes       yes",
                "  F16C        yes       yes",
                "  AVX512F     no        yes",
                "  AVX512BW    n/a       yes",
                "  AVX512VL    n/a       yes",
                "  AVX512VBMI  no        no",
                "  AVX512VNNI  no        no",
                "  AVX512BF16  no        no",
                "  AMX-INT8    no        n/a",
                "  Supported by this CPU but not compiled in: BMI2, AVX512F",
            ]
        );
    }

    #[test]
    fn a_build_needing_what_the_processor_lacks_is_called_out() {
        let lines = cpu_lines(&cpu(
            &[CpuFeature::Avx2, CpuFeature::Avx512F],
            Some(&[CpuFeature::Avx2]),
        ));
        assert_eq!(
            lines.last().map(String::as_str),
            Some("  Compiled in but not supported by this CPU: AVX512F")
        );
    }

    #[test]
    fn a_processor_that_cannot_be_queried_says_so_instead_of_answering_no() {
        let lines = cpu_lines(&cpu(&[CpuFeature::Avx2], None));
        assert_eq!(
            lines.get(2..4),
            Some(
                [
                    "  AVX         no        n/a".to_owned(),
                    "  AVX-VNNI    no        n/a".to_owned(),
                ]
                .as_slice()
            )
        );
        assert_eq!(
            lines.last().map(String::as_str),
            Some("  This CPU's instruction sets cannot be queried on this platform")
        );
    }

    #[test]
    fn an_unnamed_processor_is_reported_as_such() {
        let report = CpuReport {
            name: None,
            logical_threads: None,
            built_with: vec![],
            detected: None,
        };
        assert_eq!(
            cpu_lines(&report).first().map(String::as_str),
            Some("name not reported (threads not reported)")
        );
    }

    fn device(tpe: DeviceType, device_id: Option<&str>) -> Device {
        Device {
            id: 2,
            name: "Vulkan0".to_owned(),
            description: "Intel(R) Iris(R) Xe Graphics".to_owned(),
            device_id: device_id.map(str::to_owned),
            vram_free: 3 << 30,
            vram_total: 8 << 30,
            tpe,
            backend: Some(AccelBackend::Vulkan),
            caps: None,
        }
    }

    #[test]
    fn an_integrated_gpu_is_labelled_as_sharing_system_memory() {
        assert_eq!(
            device_lines(&device(DeviceType::Igpu, Some("0000:00:02.0")), &[]),
            [
                "Device 2 - Vulkan0",
                "  Description: Intel(R) Iris(R) Xe Graphics",
                "  Type: integrated GPU",
                "  Backend: Vulkan",
                "  Device id: 0000:00:02.0",
                "  Memory: 8.0 GB total, 3.0 GB free (integrated GPUs share system RAM)",
                "  Capabilities: not reported",
                "  Driver: not reported by ggml (`vulkaninfo --summary` lists the driver and Vulkan version)",
            ]
        );
    }

    #[test]
    fn a_discrete_gpu_without_an_id_that_automatic_selection_picks_says_so() {
        assert_eq!(
            device_lines(
                &device(DeviceType::Gpu, None),
                &[AccelBackend::Vulkan, AccelBackend::Cuda]
            ),
            [
                "Device 2 - Vulkan0",
                "  Description: Intel(R) Iris(R) Xe Graphics",
                "  Type: discrete GPU",
                "  Backend: Vulkan",
                "  Device id: not reported by ggml",
                "  Memory: 8.0 GB total, 3.0 GB free",
                "  Capabilities: not reported",
                "  Driver: not reported by ggml (`vulkaninfo --summary` lists the driver and Vulkan version)",
                "  Picked by automatic selection for: Vulkan, CUDA",
            ]
        );
    }

    #[test]
    fn a_cpu_device_has_no_description_line_when_ggml_gives_none() {
        let cpu = Device {
            id: 0,
            name: "CPU".to_owned(),
            description: String::new(),
            device_id: None,
            vram_free: 0,
            vram_total: 0,
            tpe: DeviceType::Cpu,
            backend: Some(AccelBackend::Cpu),
            caps: None,
        };
        assert_eq!(
            device_lines(&cpu, &[]),
            [
                "Device 0 - CPU",
                "  Type: CPU",
                "  Backend: CPU",
                "  Device id: not reported by ggml",
                "  Memory: not reported",
                "  Capabilities: not reported",
            ]
        );
    }

    #[test]
    fn a_device_purr_does_not_select_between_is_marked() {
        let blas = Device {
            backend: None,
            ..device(DeviceType::Accel, None)
        };
        assert_eq!(
            device_lines(&blas, &[]).get(3).map(String::as_str),
            Some("  Backend: not selectable by Purr")
        );
    }

    #[test]
    fn file_sizes_scale_through_the_units() {
        assert_eq!(format_file_size(512), "512 B");
        assert_eq!(format_file_size(1536), "1.5 KB");
        assert_eq!(format_file_size(3 << 30), "3.0 GB");
    }
}
