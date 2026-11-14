use purr_common::accel::CpuFeature;
use whisper_rs::whisper_rs_sys as ggml;

use purr_core::accel::{
    backend::{AccelBackend, BuildStatus},
    build_info::BuildInfo,
};

/// What whisper.cpp and its ggml were built with in this binary.
#[must_use]
pub fn build_info() -> BuildInfo {
    BuildInfo {
        backends: AccelBackend::ALL
            .iter()
            .map(|backend| (*backend, backend_status(*backend)))
            .collect(),
        cpu: built_with(),
    }
}

const fn backend_status(backend: AccelBackend) -> BuildStatus {
    match backend {
        AccelBackend::Cuda => feature_status(cfg!(feature = "cuda")),
        AccelBackend::CoreMl => feature_status(cfg!(feature = "coreml")),
        AccelBackend::Metal => feature_status(cfg!(feature = "metal")),
        AccelBackend::Vulkan => feature_status(cfg!(feature = "vulkan")),
        AccelBackend::Cpu => BuildStatus::Built,
        AccelBackend::Hip | AccelBackend::Sycl => BuildStatus::Unknown,
    }
}

const fn feature_status(enabled: bool) -> BuildStatus {
    if enabled {
        BuildStatus::Built
    } else {
        BuildStatus::NotBuilt
    }
}

/// ggml answers these at compile time: it says what the CPU backend was built for.
fn built_with_ggml(feature: CpuFeature) -> bool {
    let flag = match feature {
        CpuFeature::Avx => unsafe { ggml::ggml_cpu_has_avx() },
        CpuFeature::AvxVnni => unsafe { ggml::ggml_cpu_has_avx_vnni() },
        CpuFeature::Avx2 => unsafe { ggml::ggml_cpu_has_avx2() },
        CpuFeature::Bmi2 => unsafe { ggml::ggml_cpu_has_bmi2() },
        CpuFeature::Fma => unsafe { ggml::ggml_cpu_has_fma() },
        CpuFeature::F16c => unsafe { ggml::ggml_cpu_has_f16c() },
        CpuFeature::Avx512F => unsafe { ggml::ggml_cpu_has_avx512() },
        CpuFeature::Avx512Vbmi => unsafe { ggml::ggml_cpu_has_avx512_vbmi() },
        CpuFeature::Avx512Vnni => unsafe { ggml::ggml_cpu_has_avx512_vnni() },
        CpuFeature::Avx512Bf16 => unsafe { ggml::ggml_cpu_has_avx512_bf16() },
        CpuFeature::AmxInt8 => unsafe { ggml::ggml_cpu_has_amx_int8() },
        CpuFeature::Avx512Bw | CpuFeature::Avx512Vl => 0,
    };
    flag != 0
}

fn built_with() -> Vec<CpuFeature> {
    CpuFeature::ALL
        .into_iter()
        .filter(|feature| feature.reported_by_build() && built_with_ggml(*feature))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_ggml_cannot_report_are_never_built_in() {
        assert_eq!(
            build_info()
                .cpu
                .into_iter()
                .filter(|feature| !feature.reported_by_build())
                .collect::<Vec<_>>(),
            []
        );
    }

    #[test]
    fn every_backend_has_a_build_status() {
        assert_eq!(build_info().backends.len(), AccelBackend::ALL.len());
        assert_eq!(build_info().status(AccelBackend::Cpu), BuildStatus::Built);
    }
}
