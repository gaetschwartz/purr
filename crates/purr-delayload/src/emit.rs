use std::process::ExitCode;

use crate::{
    error::DiscoveryError,
    libraries::{Request, RuntimeLibraries},
    target::LinkTarget,
};

/// Environment variable carrying the CUDA runtime library names into `purr-core`.
pub const CUDA_ENV: &str = "PURR_CUDA_RUNTIME_LIBS";
/// Environment variable carrying the Vulkan runtime library names into `purr-core`.
pub const VULKAN_ENV: &str = "PURR_VULKAN_RUNTIME_LIBS";

const LINKER_IGNORE_UNUSED_DELAYLOAD: &str = "/IGNORE:4199";

/// Vulkan functions ggml calls directly instead of through its dispatcher.
///
/// On Linux each call is redirected to a `__wrap_` forwarder that `purr-core` defines, so the
/// binary does not need the Vulkan loader to start.
pub const LINUX_VULKAN_DIRECT_CALLS: &[&str] = &[
    "vkGetInstanceProcAddr",
    "vkGetPhysicalDeviceFeatures2",
    "vkCmdCopyBuffer",
];

fn windows_link_directives(libraries: &RuntimeLibraries) -> Vec<String> {
    if libraries.is_empty() {
        return Vec::new();
    }
    libraries
        .delay_loaded()
        .map(|dll| format!("cargo::rustc-link-arg-bins=/DELAYLOAD:{dll}"))
        .chain([
            "cargo::rustc-link-arg-bins=delayimp.lib".to_owned(),
            format!("cargo::rustc-link-arg-bins={LINKER_IGNORE_UNUSED_DELAYLOAD}"),
        ])
        .collect()
}

fn linux_link_directives(libraries: &RuntimeLibraries) -> Vec<String> {
    if libraries.vulkan.is_empty() {
        return Vec::new();
    }
    LINUX_VULKAN_DIRECT_CALLS
        .iter()
        .flat_map(|function| {
            [
                format!("cargo::rustc-link-arg-bins=-Wl,--wrap={function}"),
                // pulls the forwarder out of its archive whatever the link order
                format!("cargo::rustc-link-arg-bins=-Wl,--undefined=__wrap_{function}"),
            ]
        })
        .collect()
}

#[must_use]
pub fn link_directives(target: LinkTarget, libraries: &RuntimeLibraries) -> Vec<String> {
    match target {
        LinkTarget::WindowsMsvc => windows_link_directives(libraries),
        LinkTarget::LinuxGnu => linux_link_directives(libraries),
        LinkTarget::Other => Vec::new(),
    }
}

#[must_use]
pub fn env_directives(libraries: &RuntimeLibraries) -> Vec<String> {
    [(CUDA_ENV, &libraries.cuda), (VULKAN_ENV, &libraries.vulkan)]
        .into_iter()
        .filter(|(_, dlls)| !dlls.is_empty())
        .map(|(name, dlls)| format!("cargo::rustc-env={name}={}", dlls.join(",")))
        .collect()
}

fn print_all(directives: &[String]) {
    println!("cargo::rerun-if-env-changed=CUDA_PATH");
    for directive in directives {
        println!("{directive}");
    }
}

/// Ask the linker to defer loading the GPU runtime libraries of the binaries of this package.
pub fn emit_link_args() -> Result<(), DiscoveryError> {
    let request = Request::from_cargo_env();
    let libraries = request.resolve()?;
    print_all(&link_directives(request.target, &libraries));
    Ok(())
}

/// Tell the compiled crate which GPU runtime libraries its binaries load on demand.
pub fn emit_runtime_env() -> Result<(), DiscoveryError> {
    let libraries = Request::from_cargo_env().resolve()?;
    print_all(&env_directives(&libraries));
    Ok(())
}

/// Turn the outcome of an emitter into the exit code of a build script.
#[must_use]
pub fn report(outcome: Result<(), DiscoveryError>) -> ExitCode {
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            println!("cargo::error={error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn libraries() -> RuntimeLibraries {
        RuntimeLibraries {
            cuda: vec!["cublasLt64_13.dll".to_owned(), "cublas64_13.dll".to_owned()],
            vulkan: vec!["vulkan-1.dll".to_owned()],
        }
    }

    #[test]
    fn link_directives_delay_load_each_library_then_link_the_helper() {
        assert_eq!(
            link_directives(LinkTarget::WindowsMsvc, &libraries()),
            [
                "cargo::rustc-link-arg-bins=/DELAYLOAD:cublasLt64_13.dll",
                "cargo::rustc-link-arg-bins=/DELAYLOAD:cublas64_13.dll",
                "cargo::rustc-link-arg-bins=/DELAYLOAD:vulkan-1.dll",
                "cargo::rustc-link-arg-bins=delayimp.lib",
                "cargo::rustc-link-arg-bins=/IGNORE:4199",
            ]
        );
    }

    #[test]
    fn nothing_to_delay_load_emits_no_link_arguments() {
        for target in [LinkTarget::WindowsMsvc, LinkTarget::LinuxGnu] {
            assert_eq!(
                link_directives(target, &RuntimeLibraries::default()),
                Vec::<String>::new()
            );
        }
    }

    #[test]
    fn linux_wraps_each_direct_vulkan_call() {
        let libraries = RuntimeLibraries {
            cuda: vec!["libcuda.so.1".to_owned()],
            vulkan: vec!["libvulkan.so.1".to_owned()],
        };
        assert_eq!(
            link_directives(LinkTarget::LinuxGnu, &libraries),
            [
                "cargo::rustc-link-arg-bins=-Wl,--wrap=vkGetInstanceProcAddr",
                "cargo::rustc-link-arg-bins=-Wl,--undefined=__wrap_vkGetInstanceProcAddr",
                "cargo::rustc-link-arg-bins=-Wl,--wrap=vkGetPhysicalDeviceFeatures2",
                "cargo::rustc-link-arg-bins=-Wl,--undefined=__wrap_vkGetPhysicalDeviceFeatures2",
                "cargo::rustc-link-arg-bins=-Wl,--wrap=vkCmdCopyBuffer",
                "cargo::rustc-link-arg-bins=-Wl,--undefined=__wrap_vkCmdCopyBuffer",
            ]
        );
    }

    #[test]
    fn linux_cuda_alone_needs_no_link_arguments() {
        let libraries = RuntimeLibraries {
            cuda: vec!["libcuda.so.1".to_owned()],
            vulkan: Vec::new(),
        };
        assert_eq!(
            link_directives(LinkTarget::LinuxGnu, &libraries),
            Vec::<String>::new()
        );
    }

    #[test]
    fn other_targets_get_no_link_arguments() {
        assert_eq!(
            link_directives(LinkTarget::Other, &libraries()),
            Vec::<String>::new()
        );
    }

    #[test]
    fn env_directives_list_each_backend_separately() {
        assert_eq!(
            env_directives(&libraries()),
            [
                "cargo::rustc-env=PURR_CUDA_RUNTIME_LIBS=cublasLt64_13.dll,cublas64_13.dll",
                "cargo::rustc-env=PURR_VULKAN_RUNTIME_LIBS=vulkan-1.dll",
            ]
        );
    }

    #[test]
    fn env_directives_skip_backends_without_libraries() {
        let vulkan_only = RuntimeLibraries {
            cuda: Vec::new(),
            ..libraries()
        };
        assert_eq!(
            env_directives(&vulkan_only),
            ["cargo::rustc-env=PURR_VULKAN_RUNTIME_LIBS=vulkan-1.dll"]
        );
    }
}
