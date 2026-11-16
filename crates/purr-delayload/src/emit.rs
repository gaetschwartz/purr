use std::{path::Path, process::ExitCode};

use crate::{
    error::DiscoveryError,
    libraries::{Request, RuntimeLibraries},
    target::LinkTarget,
};

/// Environment variable carrying the CUDA runtime library names into `purr-core`.
pub const CUDA_ENV: &str = "PURR_CUDA_RUNTIME_LIBS";
/// Environment variable carrying the Vulkan runtime library names into `purr-core`.
pub const VULKAN_ENV: &str = "PURR_VULKAN_RUNTIME_LIBS";
/// Environment variable carrying the transcribe.cpp library names into its engine crate.
pub const TRANSCRIBE_ENV: &str = "PURR_TRANSCRIBE_RUNTIME_LIBS";

/// Environment variable carrying the folder a development build left the transcribe.cpp
/// libraries in, into its engine crate.
pub const TRANSCRIBE_BUILD_DIR_ENV: &str = "PURR_TRANSCRIBE_BUILD_DIR";
/// What cargo hands the engine crate: where the transcribe.cpp build put its runtime libraries.
const TRANSCRIBE_RUNTIME_DIR_VAR: &str = "DEP_TRANSCRIBE_CPP_RUNTIME_DIR";

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

/// Which executables of the package being built get link arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkScope {
    Bins,
    Tests,
}

impl LinkScope {
    const fn directive(self) -> &'static str {
        match self {
            Self::Bins => "cargo::rustc-link-arg-bins",
            Self::Tests => "cargo::rustc-link-arg-tests",
        }
    }
}

fn windows_link_directives(libraries: &RuntimeLibraries, scope: LinkScope) -> Vec<String> {
    if libraries.is_empty() {
        return Vec::new();
    }
    let directive = scope.directive();
    libraries
        .delay_loaded()
        .map(|dll| format!("{directive}=/DELAYLOAD:{dll}"))
        .chain([
            format!("{directive}=delayimp.lib"),
            format!("{directive}={LINKER_IGNORE_UNUSED_DELAYLOAD}"),
        ])
        .collect()
}

/// Keeps the executable's static ggml out of its dynamic symbol table. With it exported, the
/// dynamic linker binds the calls of `libtranscribe` and its own ggml to the executable's copy.
const LINUX_HIDE_ARCHIVE_SYMBOLS: &str = "cargo::rustc-link-arg-bins=-Wl,--exclude-libs,ALL";

fn linux_link_directives(libraries: &RuntimeLibraries, transcribe: bool) -> Vec<String> {
    let hide_ggml = transcribe.then(|| LINUX_HIDE_ARCHIVE_SYMBOLS.to_owned());
    let vulkan_calls: &[&str] = if libraries.vulkan.is_empty() {
        &[]
    } else {
        LINUX_VULKAN_DIRECT_CALLS
    };
    hide_ggml
        .into_iter()
        .chain(vulkan_calls.iter().flat_map(|function| {
            [
                format!("cargo::rustc-link-arg-bins=-Wl,--wrap={function}"),
                // pulls the forwarder out of its archive whatever the link order
                format!("cargo::rustc-link-arg-bins=-Wl,--undefined=__wrap_{function}"),
            ]
        }))
        .collect()
}

#[must_use]
pub fn link_directives(
    target: LinkTarget,
    libraries: &RuntimeLibraries,
    transcribe: bool,
    scope: LinkScope,
) -> Vec<String> {
    match (target, scope) {
        (LinkTarget::WindowsMsvc, scope) => windows_link_directives(libraries, scope),
        (LinkTarget::LinuxGnu, LinkScope::Bins) => linux_link_directives(libraries, transcribe),
        (LinkTarget::LinuxGnu, LinkScope::Tests) | (LinkTarget::Other, _) => Vec::new(),
    }
}

#[must_use]
pub fn env_directives(libraries: &RuntimeLibraries) -> Vec<String> {
    [
        (CUDA_ENV, &libraries.cuda),
        (VULKAN_ENV, &libraries.vulkan),
        (TRANSCRIBE_ENV, &libraries.transcribe),
    ]
    .into_iter()
    .filter(|(_, dlls)| !dlls.is_empty())
    .map(|(name, dlls)| format!("cargo::rustc-env={name}={}", dlls.join(",")))
    .collect()
}

/// The folder of the libraries a build tree holds, handed on only to a build with debug
/// assertions: a shipped build never looks into the tree it was made in.
#[must_use]
pub fn build_dir_directive(runtime_dir: Option<&Path>, debug_assertions: bool) -> Option<String> {
    runtime_dir.filter(|_| debug_assertions).map(|dir| {
        format!(
            "cargo::rustc-env={TRANSCRIBE_BUILD_DIR_ENV}={}",
            dir.display()
        )
    })
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
    print_all(&link_directives(
        request.target,
        &libraries,
        request.transcribe,
        LinkScope::Bins,
    ));
    Ok(())
}

/// Tell the compiled crate which GPU runtime libraries its binaries load on demand.
pub fn emit_runtime_env() -> Result<(), DiscoveryError> {
    let libraries = Request::from_cargo_env().resolve()?;
    print_all(&env_directives(&libraries));
    Ok(())
}

/// Make the test executables of the transcribe.cpp engine crate load its library on demand like
/// the binaries do, and tell the crate which of its libraries that is.
pub fn emit_transcribe_engine() -> Result<(), DiscoveryError> {
    let request = Request {
        cuda: false,
        vulkan: false,
        transcribe: true,
        ..Request::from_cargo_env()
    };
    println!("cargo::rerun-if-env-changed={TRANSCRIBE_RUNTIME_DIR_VAR}");
    let runtime_dir = std::env::var_os(TRANSCRIBE_RUNTIME_DIR_VAR);
    let debug_assertions = std::env::var_os("CARGO_CFG_DEBUG_ASSERTIONS").is_some();
    let libraries = request.resolve()?;
    let directives: Vec<String> = env_directives(&libraries)
        .into_iter()
        .chain(link_directives(
            request.target,
            &libraries,
            request.transcribe,
            LinkScope::Tests,
        ))
        .chain(build_dir_directive(
            runtime_dir.as_deref().map(Path::new),
            debug_assertions,
        ))
        .collect();
    print_all(&directives);
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
            transcribe: vec!["transcribe.dll".to_owned()],
        }
    }

    #[test]
    fn link_directives_delay_load_each_library_then_link_the_helper() {
        assert_eq!(
            link_directives(LinkTarget::WindowsMsvc, &libraries(), true, LinkScope::Bins),
            [
                "cargo::rustc-link-arg-bins=/DELAYLOAD:cublasLt64_13.dll",
                "cargo::rustc-link-arg-bins=/DELAYLOAD:cublas64_13.dll",
                "cargo::rustc-link-arg-bins=/DELAYLOAD:vulkan-1.dll",
                "cargo::rustc-link-arg-bins=/DELAYLOAD:transcribe.dll",
                "cargo::rustc-link-arg-bins=delayimp.lib",
                "cargo::rustc-link-arg-bins=/IGNORE:4199",
            ]
        );
    }

    #[test]
    fn test_executables_get_the_delay_load_arguments_of_their_own_scope() {
        let transcribe_only = RuntimeLibraries {
            transcribe: vec!["transcribe.dll".to_owned()],
            ..RuntimeLibraries::default()
        };
        assert_eq!(
            link_directives(
                LinkTarget::WindowsMsvc,
                &transcribe_only,
                true,
                LinkScope::Tests
            ),
            [
                "cargo::rustc-link-arg-tests=/DELAYLOAD:transcribe.dll",
                "cargo::rustc-link-arg-tests=delayimp.lib",
                "cargo::rustc-link-arg-tests=/IGNORE:4199",
            ]
        );
    }

    #[test]
    fn linux_test_executables_get_no_link_arguments() {
        assert_eq!(
            link_directives(LinkTarget::LinuxGnu, &libraries(), true, LinkScope::Tests),
            Vec::<String>::new()
        );
    }

    #[test]
    fn nothing_to_delay_load_emits_no_link_arguments() {
        for target in [LinkTarget::WindowsMsvc, LinkTarget::LinuxGnu] {
            assert_eq!(
                link_directives(target, &RuntimeLibraries::default(), false, LinkScope::Bins),
                Vec::<String>::new()
            );
        }
    }

    #[test]
    fn linux_wraps_each_direct_vulkan_call() {
        let libraries = RuntimeLibraries {
            cuda: vec!["libcuda.so.1".to_owned()],
            vulkan: vec!["libvulkan.so.1".to_owned()],
            transcribe: Vec::new(),
        };
        assert_eq!(
            link_directives(LinkTarget::LinuxGnu, &libraries, false, LinkScope::Bins),
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
    fn linux_with_transcribe_hides_the_archive_symbols_before_the_vulkan_wraps() {
        let libraries = RuntimeLibraries {
            cuda: Vec::new(),
            vulkan: vec!["libvulkan.so.1".to_owned()],
            transcribe: Vec::new(),
        };
        let directives = link_directives(LinkTarget::LinuxGnu, &libraries, true, LinkScope::Bins);
        assert_eq!(
            directives.first().map(String::as_str),
            Some("cargo::rustc-link-arg-bins=-Wl,--exclude-libs,ALL")
        );
        assert_eq!(directives.len(), 1 + 2 * LINUX_VULKAN_DIRECT_CALLS.len());
    }

    #[test]
    fn linux_with_transcribe_alone_only_hides_the_archive_symbols() {
        assert_eq!(
            link_directives(
                LinkTarget::LinuxGnu,
                &RuntimeLibraries::default(),
                true,
                LinkScope::Bins
            ),
            ["cargo::rustc-link-arg-bins=-Wl,--exclude-libs,ALL"]
        );
    }

    #[test]
    fn linux_cuda_alone_needs_no_link_arguments() {
        let libraries = RuntimeLibraries {
            cuda: vec!["libcuda.so.1".to_owned()],
            vulkan: Vec::new(),
            transcribe: Vec::new(),
        };
        assert_eq!(
            link_directives(LinkTarget::LinuxGnu, &libraries, false, LinkScope::Bins),
            Vec::<String>::new()
        );
    }

    #[test]
    fn other_targets_get_no_link_arguments() {
        assert_eq!(
            link_directives(LinkTarget::Other, &libraries(), true, LinkScope::Bins),
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
                "cargo::rustc-env=PURR_TRANSCRIBE_RUNTIME_LIBS=transcribe.dll",
            ]
        );
    }

    #[test]
    fn a_debug_build_hands_on_the_folder_of_its_libraries() {
        assert_eq!(
            build_dir_directive(Some(Path::new("/target/out/bin")), true).as_deref(),
            Some("cargo::rustc-env=PURR_TRANSCRIBE_BUILD_DIR=/target/out/bin")
        );
    }

    #[test]
    fn a_build_without_debug_assertions_hands_on_nothing() {
        assert_eq!(
            build_dir_directive(Some(Path::new("/target/out/bin")), false),
            None
        );
    }

    #[test]
    fn no_folder_means_no_directive() {
        assert_eq!(build_dir_directive(None, true), None);
    }

    #[test]
    fn env_directives_skip_backends_without_libraries() {
        let vulkan_only = RuntimeLibraries {
            cuda: Vec::new(),
            transcribe: Vec::new(),
            ..libraries()
        };
        assert_eq!(
            env_directives(&vulkan_only),
            ["cargo::rustc-env=PURR_VULKAN_RUNTIME_LIBS=vulkan-1.dll"]
        );
    }
}
