use std::collections::HashMap;

use super::{backend::AccelBackend, library_loader::is_loadable};

fn parse_list(list: Option<&'static str>) -> Vec<&'static str> {
    list.into_iter()
        .flat_map(|list| list.split(','))
        .filter(|name| !name.is_empty())
        .collect()
}

/// Dynamic libraries `backend` needs at run time that the binary does not link eagerly.
///
/// Fixed when the binary is built, in the order they should be probed.
#[must_use]
pub fn required(backend: AccelBackend) -> Vec<&'static str> {
    parse_list(match backend {
        AccelBackend::Cuda => option_env!("PURR_CUDA_RUNTIME_LIBS"),
        AccelBackend::Vulkan => option_env!("PURR_VULKAN_RUNTIME_LIBS"),
        AccelBackend::Hip
        | AccelBackend::CoreMl
        | AccelBackend::Metal
        | AccelBackend::Sycl
        | AccelBackend::Cpu => None,
    })
}

/// First of `libraries` that `loadable` rejects.
pub fn first_missing<'a>(
    libraries: &[&'a str],
    loadable: impl Fn(&str) -> bool,
) -> Option<&'a str> {
    libraries.iter().copied().find(|library| !loadable(library))
}

/// For each backend whose runtime libraries cannot be loaded, the first one that cannot.
#[must_use]
pub fn detect_missing() -> HashMap<AccelBackend, String> {
    AccelBackend::ALL
        .iter()
        .filter_map(|backend| {
            first_missing(&required(*backend), is_loadable)
                .map(|library| (*backend, library.to_owned()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_list_splits_on_commas() {
        assert_eq!(
            parse_list(Some("cublasLt64_13.dll,cublas64_13.dll")),
            ["cublasLt64_13.dll", "cublas64_13.dll"]
        );
    }

    #[test]
    fn parse_list_of_nothing_is_empty() {
        assert_eq!(parse_list(None), Vec::<&str>::new());
        assert_eq!(parse_list(Some("")), Vec::<&str>::new());
    }

    #[test]
    fn first_missing_reports_the_first_unloadable_library() {
        let libraries = ["a.dll", "b.dll", "c.dll"];
        assert_eq!(
            first_missing(&libraries, |name| name == "a.dll"),
            Some("b.dll")
        );
    }

    #[test]
    fn first_missing_is_none_when_everything_loads() {
        assert_eq!(first_missing(&["a.dll", "b.dll"], |_| true), None);
        assert_eq!(first_missing(&[], |_| false), None);
    }

    #[test]
    fn cpu_requires_no_runtime_library() {
        assert_eq!(required(AccelBackend::Cpu), Vec::<&str>::new());
    }
}
