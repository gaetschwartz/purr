use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use super::{
    backend::AccelBackend,
    library_loader::{Loader, SystemLoader, open_all},
    runtime_dir::SearchDir,
};

fn parse_list(list: Option<&'static str>) -> Vec<&'static str> {
    list.into_iter()
        .flat_map(|list| list.split(','))
        .filter(|name| !name.is_empty())
        .collect()
}

/// Dynamic libraries `backend` needs at run time that the binary does not link eagerly.
///
/// Fixed when the binary is built, in the order they should be opened: dependencies first.
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

/// For each backend whose runtime libraries cannot be opened, the first one that cannot.
///
/// The libraries that can be opened stay mapped, so the delay-loaded imports that follow bind to
/// them wherever they were found.
#[must_use]
pub fn detect_missing(dirs: &[SearchDir]) -> HashMap<AccelBackend, String> {
    let paths: Vec<PathBuf> = dirs.iter().map(|dir| dir.path.clone()).collect();
    detect_missing_with(&SystemLoader, &paths)
}

fn detect_missing_with(loader: &impl Loader, dirs: &[PathBuf]) -> HashMap<AccelBackend, String> {
    AccelBackend::ALL
        .iter()
        .filter_map(|backend| {
            open_all(loader, &required(*backend), dirs)
                .err()
                .map(|library| (*backend, library.to_owned()))
        })
        .collect()
}

/// A directory of the search order and which runtime libraries it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirContents {
    pub dir: SearchDir,
    pub exists: bool,
    pub present: Vec<&'static str>,
}

/// What each directory of the search order holds of `libraries`.
pub fn inspect(
    dirs: &[SearchDir],
    libraries: &[&'static str],
    is_dir: impl Fn(&Path) -> bool,
    is_file: impl Fn(&Path) -> bool,
) -> Vec<DirContents> {
    dirs.iter()
        .map(|dir| DirContents {
            exists: is_dir(&dir.path),
            present: libraries
                .iter()
                .copied()
                .filter(|library| is_file(&dir.path.join(library)))
                .collect(),
            dir: dir.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accel::runtime_dir::Origin;

    struct Nowhere;

    impl Loader for Nowhere {
        fn contains(&self, _dir: &Path, _library: &str) -> bool {
            false
        }

        fn open_in(&self, _dir: &Path, _library: &str) -> bool {
            false
        }

        fn open_by_name(&self, _library: &str) -> bool {
            false
        }
    }

    fn dir(path: &str) -> SearchDir {
        SearchDir {
            path: PathBuf::from(path),
            origin: Origin::UserData,
        }
    }

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
    fn cpu_requires_no_runtime_library() {
        assert_eq!(required(AccelBackend::Cpu), Vec::<&str>::new());
    }

    #[test]
    fn backends_without_runtime_libraries_miss_nothing() {
        let missing = detect_missing_with(&Nowhere, &[PathBuf::from("/runtime")]);
        assert_eq!(missing.get(&AccelBackend::Cpu), None);
        assert_eq!(missing.get(&AccelBackend::Metal), None);
    }

    #[test]
    fn inspect_lists_what_each_directory_holds() {
        let is_dir = |path: &Path| path == Path::new("/a");
        let is_file = |path: &Path| path == Path::new("/a").join("lt.dll");
        assert_eq!(
            inspect(
                &[dir("/a"), dir("/b")],
                &["lt.dll", "blas.dll"],
                is_dir,
                is_file
            ),
            [
                DirContents {
                    dir: dir("/a"),
                    exists: true,
                    present: vec!["lt.dll"],
                },
                DirContents {
                    dir: dir("/b"),
                    exists: false,
                    present: Vec::new(),
                },
            ]
        );
    }
}
