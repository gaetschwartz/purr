use std::path::{Path, PathBuf};

use crate::error::DiscoveryError;

const CUBLAS_PATTERN: &str = "cublas64_*.dll";
const CUBLAS_LT_PATTERN: &str = "cublasLt64_*.dll";
const CUBLAS_PREFIX: &str = "cublas64_";
const CUBLAS_LT_PREFIX: &str = "cublasLt64_";

/// Versioned cuBLAS DLL names of a CUDA toolkit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CublasDlls {
    pub cublas: String,
    pub cublas_lt: String,
}

fn matches_pattern(file_name: &str, prefix: &str) -> bool {
    let is_dll = Path::new(file_name)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("dll"));
    is_dll
        && file_name
            .to_ascii_lowercase()
            .starts_with(&prefix.to_ascii_lowercase())
}

fn pick_one(
    file_names: &[String],
    pattern: &'static str,
    prefix: &str,
) -> Result<String, DiscoveryError> {
    let mut found: Vec<String> = file_names
        .iter()
        .filter(|name| matches_pattern(name, prefix))
        .cloned()
        .collect();
    found.sort();
    match found.as_slice() {
        [] => Err(DiscoveryError::NotFound {
            pattern,
            searched: Vec::new(),
        }),
        [only] => Ok(only.clone()),
        [..] => Err(DiscoveryError::Ambiguous { pattern, found }),
    }
}

/// Pick the cuBLAS and cuBLASLt DLLs out of a directory listing.
pub fn pick_cublas(file_names: &[String]) -> Result<CublasDlls, DiscoveryError> {
    Ok(CublasDlls {
        cublas: pick_one(file_names, CUBLAS_PATTERN, CUBLAS_PREFIX)?,
        cublas_lt: pick_one(file_names, CUBLAS_LT_PATTERN, CUBLAS_LT_PREFIX)?,
    })
}

/// Directories of a CUDA toolkit that can hold its runtime DLLs, most specific first.
pub fn dll_directories(cuda_path: &Path) -> [PathBuf; 2] {
    [cuda_path.join("bin").join("x64"), cuda_path.join("bin")]
}

fn list_file_names(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect()
}

/// Find the cuBLAS DLLs shipped by the CUDA toolkit installed at `cuda_path`.
pub fn locate(cuda_path: &Path) -> Result<CublasDlls, DiscoveryError> {
    let directories = dll_directories(cuda_path);
    let mut missing = CUBLAS_PATTERN;
    for dir in &directories {
        match pick_cublas(&list_file_names(dir)) {
            Ok(dlls) => return Ok(dlls),
            Err(DiscoveryError::NotFound { pattern, .. }) => missing = pattern,
            Err(error @ (DiscoveryError::Ambiguous { .. } | DiscoveryError::CudaPathUnset)) => {
                return Err(error);
            }
        }
    }
    Err(DiscoveryError::NotFound {
        pattern: missing,
        searched: directories.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|name| (*name).to_owned()).collect()
    }

    fn dlls(cublas: &str, cublas_lt: &str) -> CublasDlls {
        CublasDlls {
            cublas: cublas.to_owned(),
            cublas_lt: cublas_lt.to_owned(),
        }
    }

    fn not_found(pattern: &'static str) -> DiscoveryError {
        DiscoveryError::NotFound {
            pattern,
            searched: Vec::new(),
        }
    }

    #[test]
    fn picks_versioned_cublas_dlls() {
        let listing = names(&[
            "cudart64_13.dll",
            "cublas64_13.dll",
            "cublasLt64_13.dll",
            "cufft64_12.dll",
        ]);
        assert_eq!(
            pick_cublas(&listing),
            Ok(dlls("cublas64_13.dll", "cublasLt64_13.dll"))
        );
    }

    #[test]
    fn matching_ignores_case() {
        let listing = names(&["CUBLAS64_12.DLL", "CUBLASLT64_12.DLL"]);
        assert_eq!(
            pick_cublas(&listing),
            Ok(dlls("CUBLAS64_12.DLL", "CUBLASLT64_12.DLL"))
        );
    }

    #[test]
    fn cublas_lt_does_not_satisfy_the_cublas_pattern() {
        let listing = names(&["cublasLt64_13.dll"]);
        assert_eq!(pick_cublas(&listing), Err(not_found(CUBLAS_PATTERN)));
    }

    #[test]
    fn missing_cublas_lt_is_reported_by_its_pattern() {
        let listing = names(&["cublas64_13.dll"]);
        assert_eq!(pick_cublas(&listing), Err(not_found(CUBLAS_LT_PATTERN)));
    }

    #[test]
    fn non_dll_files_are_ignored() {
        let listing = names(&[
            "cublas64_13.lib",
            "cublas64_13.dll.bak",
            "cublasLt64_13.dll",
        ]);
        assert_eq!(pick_cublas(&listing), Err(not_found(CUBLAS_PATTERN)));
    }

    #[test]
    fn several_versions_are_ambiguous() {
        let listing = names(&["cublas64_12.dll", "cublas64_13.dll", "cublasLt64_13.dll"]);
        assert_eq!(
            pick_cublas(&listing),
            Err(DiscoveryError::Ambiguous {
                pattern: CUBLAS_PATTERN,
                found: names(&["cublas64_12.dll", "cublas64_13.dll"]),
            })
        );
    }

    #[test]
    fn locate_prefers_the_x64_directory() -> std::io::Result<()> {
        let toolkit = tempfile::tempdir()?;
        let x64 = toolkit.path().join("bin").join("x64");
        std::fs::create_dir_all(&x64)?;
        std::fs::write(x64.join("cublas64_13.dll"), b"")?;
        std::fs::write(x64.join("cublasLt64_13.dll"), b"")?;
        std::fs::write(toolkit.path().join("bin").join("cublas64_12.dll"), b"")?;
        assert_eq!(
            locate(toolkit.path()),
            Ok(dlls("cublas64_13.dll", "cublasLt64_13.dll"))
        );
        Ok(())
    }

    #[test]
    fn locate_falls_back_to_the_bin_directory() -> std::io::Result<()> {
        let toolkit = tempfile::tempdir()?;
        let bin = toolkit.path().join("bin");
        std::fs::create_dir_all(&bin)?;
        std::fs::write(bin.join("cublas64_12.dll"), b"")?;
        std::fs::write(bin.join("cublasLt64_12.dll"), b"")?;
        assert_eq!(
            locate(toolkit.path()),
            Ok(dlls("cublas64_12.dll", "cublasLt64_12.dll"))
        );
        Ok(())
    }

    #[test]
    fn locate_reports_the_searched_directories() -> std::io::Result<()> {
        let toolkit = tempfile::tempdir()?;
        assert_eq!(
            locate(toolkit.path()),
            Err(DiscoveryError::NotFound {
                pattern: CUBLAS_PATTERN,
                searched: dll_directories(toolkit.path()).to_vec(),
            })
        );
        Ok(())
    }
}
