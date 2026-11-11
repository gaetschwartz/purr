use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DiscoveryError {
    #[error("CUDA_PATH is not set, so the cuBLAS DLL names cannot be determined")]
    CudaPathUnset,
    #[error(
        "no `{pattern}` DLL found in {}",
        searched.iter().map(|dir| dir.display().to_string()).collect::<Vec<_>>().join(", ")
    )]
    NotFound {
        pattern: &'static str,
        searched: Vec<PathBuf>,
    },
    #[error("several DLLs match `{pattern}` ({}); keep one CUDA toolkit", found.join(", "))]
    Ambiguous {
        pattern: &'static str,
        found: Vec<String>,
    },
}
