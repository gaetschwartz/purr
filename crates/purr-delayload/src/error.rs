use std::path::PathBuf;

#[oopsie::oopsie]
#[derive(Clone, PartialEq, Eq)]
pub enum DiscoveryError {
    #[oopsie("CUDA_PATH is not set, so the cuBLAS DLL names cannot be determined")]
    CudaPathUnset,
    #[oopsie(
        "no `{pattern}` DLL found in {}",
        searched.iter().map(|dir| dir.display().to_string()).collect::<Vec<_>>().join(", ")
    )]
    NotFound {
        pattern: &'static str,
        searched: Vec<PathBuf>,
    },
    #[oopsie("several DLLs match `{pattern}` ({}); keep one CUDA toolkit", found.join(", "))]
    Ambiguous {
        pattern: &'static str,
        found: Vec<String>,
    },
}
