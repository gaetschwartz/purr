use std::path::PathBuf;

use crate::{cuda, error::DiscoveryError, target::LinkTarget};

pub const WINDOWS_VULKAN_LOADER: &str = "vulkan-1.dll";
pub const LINUX_VULKAN_LOADER: &str = "libvulkan.so.1";
/// The NVIDIA driver library, which the CUDA runtime opens itself on first use.
pub const LINUX_CUDA_DRIVER: &str = "libcuda.so.1";
/// transcribe.cpp's own library, which opens its ggml backends from its folder.
pub const WINDOWS_TRANSCRIBE_LIBRARY: &str = "transcribe.dll";

/// Runtime libraries of each GPU backend that a binary must not link eagerly.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeLibraries {
    /// Dependencies come before their dependents, so probing in order names the library at fault.
    pub cuda: Vec<String>,
    pub vulkan: Vec<String>,
    /// The library of the transcribe.cpp engine, which no GPU backend of the other engines needs.
    pub transcribe: Vec<String>,
}

impl RuntimeLibraries {
    pub fn delay_loaded(&self) -> impl Iterator<Item = &str> {
        self.cuda
            .iter()
            .chain(&self.vulkan)
            .chain(&self.transcribe)
            .map(String::as_str)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.delay_loaded().next().is_none()
    }
}

/// What a build script knows about the binary being linked.
#[derive(Debug, Clone, Default)]
pub struct Request {
    pub target: LinkTarget,
    pub cuda: bool,
    pub vulkan: bool,
    pub transcribe: bool,
    pub cuda_path: Option<PathBuf>,
}

fn library_if(enabled: bool, name: &str) -> Vec<String> {
    if enabled {
        vec![name.to_owned()]
    } else {
        Vec::new()
    }
}

impl Request {
    #[must_use]
    pub fn from_cargo_env() -> Self {
        Self {
            target: LinkTarget::from_cargo_env(),
            cuda: std::env::var_os("CARGO_FEATURE_CUDA").is_some(),
            vulkan: std::env::var_os("CARGO_FEATURE_VULKAN").is_some(),
            transcribe: std::env::var_os("CARGO_FEATURE_TRANSCRIBE").is_some(),
            cuda_path: std::env::var_os("CUDA_PATH").map(PathBuf::from),
        }
    }

    pub fn resolve(&self) -> Result<RuntimeLibraries, DiscoveryError> {
        match self.target {
            LinkTarget::WindowsMsvc => self.resolve_windows(),
            LinkTarget::LinuxGnu => Ok(RuntimeLibraries {
                cuda: library_if(self.cuda, LINUX_CUDA_DRIVER),
                vulkan: library_if(self.vulkan, LINUX_VULKAN_LOADER),
                transcribe: Vec::new(),
            }),
            LinkTarget::Other => Ok(RuntimeLibraries::default()),
        }
    }

    fn resolve_windows(&self) -> Result<RuntimeLibraries, DiscoveryError> {
        let cuda = if self.cuda {
            let cuda_path = self
                .cuda_path
                .as_deref()
                .ok_or(DiscoveryError::CudaPathUnset)?;
            let dlls = cuda::locate(cuda_path)?;
            vec![dlls.cublas_lt, dlls.cublas]
        } else {
            Vec::new()
        };
        Ok(RuntimeLibraries {
            cuda,
            vulkan: library_if(self.vulkan, WINDOWS_VULKAN_LOADER),
            transcribe: library_if(self.transcribe, WINDOWS_TRANSCRIBE_LIBRARY),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> Request {
        Request {
            target: LinkTarget::WindowsMsvc,
            ..Request::default()
        }
    }

    #[test]
    fn other_targets_delay_load_nothing() {
        let request = Request {
            target: LinkTarget::Other,
            cuda: true,
            vulkan: true,
            transcribe: true,
            cuda_path: None,
        };
        assert_eq!(request.resolve(), Ok(RuntimeLibraries::default()));
    }

    #[test]
    fn no_gpu_feature_delay_loads_nothing() {
        assert_eq!(request().resolve(), Ok(RuntimeLibraries::default()));
    }

    #[test]
    fn vulkan_delay_loads_the_loader() {
        let request = Request {
            vulkan: true,
            ..request()
        };
        let libraries = request.resolve().unwrap();
        assert_eq!(libraries.vulkan, [WINDOWS_VULKAN_LOADER]);
        assert_eq!(libraries.cuda, Vec::<String>::new());
    }

    #[test]
    fn transcribe_delay_loads_its_library() {
        let request = Request {
            transcribe: true,
            ..request()
        };
        let libraries = request.resolve().unwrap();
        assert_eq!(libraries.transcribe, [WINDOWS_TRANSCRIBE_LIBRARY]);
        assert_eq!(
            libraries.delay_loaded().collect::<Vec<_>>(),
            [WINDOWS_TRANSCRIBE_LIBRARY]
        );
        assert!(!libraries.is_empty());
    }

    #[test]
    fn cuda_without_a_toolkit_path_is_an_error() {
        let request = Request {
            cuda: true,
            ..request()
        };
        assert_eq!(request.resolve(), Err(DiscoveryError::CudaPathUnset));
    }

    #[test]
    fn cuda_delay_loads_both_cublas_dlls() -> std::io::Result<()> {
        let toolkit = tempfile::tempdir()?;
        let bin = toolkit.path().join("bin");
        std::fs::create_dir_all(&bin)?;
        std::fs::write(bin.join("cublas64_13.dll"), b"")?;
        std::fs::write(bin.join("cublasLt64_13.dll"), b"")?;
        let request = Request {
            cuda: true,
            cuda_path: Some(toolkit.path().to_path_buf()),
            ..request()
        };
        let libraries = request.resolve().unwrap();
        assert_eq!(libraries.cuda, ["cublasLt64_13.dll", "cublas64_13.dll"]);
        assert_eq!(
            libraries.delay_loaded().collect::<Vec<_>>(),
            ["cublasLt64_13.dll", "cublas64_13.dll"]
        );
        Ok(())
    }

    #[test]
    fn linux_defers_the_cuda_driver_and_the_vulkan_loader() {
        let request = Request {
            target: LinkTarget::LinuxGnu,
            cuda: true,
            vulkan: true,
            transcribe: true,
            cuda_path: None,
        };
        assert_eq!(
            request.resolve(),
            Ok(RuntimeLibraries {
                cuda: vec![LINUX_CUDA_DRIVER.to_owned()],
                vulkan: vec![LINUX_VULKAN_LOADER.to_owned()],
                transcribe: Vec::new(),
            })
        );
    }

    #[test]
    fn linux_links_transcribe_eagerly() {
        let request = Request {
            target: LinkTarget::LinuxGnu,
            transcribe: true,
            ..Request::default()
        };
        assert_eq!(request.resolve(), Ok(RuntimeLibraries::default()));
    }

    #[test]
    fn linux_without_gpu_features_defers_nothing() {
        let request = Request {
            target: LinkTarget::LinuxGnu,
            ..Request::default()
        };
        assert_eq!(request.resolve(), Ok(RuntimeLibraries::default()));
    }
}
