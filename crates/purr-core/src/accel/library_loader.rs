use std::path::{Path, PathBuf};

/// Opens dynamic libraries and leaves them mapped, so a library that imports one later binds to
/// the module that was opened here rather than searching again.
pub trait Loader {
    fn contains(&self, dir: &Path, library: &str) -> bool;
    /// Open `library` from `dir` by its full path.
    fn open_in(&self, dir: &Path, library: &str) -> bool;
    /// Open `library` through the system's own search order.
    fn open_by_name(&self, library: &str) -> bool;
}

/// Where a backend's libraries were opened from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    Directory(PathBuf),
    System,
}

/// Open `libraries`, dependencies first, from the first of `dirs` that holds all of them, so one
/// backend never mixes versions from two places. Failing that, leave it to the system's search.
///
/// On failure, names the first library that could not be opened.
pub fn open_all<'a>(
    loader: &impl Loader,
    libraries: &[&'a str],
    dirs: &[PathBuf],
) -> Result<Source, &'a str> {
    if let Some(dir) = dirs.iter().find(|dir| {
        !libraries.is_empty()
            && libraries
                .iter()
                .all(|library| loader.contains(dir, library))
            && libraries.iter().all(|library| loader.open_in(dir, library))
    }) {
        return Ok(Source::Directory(dir.clone()));
    }
    libraries
        .iter()
        .copied()
        .find(|library| !loader.open_by_name(library))
        .map_or(Ok(Source::System), Err)
}

/// The operating system's loader.
pub struct SystemLoader;

#[cfg(windows)]
impl Loader for SystemLoader {
    fn contains(&self, dir: &Path, library: &str) -> bool {
        dir.join(library).is_file()
    }

    fn open_in(&self, dir: &Path, library: &str) -> bool {
        use std::os::windows::ffi::OsStrExt as _;
        use windows::{
            Win32::System::LibraryLoader::{
                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LoadLibraryExW,
            },
            core::PCWSTR,
        };

        let wide: Vec<u16> = dir
            .join(library)
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        unsafe {
            LoadLibraryExW(
                PCWSTR(wide.as_ptr()),
                None,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        }
        .is_ok()
    }

    fn open_by_name(&self, library: &str) -> bool {
        use windows::{
            Win32::System::LibraryLoader::{LOAD_LIBRARY_FLAGS, LoadLibraryExW},
            core::PCWSTR,
        };

        let wide: Vec<u16> = library.encode_utf16().chain(std::iter::once(0)).collect();
        unsafe { LoadLibraryExW(PCWSTR(wide.as_ptr()), None, LOAD_LIBRARY_FLAGS(0)) }.is_ok()
    }
}

#[cfg(target_os = "linux")]
impl Loader for SystemLoader {
    fn contains(&self, dir: &Path, library: &str) -> bool {
        dir.join(library).is_file()
    }

    fn open_in(&self, dir: &Path, library: &str) -> bool {
        open_linux(dir.join(library).as_os_str())
    }

    fn open_by_name(&self, library: &str) -> bool {
        open_linux(std::ffi::OsStr::new(library))
    }
}

#[cfg(target_os = "linux")]
fn open_linux(name: &std::ffi::OsStr) -> bool {
    use std::os::unix::ffi::OsStrExt as _;

    let Ok(name) = std::ffi::CString::new(name.as_bytes()) else {
        return false;
    };
    !unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) }.is_null()
}

#[cfg(not(any(windows, target_os = "linux")))]
impl Loader for SystemLoader {
    fn contains(&self, _dir: &Path, _library: &str) -> bool {
        false
    }

    fn open_in(&self, _dir: &Path, _library: &str) -> bool {
        false
    }

    fn open_by_name(&self, _library: &str) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[derive(Default)]
    struct Fake {
        files: HashSet<(PathBuf, String)>,
        unopenable: HashSet<(PathBuf, String)>,
        system: HashSet<String>,
    }

    impl Fake {
        fn with_files(mut self, dir: &str, libraries: &[&str]) -> Self {
            self.files.extend(
                libraries
                    .iter()
                    .map(|library| (PathBuf::from(dir), (*library).to_owned())),
            );
            self
        }

        fn with_system(mut self, libraries: &[&str]) -> Self {
            self.system
                .extend(libraries.iter().map(|library| (*library).to_owned()));
            self
        }

        fn unopenable(mut self, dir: &str, library: &str) -> Self {
            self.unopenable
                .insert((PathBuf::from(dir), library.to_owned()));
            self
        }
    }

    impl Loader for Fake {
        fn contains(&self, dir: &Path, library: &str) -> bool {
            self.files
                .contains(&(dir.to_path_buf(), library.to_owned()))
        }

        fn open_in(&self, dir: &Path, library: &str) -> bool {
            let key = (dir.to_path_buf(), library.to_owned());
            self.files.contains(&key) && !self.unopenable.contains(&key)
        }

        fn open_by_name(&self, library: &str) -> bool {
            self.system.contains(library)
        }
    }

    fn dirs(list: &[&str]) -> Vec<PathBuf> {
        list.iter().map(PathBuf::from).collect()
    }

    #[test]
    fn the_first_directory_holding_every_library_wins() {
        let loader = Fake::default()
            .with_files("/a", &["lt.dll"])
            .with_files("/b", &["lt.dll", "blas.dll"])
            .with_files("/c", &["lt.dll", "blas.dll"]);
        assert_eq!(
            open_all(&loader, &["lt.dll", "blas.dll"], &dirs(&["/a", "/b", "/c"])),
            Ok(Source::Directory(PathBuf::from("/b")))
        );
    }

    #[test]
    fn a_directory_whose_library_fails_to_open_is_passed_over() {
        let loader = Fake::default()
            .with_files("/a", &["lt.dll", "blas.dll"])
            .with_files("/b", &["lt.dll", "blas.dll"])
            .unopenable("/a", "blas.dll");
        assert_eq!(
            open_all(&loader, &["lt.dll", "blas.dll"], &dirs(&["/a", "/b"])),
            Ok(Source::Directory(PathBuf::from("/b")))
        );
    }

    #[test]
    fn the_system_search_is_the_last_resort() {
        let loader = Fake::default()
            .with_files("/a", &["lt.dll"])
            .with_system(&["lt.dll", "blas.dll"]);
        assert_eq!(
            open_all(&loader, &["lt.dll", "blas.dll"], &dirs(&["/a"])),
            Ok(Source::System)
        );
    }

    #[test]
    fn the_first_library_nothing_can_open_is_named() {
        let loader = Fake::default()
            .with_files("/a", &["lt.dll"])
            .with_system(&["lt.dll"]);
        assert_eq!(
            open_all(
                &loader,
                &["lt.dll", "blas.dll", "extra.dll"],
                &dirs(&["/a"])
            ),
            Err("blas.dll")
        );
    }

    #[test]
    fn nothing_to_open_needs_no_directory() {
        assert_eq!(
            open_all(&Fake::default(), &[], &dirs(&["/a"])),
            Ok(Source::System)
        );
    }

    #[cfg(windows)]
    #[test]
    fn a_library_opened_from_a_directory_is_then_found_by_name() -> std::io::Result<()> {
        let dir = tempfile::tempdir()?;
        let system = std::env::var_os("SystemRoot")
            .map_or_else(|| PathBuf::from(r"C:\Windows"), PathBuf::from);
        std::fs::copy(
            system.join("System32").join("wtsapi32.dll"),
            dir.path().join("purr_runtime_probe.dll"),
        )?;
        assert!(!SystemLoader.open_by_name("purr_runtime_probe.dll"));
        assert_eq!(
            open_all(
                &SystemLoader,
                &["purr_runtime_probe.dll"],
                &[dir.path().to_path_buf()]
            ),
            Ok(Source::Directory(dir.path().to_path_buf()))
        );
        assert!(SystemLoader.open_by_name("purr_runtime_probe.dll"));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_c_library_opens_by_name() {
        assert!(SystemLoader.open_by_name("libc.so.6"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_missing_library_does_not_open() {
        assert!(!SystemLoader.open_by_name("libpurr-missing-library.so.1"));
        assert!(!SystemLoader.open_by_name("libc.so.6\0"));
    }
}
