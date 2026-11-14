//! Loading transcribe.cpp on demand.
//!
//! On Windows the binaries delay-load `transcribe.dll`, so nothing of transcribe.cpp may be
//! called before [`get`] has opened the library: the delay-load helper aborts the process when
//! the library is absent. [`get`] opens it from the runtime folders, never through the system's
//! search, because `ggml.dll` is a name other software puts on `PATH`.

use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

use purr_core::accel::{
    library_loader::{Loader, SystemLoader},
    runtime_dir::{Environment, search_dirs},
};

use crate::logging;

const LIBRARIES: Option<&str> = option_env!("PURR_TRANSCRIBE_RUNTIME_LIBS");

/// transcribe.cpp, loaded and with its backend modules registered.
#[derive(Debug)]
pub struct Runtime {
    /// The version the library reports.
    pub version: String,
    /// The folder the library was opened from; `None` when the binary links it eagerly.
    pub directory: Option<PathBuf>,
}

/// Why transcribe.cpp cannot be used in this process.
#[derive(Debug, thiserror::Error)]
pub enum Unavailable {
    #[error("the library {library} was not found in any runtime folder")]
    LibraryMissing {
        library: String,
        searched: Vec<PathBuf>,
    },
    #[error("the library {library} in {} cannot be loaded, a library it needs may be missing", directory.display())]
    LibraryUnloadable { library: String, directory: PathBuf },
    #[error("transcribe.cpp registered no compute backend")]
    Backends(#[source] transcribe_cpp::Error),
}

static RUNTIME: OnceLock<Result<Runtime, Unavailable>> = OnceLock::new();

/// Loads transcribe.cpp the first time, and answers with the outcome ever after.
pub fn get() -> Result<&'static Runtime, &'static Unavailable> {
    let outcome = RUNTIME.get_or_init(load).as_ref();
    if outcome.is_ok() {
        logging::install_if_requested();
    }
    outcome
}

/// The outcome if the library has been loaded or refused already, without loading it.
pub(crate) fn loaded() -> Option<Result<&'static Runtime, &'static Unavailable>> {
    RUNTIME.get().map(Result::as_ref)
}

fn libraries() -> impl Iterator<Item = &'static str> {
    LIBRARIES
        .into_iter()
        .flat_map(|list| list.split(','))
        .filter(|name| !name.is_empty())
}

fn load() -> Result<Runtime, Unavailable> {
    let searched: Vec<PathBuf> = search_dirs(&Environment::current())
        .into_iter()
        .map(|dir| dir.path)
        .collect();
    let mut directory = None;
    for library in libraries() {
        directory = Some(match open_in_first(&SystemLoader, library, &searched) {
            Opening::Opened(dir) => dir.to_path_buf(),
            Opening::Unopenable(dir) => {
                return Err(Unavailable::LibraryUnloadable {
                    library: library.to_owned(),
                    directory: dir.to_path_buf(),
                });
            }
            Opening::Absent => {
                return Err(Unavailable::LibraryMissing {
                    library: library.to_owned(),
                    searched,
                });
            }
        });
    }
    logging::install_if_requested();
    transcribe_cpp::init_backends_default().map_err(Unavailable::Backends)?;
    Ok(Runtime {
        version: transcribe_cpp::version(),
        directory,
    })
}

enum Opening<'a> {
    Opened(&'a Path),
    /// The first folder that holds the library although the loader could not open it
    Unopenable(&'a Path),
    Absent,
}

/// Opens `library` from the first of `dirs` that holds it and lets the loader open it. The library
/// stays mapped, so the delay-load helper binds to it by name afterwards.
fn open_in_first<'a>(loader: &impl Loader, library: &str, dirs: &'a [PathBuf]) -> Opening<'a> {
    let mut unopenable = None;
    for dir in dirs.iter().filter(|dir| loader.contains(dir, library)) {
        if loader.open_in(dir, library) {
            return Opening::Opened(dir);
        }
        unopenable.get_or_insert(dir.as_path());
    }
    unopenable.map_or(Opening::Absent, Opening::Unopenable)
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

    impl Loader for Fake {
        fn contains(&self, dir: &Path, library: &str) -> bool {
            self.files
                .contains(&(dir.to_path_buf(), library.to_owned()))
        }

        fn open_in(&self, dir: &Path, library: &str) -> bool {
            let key = (dir.to_path_buf(), library.to_owned());
            !self.unopenable.contains(&key)
        }

        fn open_by_name(&self, library: &str) -> bool {
            self.system.contains(library)
        }
    }

    fn dirs(list: &[&str]) -> Vec<PathBuf> {
        list.iter().map(PathBuf::from).collect()
    }

    fn fake(files: &[(&str, &str)]) -> Fake {
        Fake {
            files: files
                .iter()
                .map(|(dir, library)| (PathBuf::from(dir), (*library).to_owned()))
                .collect(),
            ..Fake::default()
        }
    }

    #[test]
    fn the_first_folder_holding_the_library_wins() {
        let loader = fake(&[("/b", "transcribe.dll"), ("/c", "transcribe.dll")]);
        assert!(matches!(
            open_in_first(&loader, "transcribe.dll", &dirs(&["/a", "/b", "/c"])),
            Opening::Opened(dir) if dir == Path::new("/b")
        ));
    }

    #[test]
    fn a_folder_whose_copy_does_not_open_is_passed_over() {
        let mut loader = fake(&[("/a", "transcribe.dll"), ("/b", "transcribe.dll")]);
        loader
            .unopenable
            .insert((PathBuf::from("/a"), "transcribe.dll".to_owned()));
        assert!(matches!(
            open_in_first(&loader, "transcribe.dll", &dirs(&["/a", "/b"])),
            Opening::Opened(dir) if dir == Path::new("/b")
        ));
    }

    #[test]
    fn a_copy_nothing_opens_is_told_apart_from_a_missing_one() {
        let mut loader = fake(&[("/a", "transcribe.dll")]);
        loader
            .unopenable
            .insert((PathBuf::from("/a"), "transcribe.dll".to_owned()));
        assert!(matches!(
            open_in_first(&loader, "transcribe.dll", &dirs(&["/a", "/b"])),
            Opening::Unopenable(dir) if dir == Path::new("/a")
        ));
        assert!(matches!(
            open_in_first(&loader, "transcribe.dll", &dirs(&["/b"])),
            Opening::Absent
        ));
    }

    #[test]
    fn the_system_search_is_never_used() {
        let mut loader = fake(&[]);
        loader.system.insert("transcribe.dll".to_owned());
        assert!(matches!(
            open_in_first(&loader, "transcribe.dll", &dirs(&["/a"])),
            Opening::Absent
        ));
    }
}
