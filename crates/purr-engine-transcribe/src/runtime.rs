//! Loading transcribe.cpp on demand.
//!
//! On Windows the binaries delay-load `transcribe.dll`, so nothing of transcribe.cpp may be
//! called before [`get`] has opened the library: the delay-load helper aborts the process when
//! the library is absent. [`get`] opens it from the runtime folders, never through the system's
//! search, because `ggml.dll` is a name other software puts on `PATH`.

use std::{
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use parking_lot::Mutex;
use purr_core::accel::{
    library_loader::{Loader, SystemLoader},
    runtime_dir::{Environment, search_dirs},
    runtime_library::detect_missing,
};
use tracing::debug;

use crate::logging;

const LIBRARIES: Option<&str> = option_env!("PURR_TRANSCRIBE_RUNTIME_LIBS");
const BUILD_DIR: Option<&str> = option_env!("PURR_TRANSCRIBE_BUILD_DIR");

/// transcribe.cpp, loaded and with its backend modules registered.
#[derive(Debug)]
pub struct Runtime {
    /// The version the library reports.
    pub version: String,
    /// The folder the library was opened from; `None` when the binary links it eagerly.
    pub directory: Option<PathBuf>,
}

/// Why transcribe.cpp cannot be used in this process.
#[oopsie::oopsie]
#[derive(Clone)]
pub enum Unavailable {
    #[oopsie("the library {library} was not found in any runtime folder")]
    LibraryMissing {
        library: String,
        searched: Vec<PathBuf>,
    },
    #[oopsie("the library {library} in {} cannot be loaded, a library it needs may be missing", directory.display())]
    LibraryUnloadable { library: String, directory: PathBuf },
    #[oopsie("transcribe.cpp registered no compute backend")]
    Backends { source: Arc<transcribe_cpp::Error> },
}

impl Unavailable {
    /// Whether the outcome stays the same for the life of the process. A library that is missing
    /// or cannot be opened may be fixed by installing it; transcribe.cpp scans a folder for
    /// backend modules once and answers the same ever after.
    const fn is_final(&self) -> bool {
        match self {
            Self::LibraryMissing { .. } | Self::LibraryUnloadable { .. } => false,
            Self::Backends { .. } => true,
        }
    }
}

/// The outcome of loading, kept once it cannot change.
struct Slot {
    settled: OnceLock<Result<Runtime, Unavailable>>,
    gate: Mutex<()>,
}

impl Slot {
    const fn new() -> Self {
        Self {
            settled: OnceLock::new(),
            gate: Mutex::new(()),
        }
    }

    /// The kept outcome, or the result of `load`, which runs one call at a time and again on the
    /// next call after a failure that installing something can fix.
    fn get_or_load(
        &self,
        load: impl FnOnce() -> Result<Runtime, Unavailable>,
    ) -> Result<&Runtime, Unavailable> {
        if let Some(settled) = self.settled.get() {
            return settled.as_ref().map_err(Clone::clone);
        }
        let _one_at_a_time = self.gate.lock();
        if let Some(settled) = self.settled.get() {
            return settled.as_ref().map_err(Clone::clone);
        }
        match load() {
            Err(why) if !why.is_final() => Err(why),
            outcome => self
                .settled
                .get_or_init(|| outcome)
                .as_ref()
                .map_err(Clone::clone),
        }
    }

    fn loaded(&self) -> Option<&Runtime> {
        self.settled.get().and_then(|outcome| outcome.as_ref().ok())
    }
}

static RUNTIME: Slot = Slot::new();

/// Loads transcribe.cpp the first time it succeeds, and answers with that ever after. A library
/// that was not there is looked for again on the next call.
///
/// The GPU runtime libraries are opened from the runtime folders first: transcribe.cpp scans its
/// backend modules once, and a module whose imports cannot be loaded then is left out for good.
pub fn get() -> Result<&'static Runtime, Unavailable> {
    let runtime = RUNTIME.get_or_load(load)?;
    logging::install_if_requested();
    Ok(runtime)
}

/// The runtime if the library has been loaded already, without loading it.
pub(crate) fn loaded() -> Option<&'static Runtime> {
    RUNTIME.loaded()
}

fn libraries() -> impl Iterator<Item = &'static str> {
    LIBRARIES
        .into_iter()
        .flat_map(|list| list.split(','))
        .filter(|name| !name.is_empty())
}

/// Where the build left its libraries; only a build with debug assertions has any.
fn build_output_dirs() -> Vec<PathBuf> {
    BUILD_DIR
        .filter(|_| cfg!(debug_assertions))
        .map(PathBuf::from)
        .into_iter()
        .collect()
}

fn load() -> Result<Runtime, Unavailable> {
    let environment = Environment {
        build_output_dirs: build_output_dirs(),
        ..Environment::current()
    };
    let dirs = search_dirs(&environment);
    let missing = detect_missing(&dirs);
    debug!("GPU runtime libraries that cannot be opened: {missing:?}");
    let searched: Vec<PathBuf> = dirs.into_iter().map(|dir| dir.path).collect();
    let libraries: Vec<&str> = libraries().collect();
    open_libraries(&SystemLoader, &libraries, searched, |directory| {
        logging::install_if_requested();
        transcribe_cpp::init_backends_default().map_err(|error| Unavailable::Backends {
            source: Arc::new(error),
        })?;
        Ok(Runtime {
            version: transcribe_cpp::version(),
            directory,
        })
    })
}

/// Opens each of `libraries`, then runs `start` with the folder of the last one. A library that
/// no folder holds is not opened at all, so looking again costs a few file checks.
fn open_libraries(
    loader: &impl Loader,
    libraries: &[&str],
    searched: Vec<PathBuf>,
    start: impl FnOnce(Option<PathBuf>) -> Result<Runtime, Unavailable>,
) -> Result<Runtime, Unavailable> {
    let mut directory = None;
    for library in libraries {
        directory = Some(match open_in_first(loader, library, &searched) {
            Opening::Opened(dir) => dir.to_path_buf(),
            Opening::Unopenable(dir) => {
                return Err(Unavailable::LibraryUnloadable {
                    library: (*library).to_owned(),
                    directory: dir.to_path_buf(),
                });
            }
            Opening::Absent => {
                return Err(Unavailable::LibraryMissing {
                    library: (*library).to_owned(),
                    searched,
                });
            }
        });
    }
    start(directory)
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
    use std::{cell::Cell, collections::HashSet};

    use pretty_assertions::assert_eq;

    use super::*;

    #[derive(Default)]
    struct Fake {
        files: HashSet<(PathBuf, String)>,
        unopenable: HashSet<(PathBuf, String)>,
        system: HashSet<String>,
        opened: Cell<usize>,
    }

    impl Loader for Fake {
        fn contains(&self, dir: &Path, library: &str) -> bool {
            self.files
                .contains(&(dir.to_path_buf(), library.to_owned()))
        }

        fn open_in(&self, dir: &Path, library: &str) -> bool {
            self.opened.set(self.opened.get() + 1);
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

    fn runtime(version: &str) -> Runtime {
        Runtime {
            version: version.to_owned(),
            directory: None,
        }
    }

    fn missing() -> Unavailable {
        Unavailable::LibraryMissing {
            library: "transcribe.dll".to_owned(),
            searched: Vec::new(),
        }
    }

    fn no_backend() -> Unavailable {
        Unavailable::Backends {
            source: Arc::new(transcribe_cpp::Error::Backend("no device".to_owned())),
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

    #[test]
    fn a_library_that_is_absent_is_reported_without_opening_or_starting_anything() {
        let loader = fake(&[]);
        let started = Cell::new(0);
        let outcome = open_libraries(&loader, &["transcribe.dll"], dirs(&["/a"]), |_| {
            started.set(started.get() + 1);
            Ok(runtime("1"))
        });
        assert!(matches!(
            outcome,
            Err(Unavailable::LibraryMissing { library, searched })
                if library == "transcribe.dll" && searched == dirs(&["/a"])
        ));
        assert_eq!((loader.opened.get(), started.get()), (0, 0));
    }

    #[test]
    fn the_start_gets_the_folder_the_last_library_opened_from() {
        let loader = fake(&[("/b", "transcribe.dll")]);
        let outcome = open_libraries(
            &loader,
            &["transcribe.dll"],
            dirs(&["/a", "/b"]),
            |directory| {
                Ok(Runtime {
                    version: "1".to_owned(),
                    directory,
                })
            },
        );
        assert_eq!(
            outcome.ok().and_then(|runtime| runtime.directory),
            Some(PathBuf::from("/b"))
        );
    }

    #[test]
    fn a_library_that_was_missing_is_loaded_by_a_later_call() {
        let slot = Slot::new();
        let calls = Cell::new(0);
        let first = slot.get_or_load(|| {
            calls.set(calls.get() + 1);
            Err(missing())
        });
        assert!(matches!(first, Err(Unavailable::LibraryMissing { .. })));
        let second = slot.get_or_load(|| {
            calls.set(calls.get() + 1);
            Ok(runtime("1"))
        });
        assert_eq!(
            second.ok().map(|runtime| runtime.version.as_str()),
            Some("1")
        );
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn a_failure_is_tried_again_every_time() {
        let slot = Slot::new();
        let calls = Cell::new(0);
        for _ in 0..3 {
            let outcome = slot.get_or_load(|| {
                calls.set(calls.get() + 1);
                Err(Unavailable::LibraryUnloadable {
                    library: "transcribe.dll".to_owned(),
                    directory: PathBuf::from("/a"),
                })
            });
            assert!(outcome.is_err());
        }
        assert_eq!(calls.get(), 3);
        assert!(slot.loaded().is_none());
    }

    #[test]
    fn a_success_is_kept_and_not_loaded_again() {
        let slot = Slot::new();
        let calls = Cell::new(0);
        for _ in 0..3 {
            let outcome = slot.get_or_load(|| {
                calls.set(calls.get() + 1);
                Ok(runtime("1"))
            });
            assert_eq!(
                outcome.ok().map(|runtime| runtime.version.as_str()),
                Some("1")
            );
        }
        assert_eq!(calls.get(), 1);
        assert_eq!(
            slot.loaded().map(|runtime| runtime.version.as_str()),
            Some("1")
        );
    }

    #[test]
    fn a_loaded_library_stays_whatever_a_later_call_would_find() {
        let slot = Slot::new();
        assert!(slot.get_or_load(|| Ok(runtime("1"))).is_ok());
        assert!(slot.get_or_load(|| Err(missing())).is_ok());
    }

    #[test]
    fn a_missing_backend_is_final_and_not_scanned_again() {
        let slot = Slot::new();
        let calls = Cell::new(0);
        for _ in 0..3 {
            let outcome = slot.get_or_load(|| {
                calls.set(calls.get() + 1);
                Err(no_backend())
            });
            assert!(matches!(outcome, Err(Unavailable::Backends { .. })));
        }
        assert_eq!(calls.get(), 1);
    }
}
