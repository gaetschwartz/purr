//! Where the GPU runtime libraries are looked for, in order.
//!
//! The order is the same for the command line and the desktop app:
//!
//! 1. the directories of `PURR_RUNTIME_DIR`, so a developer or an installer can force a place;
//! 2. next to the executable, so a portable folder keeps working and matches the build it ships
//!    with: the executable's own directory on Windows, `lib` on Linux, and a `runtime` folder on
//!    both;
//! 3. the per-user data directory, shared by every copy of Purr that the user runs;
//! 4. the CUDA toolkit (`CUDA_PATH`) on Windows.
//!
//! Whatever is not found there is left to the system's own library search.

use std::{
    fmt,
    path::{Path, PathBuf},
};

pub const OVERRIDE_ENV: &str = "PURR_RUNTIME_DIR";

const RUNTIME_DIR_NAME: &str = "runtime";
const LINUX_BUNDLED_DIR_NAME: &str = "lib";
const USER_DATA_APP_DIR_NAME: &str = "purr";
const CUDA_ENV: &str = "CUDA_PATH";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    Windows,
    Linux,
    /// Needs no runtime library directory.
    Other,
}

impl Os {
    pub const CURRENT: Self = if cfg!(windows) {
        Self::Windows
    } else if cfg!(target_os = "linux") {
        Self::Linux
    } else {
        Self::Other
    };
}

/// Why a directory is searched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    Override,
    Executable,
    ExecutableRuntime,
    ExecutableLib,
    UserData,
    CudaToolkit,
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Override => "set by PURR_RUNTIME_DIR",
            Self::Executable => "next to the executable",
            Self::ExecutableRuntime => "runtime folder next to the executable",
            Self::ExecutableLib => "lib folder next to the executable",
            Self::UserData => "per-user runtime folder",
            Self::CudaToolkit => "CUDA toolkit",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchDir {
    pub path: PathBuf,
    pub origin: Origin,
}

/// What the order is computed from.
#[derive(Debug, Clone)]
pub struct Environment {
    pub os: Os,
    pub override_dirs: Vec<PathBuf>,
    pub executable_dir: Option<PathBuf>,
    /// The per-user directory for local application data.
    pub user_data_dir: Option<PathBuf>,
    pub cuda_path: Option<PathBuf>,
}

impl Environment {
    #[must_use]
    pub fn current() -> Self {
        Self {
            os: Os::CURRENT,
            override_dirs: std::env::var_os(OVERRIDE_ENV)
                .map(|value| {
                    std::env::split_paths(&value)
                        .filter(|dir| !dir.as_os_str().is_empty())
                        .collect()
                })
                .unwrap_or_default(),
            executable_dir: std::env::current_exe()
                .ok()
                .and_then(|exe| exe.parent().map(Path::to_path_buf)),
            user_data_dir: directories::BaseDirs::new()
                .map(|dirs| dirs.data_local_dir().to_path_buf()),
            cuda_path: std::env::var_os(CUDA_ENV).map(PathBuf::from),
        }
    }

    fn user_runtime_dir(&self) -> Option<PathBuf> {
        self.user_data_dir
            .as_ref()
            .map(|dir| dir.join(USER_DATA_APP_DIR_NAME).join(RUNTIME_DIR_NAME))
    }
}

fn push_unique(dirs: &mut Vec<SearchDir>, path: PathBuf, origin: Origin) {
    if !dirs.iter().any(|known| known.path == path) {
        dirs.push(SearchDir { path, origin });
    }
}

/// The directories to search, first match wins.
#[must_use]
pub fn search_dirs(env: &Environment) -> Vec<SearchDir> {
    if env.os == Os::Other {
        return Vec::new();
    }
    let mut dirs = Vec::new();
    for dir in &env.override_dirs {
        push_unique(&mut dirs, dir.clone(), Origin::Override);
    }
    if let Some(executable_dir) = &env.executable_dir {
        match env.os {
            Os::Windows => push_unique(&mut dirs, executable_dir.clone(), Origin::Executable),
            Os::Linux => push_unique(
                &mut dirs,
                executable_dir.join(LINUX_BUNDLED_DIR_NAME),
                Origin::ExecutableLib,
            ),
            Os::Other => {}
        }
        push_unique(
            &mut dirs,
            executable_dir.join(RUNTIME_DIR_NAME),
            Origin::ExecutableRuntime,
        );
    }
    if let Some(user_dir) = env.user_runtime_dir() {
        push_unique(&mut dirs, user_dir, Origin::UserData);
    }
    if env.os == Os::Windows
        && let Some(cuda_path) = &env.cuda_path
    {
        for dir in [cuda_path.join("bin").join("x64"), cuda_path.join("bin")] {
            push_unique(&mut dirs, dir, Origin::CudaToolkit);
        }
    }
    dirs
}

/// Where a runtime pack should be unpacked: the first override, else the per-user directory.
#[must_use]
pub fn install_dir(env: &Environment) -> Option<PathBuf> {
    env.override_dirs
        .first()
        .cloned()
        .or_else(|| env.user_runtime_dir())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(text: &str) -> PathBuf {
        PathBuf::from(text)
    }

    fn dir(text: &str, origin: Origin) -> SearchDir {
        SearchDir {
            path: path(text),
            origin,
        }
    }

    fn environment(os: Os) -> Environment {
        Environment {
            os,
            override_dirs: Vec::new(),
            executable_dir: Some(path("/app")),
            user_data_dir: Some(path("/data")),
            cuda_path: Some(path("/cuda")),
        }
    }

    #[test]
    fn windows_searches_the_override_the_executable_the_user_directory_then_the_toolkit() {
        let env = Environment {
            override_dirs: vec![path("/first"), path("/second")],
            ..environment(Os::Windows)
        };
        assert_eq!(
            search_dirs(&env),
            [
                dir("/first", Origin::Override),
                dir("/second", Origin::Override),
                dir("/app", Origin::Executable),
                dir("/app/runtime", Origin::ExecutableRuntime),
                dir("/data/purr/runtime", Origin::UserData),
                dir("/cuda/bin/x64", Origin::CudaToolkit),
                dir("/cuda/bin", Origin::CudaToolkit),
            ]
        );
    }

    #[test]
    fn linux_searches_the_bundled_lib_folder_instead_of_the_executable_directory() {
        assert_eq!(
            search_dirs(&environment(Os::Linux)),
            [
                dir("/app/lib", Origin::ExecutableLib),
                dir("/app/runtime", Origin::ExecutableRuntime),
                dir("/data/purr/runtime", Origin::UserData),
            ]
        );
    }

    #[test]
    fn other_systems_search_nothing() {
        let env = Environment {
            override_dirs: vec![path("/first")],
            ..environment(Os::Other)
        };
        assert_eq!(search_dirs(&env), Vec::<SearchDir>::new());
    }

    #[test]
    fn missing_pieces_are_left_out() {
        let env = Environment {
            os: Os::Windows,
            override_dirs: Vec::new(),
            executable_dir: None,
            user_data_dir: None,
            cuda_path: None,
        };
        assert_eq!(search_dirs(&env), Vec::<SearchDir>::new());
    }

    #[test]
    fn a_directory_named_twice_keeps_its_first_place() {
        let env = Environment {
            override_dirs: vec![path("/app"), path("/data/purr/runtime")],
            ..environment(Os::Windows)
        };
        assert_eq!(
            search_dirs(&env),
            [
                dir("/app", Origin::Override),
                dir("/data/purr/runtime", Origin::Override),
                dir("/app/runtime", Origin::ExecutableRuntime),
                dir("/cuda/bin/x64", Origin::CudaToolkit),
                dir("/cuda/bin", Origin::CudaToolkit),
            ]
        );
    }

    #[test]
    fn packs_install_into_the_override_first() {
        let env = Environment {
            override_dirs: vec![path("/first"), path("/second")],
            ..environment(Os::Windows)
        };
        assert_eq!(install_dir(&env), Some(path("/first")));
    }

    #[test]
    fn packs_install_into_the_user_directory_without_an_override() {
        assert_eq!(
            install_dir(&environment(Os::Linux)),
            Some(path("/data/purr/runtime"))
        );
    }

    #[test]
    fn nowhere_to_install_without_an_override_or_a_user_directory() {
        let env = Environment {
            user_data_dir: None,
            ..environment(Os::Windows)
        };
        assert_eq!(install_dir(&env), None);
    }
}
