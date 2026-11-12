use std::path::{Path, PathBuf};

use miette::miette;
use purr_core::accel::{
    backend::AccelBackend,
    runtime_dir::{Environment, install_dir, search_dirs},
    runtime_library::{DirContents, inspect, required},
};

use crate::cli::RuntimeCommands;

pub fn handle_runtime_command(command: RuntimeCommands) -> miette::Result<()> {
    let env = Environment::current();
    match command {
        RuntimeCommands::Path => {
            let dir = install_dir(&env).ok_or_else(|| {
                miette!("no per-user data directory is known; set PURR_RUNTIME_DIR")
            })?;
            println!("{}", dir.display());
        }
        RuntimeCommands::Status => {
            let dirs = search_dirs(&env);
            let needs: Vec<_> = AccelBackend::DEFAULT_PRIORITY
                .iter()
                .map(|backend| (*backend, required(*backend)))
                .filter(|(_, libraries)| !libraries.is_empty())
                .collect();
            let libraries: Vec<&'static str> = needs
                .iter()
                .flat_map(|(_, libraries)| libraries.iter().copied())
                .collect();
            let contents = inspect(&dirs, &libraries, Path::is_dir, Path::is_file);
            for line in render_status(&contents, &needs) {
                println!("{line}");
            }
        }
    }
    Ok(())
}

#[must_use]
pub fn searched_summary(searched: &[PathBuf]) -> String {
    if searched.is_empty() {
        return "looked in the system's library folders".to_owned();
    }
    let dirs: Vec<_> = searched
        .iter()
        .map(|dir| dir.display().to_string())
        .collect();
    format!(
        "looked in {}, then the system's library folders",
        dirs.join(", ")
    )
}

fn render_status(
    contents: &[DirContents],
    needs: &[(AccelBackend, Vec<&'static str>)],
) -> Vec<String> {
    if contents.is_empty() {
        return vec!["This platform needs no runtime library folder.".to_owned()];
    }
    let mut lines = vec![
        "GPU runtime libraries are searched here, in order, then in the system's library folders:"
            .to_owned(),
    ];
    for (index, entry) in contents.iter().enumerate() {
        let holds = if !entry.exists {
            "folder does not exist".to_owned()
        } else if entry.present.is_empty() {
            "nothing found".to_owned()
        } else {
            format!("found {}", entry.present.join(", "))
        };
        lines.push(format!(
            "  {}. {} ({}): {holds}",
            index + 1,
            entry.dir.path.display(),
            entry.dir.origin
        ));
    }
    if needs.is_empty() {
        lines.push("This build loads no GPU runtime library on demand.".to_owned());
    } else {
        lines.push("Libraries this build loads on demand:".to_owned());
        for (backend, libraries) in needs {
            lines.push(format!(
                "  {}: {}",
                backend.pretty_name(),
                libraries.join(", ")
            ));
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use purr_core::accel::runtime_dir::{Origin, SearchDir};

    use super::*;

    fn entry(path: &str, origin: Origin, exists: bool, present: &[&'static str]) -> DirContents {
        DirContents {
            dir: SearchDir {
                path: PathBuf::from(path),
                origin,
            },
            exists,
            present: present.to_vec(),
        }
    }

    #[test]
    fn status_numbers_the_directories_and_says_what_each_holds() {
        let contents = [
            entry("/app", Origin::Executable, true, &[]),
            entry("/data/runtime", Origin::UserData, true, &["a.dll", "b.dll"]),
            entry("/cuda/bin", Origin::CudaToolkit, false, &[]),
        ];
        let needs = [(AccelBackend::Cuda, vec!["a.dll", "b.dll"])];
        assert_eq!(
            render_status(&contents, &needs),
            [
                "GPU runtime libraries are searched here, in order, then in the system's library folders:",
                "  1. /app (next to the executable): nothing found",
                "  2. /data/runtime (per-user runtime folder): found a.dll, b.dll",
                "  3. /cuda/bin (CUDA toolkit): folder does not exist",
                "Libraries this build loads on demand:",
                "  CUDA: a.dll, b.dll",
            ]
        );
    }

    #[test]
    fn status_on_a_platform_without_runtime_folders_says_so() {
        assert_eq!(
            render_status(&[], &[]),
            ["This platform needs no runtime library folder."]
        );
    }

    #[test]
    fn searched_summary_lists_the_directories_then_the_system() {
        assert_eq!(
            searched_summary(&[PathBuf::from("/a"), PathBuf::from("/b")]),
            "looked in /a, /b, then the system's library folders"
        );
        assert_eq!(
            searched_summary(&[]),
            "looked in the system's library folders"
        );
    }
}
