use std::fs::File;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

const PART_SUFFIX: &str = ".part";

#[oopsie::oopsie]
pub enum WriterError {
    #[oopsie(
        "Output path has no file name: {}",
        path.display(),
        code = "purr::output::no_file_name"
    )]
    NoFileName { path: PathBuf },
    #[oopsie(
        "Cannot create the folder {}",
        path.display(),
        code = "purr::output::create_directory_failed"
    )]
    CreateDirectory { path: PathBuf, source: io::Error },
    #[oopsie(
        "Cannot create {}",
        path.display(),
        code = "purr::output::create_failed"
    )]
    Create { path: PathBuf, source: io::Error },
    #[oopsie(
        "Cannot write to {}",
        path.display(),
        code = "purr::output::append_failed"
    )]
    Append { path: PathBuf, source: io::Error },
    #[oopsie(
        "Cannot move {} to {}",
        part.display(),
        path.display(),
        code = "purr::output::complete_failed"
    )]
    Complete {
        part: PathBuf,
        path: PathBuf,
        source: io::Error,
    },
}

pub struct PartFile {
    file: File,
    part: PathBuf,
    path: PathBuf,
}

impl PartFile {
    pub fn create(final_path: &Path) -> Result<Self, WriterError> {
        let name = final_path
            .file_name()
            .ok_or_else(|| WriterError::NoFileName {
                path: final_path.to_path_buf(),
            })?;
        if let Some(parent) = final_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|source| WriterError::CreateDirectory {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let mut part_name = name.to_os_string();
        part_name.push(PART_SUFFIX);
        let part = final_path.with_file_name(part_name);
        let file = File::create(&part).map_err(|source| WriterError::Create {
            path: part.clone(),
            source,
        })?;
        Ok(Self {
            file,
            part,
            path: final_path.to_path_buf(),
        })
    }

    pub fn append(&mut self, text: &str) -> Result<(), WriterError> {
        self.file
            .write_all(text.as_bytes())
            .and_then(|()| self.file.flush())
            .map_err(|source| WriterError::Append {
                path: self.part.clone(),
                source,
            })
    }

    pub fn part_path(&self) -> &Path {
        &self.part
    }

    pub fn complete(self) -> Result<PathBuf, WriterError> {
        let Self { file, part, path } = self;
        drop(file);
        std::fs::rename(&part, &path).map_err(|source| WriterError::Complete {
            part,
            path: path.clone(),
            source,
        })?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use tempfile::TempDir;

    #[test]
    fn append_is_visible_in_the_part_file_before_completion() {
        let dir = TempDir::new().unwrap();
        let final_path = dir.path().join("a.txt");
        let mut file = PartFile::create(&final_path).unwrap();
        file.append("first ").unwrap();
        assert_eq!(std::fs::read_to_string(file.part_path()).unwrap(), "first ");
        file.append("second").unwrap();
        assert_eq!(
            std::fs::read_to_string(file.part_path()).unwrap(),
            "first second"
        );
        assert!(!final_path.exists());
    }

    #[test]
    fn part_path_is_the_final_name_with_part_appended() {
        let dir = TempDir::new().unwrap();
        let file = PartFile::create(&dir.path().join("a.txt")).unwrap();
        assert_eq!(file.part_path(), dir.path().join("a.txt.part"));
    }

    #[test]
    fn complete_renames_over_the_final_path_and_removes_the_part_file() {
        let dir = TempDir::new().unwrap();
        let final_path = dir.path().join("a.txt");
        let mut file = PartFile::create(&final_path).unwrap();
        file.append("done").unwrap();
        let part = file.part_path().to_path_buf();
        let completed = file.complete().unwrap();
        assert_eq!(completed, final_path);
        assert_eq!(std::fs::read_to_string(&final_path).unwrap(), "done");
        assert!(!part.exists());
    }

    #[test]
    fn complete_replaces_an_existing_final_file() {
        let dir = TempDir::new().unwrap();
        let final_path = dir.path().join("a.txt");
        std::fs::write(&final_path, "old").unwrap();
        let mut file = PartFile::create(&final_path).unwrap();
        file.append("new").unwrap();
        file.complete().unwrap();
        assert_eq!(std::fs::read_to_string(&final_path).unwrap(), "new");
    }

    #[test]
    fn dropping_without_completing_keeps_the_part_file_and_its_content() {
        let dir = TempDir::new().unwrap();
        let final_path = dir.path().join("a.txt");
        let mut file = PartFile::create(&final_path).unwrap();
        file.append("partial").unwrap();
        let part = file.part_path().to_path_buf();
        drop(file);
        assert_eq!(std::fs::read_to_string(&part).unwrap(), "partial");
        assert!(!final_path.exists());
    }

    #[test]
    fn parent_directories_are_created() {
        let dir = TempDir::new().unwrap();
        let final_path = dir.path().join("x").join("y").join("a.srt");
        let mut file = PartFile::create(&final_path).unwrap();
        file.append("1").unwrap();
        file.complete().unwrap();
        assert_eq!(std::fs::read_to_string(&final_path).unwrap(), "1");
    }

    #[test]
    fn an_empty_file_completes_to_an_empty_final_file() {
        let dir = TempDir::new().unwrap();
        let final_path = dir.path().join("a.txt");
        PartFile::create(&final_path).unwrap().complete().unwrap();
        assert_eq!(std::fs::read_to_string(&final_path).unwrap(), "");
    }

    #[test]
    fn creating_again_starts_the_part_file_over() {
        let dir = TempDir::new().unwrap();
        let final_path = dir.path().join("a.txt");
        let mut first = PartFile::create(&final_path).unwrap();
        first.append("stale").unwrap();
        drop(first);
        let second = PartFile::create(&final_path).unwrap();
        assert_eq!(std::fs::read_to_string(second.part_path()).unwrap(), "");
    }

    #[test]
    fn a_path_without_a_file_name_is_an_error() {
        let error = PartFile::create(Path::new("..")).err().unwrap();
        assert!(matches!(error, WriterError::NoFileName { .. }));
    }
}
