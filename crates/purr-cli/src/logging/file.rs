use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing_subscriber::fmt::MakeWriter;

#[derive(Debug, thiserror::Error, miette::Diagnostic)]
#[error("cannot open the log file {}: {source}", path.display())]
#[diagnostic(help("check that the folder exists and that you may write to the file"))]
pub struct LogFileError {
    pub path: PathBuf,
    #[source]
    pub source: io::Error,
}

/// The `--log-file`, opened for appending. The subscriber writes every event with a single call
/// to the operating system, so nothing is buffered and nothing needs flushing when the process
/// exits, however it exits.
#[derive(Clone, Debug)]
pub struct LogFile(Arc<File>);

impl LogFile {
    pub fn open(path: &Path) -> Result<Self, LogFileError> {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map(|file| Self(Arc::new(file)))
            .map_err(|source| LogFileError {
                path: path.to_path_buf(),
                source,
            })
    }
}

impl<'a> MakeWriter<'a> for LogFile {
    type Writer = &'a File;

    fn make_writer(&'a self) -> Self::Writer {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use io::Write as _;

    #[test]
    fn opening_appends_to_what_is_already_there() -> io::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("purr.log");
        std::fs::write(&path, "first\n")?;
        let file = LogFile::open(&path).map_err(io::Error::other)?;
        let mut writer = file.make_writer();
        writer.write_all(b"second\n")?;
        assert_eq!(std::fs::read_to_string(&path)?, "first\nsecond\n");
        Ok(())
    }

    #[test]
    fn a_file_that_cannot_be_opened_is_a_typed_error() -> io::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("missing").join("purr.log");
        let error = LogFile::open(&path).err();
        assert_eq!(
            error.map(|error| (error.path, error.source.kind())),
            Some((path, io::ErrorKind::NotFound))
        );
        Ok(())
    }
}
