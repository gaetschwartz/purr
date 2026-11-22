#![cfg_attr(not(test), expect(dead_code, reason = "used by the batch run"))]

use std::io;
use std::path::{Path, PathBuf};
use url::Url;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputForm {
    Single,
    Batch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputSource {
    File(PathBuf),
    Url(Url),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputItem {
    pub index: usize,
    pub source: InputSource,
    pub display: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inputs {
    pub form: InputForm,
    pub items: Vec<InputItem>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchFile {
    #[cfg_attr(
        test,
        expect(dead_code, reason = "standard input cannot be read in a test")
    )]
    Stdin,
    Path(PathBuf),
}

#[oopsie::oopsie]
pub enum InputsError {
    #[oopsie(
        "Not a file path or URL: '{input}'",
        code = "purr::input::invalid",
        help = "Pass an audio file path, a wildcard or an http(s) URL"
    )]
    Invalid { input: String },
    #[oopsie(
        "Not a valid wildcard: '{pattern}'",
        code = "purr::input::invalid_pattern"
    )]
    InvalidPattern {
        pattern: String,
        source: glob::PatternError,
    },
    #[oopsie(
        "Cannot read the files matching '{pattern}'",
        code = "purr::input::pattern_unreadable"
    )]
    ReadMatch {
        pattern: String,
        source: glob::GlobError,
    },
    #[oopsie(
        "No file matches '{pattern}'",
        code = "purr::input::no_match",
        help = "Check the path and the wildcard"
    )]
    NoMatch { pattern: String },
    #[oopsie(
        "{} is a directory",
        path.display(),
        code = "purr::input::directory",
        help = "Name the files with a wildcard, such as 'dir/*.wav', or 'dir/**/*.mp3' to include subdirectories"
    )]
    IsDirectory { path: PathBuf },
    #[oopsie(
        "Cannot read the batch file {}",
        path.display(),
        code = "purr::input::batch_file_unreadable"
    )]
    ReadBatchFile { path: PathBuf, source: io::Error },
    #[oopsie(
        "Cannot read the batch list from standard input",
        code = "purr::input::batch_stdin_unreadable"
    )]
    ReadBatchStdin { source: io::Error },
    #[oopsie(
        "No input to transcribe",
        code = "purr::input::none",
        help = "Pass files or URLs, or a batch file with --batch-file"
    )]
    NoInputs,
}

enum Entry {
    Url(Url),
    Plain(PathBuf),
    Pattern(String),
}

impl Entry {
    fn forces_batch(&self) -> bool {
        match self {
            Self::Url(_) | Self::Plain(_) => false,
            Self::Pattern(_) => true,
        }
    }
}

pub fn expand(arguments: &[String], batch_file: Option<&BatchFile>) -> Result<Inputs, InputsError> {
    let mut texts: Vec<String> = arguments.to_vec();
    if let Some(batch_file) = batch_file {
        texts.extend(read_batch_file(batch_file)?);
    }

    let mut batch = batch_file.is_some() || texts.len() > 1;
    let mut sources: Vec<(InputSource, String)> = Vec::new();
    for text in &texts {
        let entry = classify(text)?;
        batch |= entry.forces_batch();
        match entry {
            Entry::Url(url) => sources.push((InputSource::Url(url), text.clone())),
            Entry::Plain(path) => sources.push((InputSource::File(path), text.clone())),
            Entry::Pattern(pattern) => {
                for path in expand_pattern(&pattern)? {
                    let display = path.display().to_string();
                    sources.push((InputSource::File(path), display));
                }
            }
        }
    }

    if sources.is_empty() {
        return Err(InputsError::NoInputs);
    }

    let items = sources
        .into_iter()
        .enumerate()
        .map(|(position, (source, display))| InputItem {
            index: position + 1,
            source,
            display,
        })
        .collect();
    let form = if batch {
        InputForm::Batch
    } else {
        InputForm::Single
    };
    Ok(Inputs { form, items })
}

fn classify(text: &str) -> Result<Entry, InputsError> {
    if text.trim().is_empty() {
        return Err(InputsError::Invalid {
            input: text.to_owned(),
        });
    }
    if let Ok(url) = Url::parse(text) {
        match url.scheme() {
            "http" | "https" => return Ok(Entry::Url(url)),
            _ => {}
        }
    }

    let path = Path::new(text);
    if path.is_dir() {
        Err(InputsError::IsDirectory {
            path: path.to_path_buf(),
        })
    } else if is_wildcard(text) && !path.exists() {
        Ok(Entry::Pattern(text.to_owned()))
    } else {
        Ok(Entry::Plain(path.to_path_buf()))
    }
}

fn is_wildcard(text: &str) -> bool {
    glob::Pattern::escape(text) != text
}

fn read_batch_file(batch_file: &BatchFile) -> Result<Vec<String>, InputsError> {
    let content = match batch_file {
        BatchFile::Stdin => io::read_to_string(io::stdin())
            .map_err(|source| InputsError::ReadBatchStdin { source })?,
        BatchFile::Path(path) => {
            std::fs::read_to_string(path).map_err(|source| InputsError::ReadBatchFile {
                path: path.clone(),
                source,
            })?
        }
    };
    Ok(batch_file_lines(&content))
}

/// The batch file format: one input per line; blank lines and `#` lines are ignored.
fn batch_file_lines(content: &str) -> Vec<String> {
    content
        .trim_start_matches('\u{feff}')
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect()
}

fn expand_pattern(pattern: &str) -> Result<Vec<PathBuf>, InputsError> {
    let options = glob::MatchOptions {
        case_sensitive: !cfg!(windows),
        require_literal_separator: true,
        require_literal_leading_dot: false,
    };
    let paths =
        glob::glob_with(pattern, options).map_err(|source| InputsError::InvalidPattern {
            pattern: pattern.to_owned(),
            source,
        })?;

    let mut matches = Vec::new();
    for entry in paths {
        let path = entry.map_err(|source| InputsError::ReadMatch {
            pattern: pattern.to_owned(),
            source,
        })?;
        if path.is_file() {
            matches.push(path);
        }
    }
    if matches.is_empty() {
        return Err(InputsError::NoMatch {
            pattern: pattern.to_owned(),
        });
    }
    matches.sort();
    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use tempfile::TempDir;

    fn touch(dir: &Path, relative: &str) -> PathBuf {
        let path = dir.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, b"x").unwrap();
        path
    }

    fn pattern_in(dir: &Path, tail: &str) -> String {
        let escaped = glob::Pattern::escape(&dir.display().to_string());
        format!("{escaped}{}{tail}", std::path::MAIN_SEPARATOR)
    }

    fn args(texts: &[&str]) -> Vec<String> {
        texts.iter().map(|text| (*text).to_owned()).collect()
    }

    fn files_of(inputs: &Inputs) -> Vec<PathBuf> {
        inputs
            .items
            .iter()
            .map(|item| match &item.source {
                InputSource::File(path) => path.clone(),
                InputSource::Url(url) => panic!("unexpected URL {url}"),
            })
            .collect()
    }

    #[test]
    fn one_plain_path_is_single_form() {
        let inputs = expand(&args(&["missing.wav"]), None).unwrap();
        assert_eq!(inputs.form, InputForm::Single);
        assert_eq!(
            inputs.items,
            vec![InputItem {
                index: 1,
                source: InputSource::File(PathBuf::from("missing.wav")),
                display: "missing.wav".to_owned(),
            }]
        );
    }

    #[test]
    fn dotted_file_name_is_a_path() {
        let inputs = expand(&args(&["my.recording.v2.wav"]), None).unwrap();
        assert_eq!(
            files_of(&inputs),
            vec![PathBuf::from("my.recording.v2.wav")]
        );
    }

    #[test]
    fn two_arguments_are_batch_form() {
        let inputs = expand(&args(&["a.wav", "b.wav"]), None).unwrap();
        assert_eq!(inputs.form, InputForm::Batch);
        assert_eq!(inputs.items.len(), 2);
    }

    #[test]
    fn wildcard_matching_one_file_is_still_batch_form() {
        let dir = TempDir::new().unwrap();
        let only = touch(dir.path(), "only.wav");
        let inputs = expand(&[pattern_in(dir.path(), "*.wav")], None).unwrap();
        assert_eq!(inputs.form, InputForm::Batch);
        assert_eq!(files_of(&inputs), vec![only]);
    }

    #[test]
    fn wildcard_matches_are_sorted() {
        let dir = TempDir::new().unwrap();
        let b = touch(dir.path(), "b.wav");
        let c = touch(dir.path(), "c.wav");
        let a = touch(dir.path(), "a.wav");
        touch(dir.path(), "notes.txt");
        let inputs = expand(&[pattern_in(dir.path(), "*.wav")], None).unwrap();
        assert_eq!(files_of(&inputs), vec![a, b, c]);
    }

    #[test]
    fn wildcard_does_not_match_directories() {
        let dir = TempDir::new().unwrap();
        let file = touch(dir.path(), "a.wav");
        std::fs::create_dir(dir.path().join("sub.wav")).unwrap();
        let inputs = expand(&[pattern_in(dir.path(), "*.wav")], None).unwrap();
        assert_eq!(files_of(&inputs), vec![file]);
    }

    #[test]
    fn wildcard_without_matches_is_an_error() {
        let dir = TempDir::new().unwrap();
        touch(dir.path(), "a.txt");
        let error = expand(&[pattern_in(dir.path(), "*.wav")], None).unwrap_err();
        assert!(matches!(error, InputsError::NoMatch { .. }), "{error:?}");
    }

    #[test]
    fn existing_file_with_bracket_in_its_name_is_a_plain_path() {
        let dir = TempDir::new().unwrap();
        let live = touch(dir.path(), "song [live].wav");
        let inputs = expand(&[live.display().to_string()], None).unwrap();
        assert_eq!(inputs.form, InputForm::Single);
        assert_eq!(files_of(&inputs), vec![live]);
    }

    #[test]
    fn double_star_recurses_into_subdirectories() {
        let dir = TempDir::new().unwrap();
        let top = touch(dir.path(), "a.mp3");
        let deep = touch(dir.path(), "sub/deep/b.mp3");
        let nested = touch(dir.path(), "sub/c.mp3");
        touch(dir.path(), "sub/cover.jpg");
        let mut expected = vec![top, deep, nested];
        expected.sort();
        let inputs = expand(&[pattern_in(dir.path(), "**/*.mp3")], None).unwrap();
        assert_eq!(inputs.form, InputForm::Batch);
        assert_eq!(files_of(&inputs), expected);
    }

    #[test]
    fn directory_argument_is_an_error() {
        let dir = TempDir::new().unwrap();
        touch(dir.path(), "a.wav");
        let error = expand(&[dir.path().display().to_string()], None).unwrap_err();
        assert!(
            matches!(&error, InputsError::IsDirectory { path } if path == dir.path()),
            "{error:?}"
        );
    }

    #[test]
    fn directory_line_in_a_batch_file_is_an_error() {
        let dir = TempDir::new().unwrap();
        let list = dir.path().join("list.txt");
        std::fs::write(&list, format!("{}\n", dir.path().display())).unwrap();
        let error = expand(&[], Some(&BatchFile::Path(list))).unwrap_err();
        assert!(
            matches!(error, InputsError::IsDirectory { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn url_argument_is_kept_as_url() {
        let inputs = expand(&args(&["https://example.com/live.mp3"]), None).unwrap();
        assert_eq!(inputs.form, InputForm::Single);
        assert_eq!(
            inputs.items,
            vec![InputItem {
                index: 1,
                source: InputSource::Url(Url::parse("https://example.com/live.mp3").unwrap()),
                display: "https://example.com/live.mp3".to_owned(),
            }]
        );
    }

    #[test]
    fn batch_file_skips_blank_and_hash_lines() {
        let dir = TempDir::new().unwrap();
        let list = dir.path().join("list.txt");
        std::fs::write(
            &list,
            "\u{feff}# first\r\n\r\na.wav\n   \n  # indented comment\n  https://example.com/b.mp3  \r\nc.wav\n",
        )
        .unwrap();
        let inputs = expand(&[], Some(&BatchFile::Path(list))).unwrap();
        assert_eq!(inputs.form, InputForm::Batch);
        assert_eq!(
            inputs
                .items
                .iter()
                .map(|item| item.display.as_str())
                .collect::<Vec<_>>(),
            vec!["a.wav", "https://example.com/b.mp3", "c.wav"]
        );
    }

    #[test]
    fn batch_file_with_one_line_is_batch_form() {
        let dir = TempDir::new().unwrap();
        let list = dir.path().join("list.txt");
        std::fs::write(&list, "a.wav\n").unwrap();
        let inputs = expand(&[], Some(&BatchFile::Path(list))).unwrap();
        assert_eq!(inputs.form, InputForm::Batch);
    }

    #[test]
    fn batch_file_lines_may_be_wildcards() {
        let dir = TempDir::new().unwrap();
        let a = touch(dir.path(), "a.wav");
        let b = touch(dir.path(), "b.wav");
        let list = dir.path().join("list.txt");
        std::fs::write(&list, format!("{}\n", pattern_in(dir.path(), "*.wav"))).unwrap();
        let inputs = expand(&[], Some(&BatchFile::Path(list))).unwrap();
        assert_eq!(files_of(&inputs), vec![a, b]);
    }

    #[test]
    fn arguments_come_before_batch_file_entries() {
        let dir = TempDir::new().unwrap();
        let list = dir.path().join("list.txt");
        std::fs::write(&list, "b.wav\n").unwrap();
        let inputs = expand(&args(&["a.wav"]), Some(&BatchFile::Path(list))).unwrap();
        assert_eq!(
            files_of(&inputs),
            vec![PathBuf::from("a.wav"), PathBuf::from("b.wav")]
        );
    }

    #[test]
    fn empty_batch_file_is_an_error() {
        let dir = TempDir::new().unwrap();
        let list = dir.path().join("list.txt");
        std::fs::write(&list, "# nothing here\n\n").unwrap();
        let error = expand(&[], Some(&BatchFile::Path(list))).unwrap_err();
        assert!(matches!(error, InputsError::NoInputs), "{error:?}");
    }

    #[test]
    fn missing_batch_file_is_an_error() {
        let dir = TempDir::new().unwrap();
        let error = expand(&[], Some(&BatchFile::Path(dir.path().join("absent.txt")))).unwrap_err();
        assert!(
            matches!(error, InputsError::ReadBatchFile { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn no_arguments_and_no_batch_file_is_an_error() {
        let error = expand(&[], None).unwrap_err();
        assert!(matches!(error, InputsError::NoInputs), "{error:?}");
    }

    #[test]
    fn indexes_start_at_one_after_expansion() {
        let dir = TempDir::new().unwrap();
        touch(dir.path(), "a.wav");
        touch(dir.path(), "b.wav");
        let inputs = expand(
            &[
                "first.wav".to_owned(),
                pattern_in(dir.path(), "*.wav"),
                "https://example.com/z.mp3".to_owned(),
            ],
            None,
        )
        .unwrap();
        assert_eq!(
            inputs
                .items
                .iter()
                .map(|item| item.index)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn blank_argument_is_an_error() {
        for text in ["   ", ""] {
            let error = expand(&args(&[text]), None).unwrap_err();
            assert!(
                matches!(error, InputsError::Invalid { .. }),
                "{text:?}: {error:?}"
            );
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_style_pattern_expands() {
        let dir = TempDir::new().unwrap();
        let a = touch(dir.path(), "a.wav");
        let pattern = format!(
            "{}\\*.wav",
            dir.path().display().to_string().replace('/', "\\")
        );
        let inputs = expand(&[pattern], None).unwrap();
        assert_eq!(files_of(&inputs), vec![a]);
    }

    #[cfg(windows)]
    #[test]
    fn wildcard_matching_ignores_case_on_windows() {
        let dir = TempDir::new().unwrap();
        let loud = touch(dir.path(), "LOUD.WAV");
        let inputs = expand(&[pattern_in(dir.path(), "*.wav")], None).unwrap();
        assert_eq!(files_of(&inputs), vec![loud]);
    }
}
