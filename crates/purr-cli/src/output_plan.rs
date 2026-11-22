#![cfg_attr(not(test), expect(dead_code, reason = "used by the batch run"))]

use crate::cli::OutputFormat;
use crate::inputs::{InputForm, Inputs};
use crate::template::{OutputTemplate, OutputTemplateError};
use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::hash::{Hash, Hasher};
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sink {
    Stdout { batch: bool },
    PerInput(Vec<PathBuf>),
    Merged(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputPlan {
    pub sink: Sink,
    pub skipped: Vec<usize>,
}

#[oopsie::oopsie]
pub enum OutputPlanError {
    #[oopsie(transparent)]
    Template { source: OutputTemplateError },
    #[oopsie(
        "SRT output for several inputs needs an output file",
        code = "purr::output::srt_needs_file",
        help = "Pass --output-file with a name such as '{{stem}}.{{ext}}'"
    )]
    SrtNeedsFile,
    #[oopsie(
        "--merge needs an output file",
        code = "purr::output::merge_needs_file",
        help = "Pass --output-file with the name of the merged file"
    )]
    MergeNeedsFile,
    #[oopsie(
        "Several inputs need one output file each, or --merge",
        code = "purr::output::placeholderless_file",
        help = "Put a placeholder such as {{stem}} or {{index}} in the output file name, or add --merge"
    )]
    PlaceholderlessFile,
    #[oopsie(
        "--merge writes one file, so the output file name takes no placeholders",
        code = "purr::output::merge_with_template"
    )]
    MergeWithTemplate,
    #[oopsie(
        "SRT output cannot be merged into one file",
        code = "purr::output::merge_srt"
    )]
    MergeSrt,
    #[oopsie(
        "--merge needs several inputs",
        code = "purr::output::merge_in_single_form"
    )]
    MergeInSingleForm,
    #[oopsie(
        "--skip-existing needs one output file per input",
        code = "purr::output::skip_existing_needs_files",
        help = "Pass --output-file with a placeholder such as {{stem}}"
    )]
    SkipExistingNeedsPerInputFiles,
    #[oopsie(
        "'{first}' and '{second}' would both be written to {}",
        path.display(),
        code = "purr::output::collision",
        help = "Add {{index}} or {{dir}} to the output file name"
    )]
    Collision {
        path: PathBuf,
        first: String,
        second: String,
    },
}

pub fn plan(
    inputs: &Inputs,
    format: OutputFormat,
    file: Option<&str>,
    merge: bool,
    skip_existing: bool,
) -> Result<OutputPlan, OutputPlanError> {
    let batch = match inputs.form {
        InputForm::Single => false,
        InputForm::Batch => true,
    };
    if merge && !batch {
        return Err(OutputPlanError::MergeInSingleForm);
    }
    if merge && format == OutputFormat::Srt {
        return Err(OutputPlanError::MergeSrt);
    }

    let Some(file) = file else {
        if merge {
            return Err(OutputPlanError::MergeNeedsFile);
        }
        if batch && format == OutputFormat::Srt {
            return Err(OutputPlanError::SrtNeedsFile);
        }
        if skip_existing {
            return Err(OutputPlanError::SkipExistingNeedsPerInputFiles);
        }
        return Ok(OutputPlan {
            sink: Sink::Stdout { batch },
            skipped: Vec::new(),
        });
    };

    let template = output_template(file)?;
    if merge {
        let Some(path) = template.literal() else {
            return Err(OutputPlanError::MergeWithTemplate);
        };
        if skip_existing {
            return Err(OutputPlanError::SkipExistingNeedsPerInputFiles);
        }
        return Ok(OutputPlan {
            sink: Sink::Merged(path.to_path_buf()),
            skipped: Vec::new(),
        });
    }
    if batch && !template.has_placeholders() {
        return Err(OutputPlanError::PlaceholderlessFile);
    }

    let mut paths = Vec::with_capacity(inputs.items.len());
    let mut taken: HashMap<CollisionKey, usize> = HashMap::new();
    for (position, item) in inputs.items.iter().enumerate() {
        let path = template.render(item, format)?;
        if let Some(earlier) = taken.insert(CollisionKey::new(&path), position)
            && let Some(first) = inputs.items.get(earlier)
        {
            return Err(OutputPlanError::Collision {
                path,
                first: first.display.clone(),
                second: item.display.clone(),
            });
        }
        paths.push(path);
    }

    let skipped = if skip_existing {
        paths
            .iter()
            .enumerate()
            .filter(|(_, path)| path.exists())
            .map(|(position, _)| position)
            .collect()
    } else {
        Vec::new()
    };
    Ok(OutputPlan {
        sink: Sink::PerInput(paths),
        skipped,
    })
}

fn output_template(file: &str) -> Result<OutputTemplate, OutputTemplateError> {
    let path = Path::new(file);
    if path.is_dir() {
        Ok(OutputTemplate::in_directory(path.to_path_buf()))
    } else {
        OutputTemplate::parse(file)
    }
}

#[derive(Debug, Eq)]
struct CollisionKey(Vec<OsString>);

impl CollisionKey {
    fn new(path: &Path) -> Self {
        Self(
            path.components()
                .filter(|component| !matches!(component, Component::CurDir))
                .map(|component| component.as_os_str().to_owned())
                .collect(),
        )
    }
}

impl PartialEq for CollisionKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(&other.0)
                .all(|(left, right)| names_match(left, right))
    }
}

impl Hash for CollisionKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for name in &self.0 {
            let bytes = name.as_encoded_bytes();
            if cfg!(windows) {
                bytes.to_ascii_lowercase().hash(state);
            } else {
                bytes.hash(state);
            }
        }
    }
}

fn names_match(left: &OsStr, right: &OsStr) -> bool {
    if cfg!(windows) {
        left.eq_ignore_ascii_case(right)
    } else {
        left == right
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::{InputItem, InputSource};
    use pretty_assertions::assert_eq;
    use tempfile::TempDir;

    fn item(index: usize, path: &str) -> InputItem {
        InputItem {
            index,
            source: InputSource::File(PathBuf::from(path)),
            display: path.to_owned(),
        }
    }

    fn single() -> Inputs {
        Inputs {
            form: InputForm::Single,
            items: vec![item(1, "a.wav")],
        }
    }

    fn batch(names: &[&str]) -> Inputs {
        Inputs {
            form: InputForm::Batch,
            items: names
                .iter()
                .enumerate()
                .map(|(position, name)| item(position + 1, name))
                .collect(),
        }
    }

    #[test]
    fn single_form_without_file_is_stdout() {
        let plan = plan(&single(), OutputFormat::Text, None, false, false).unwrap();
        assert_eq!(
            plan,
            OutputPlan {
                sink: Sink::Stdout { batch: false },
                skipped: vec![],
            }
        );
    }

    #[test]
    fn single_form_with_a_template_expands_it() {
        let plan = plan(
            &single(),
            OutputFormat::Text,
            Some("out_{stem}.txt"),
            false,
            false,
        )
        .unwrap();
        assert_eq!(plan.sink, Sink::PerInput(vec![PathBuf::from("out_a.txt")]));
    }

    #[test]
    fn single_form_with_a_literal_file_is_literal() {
        let plan = plan(
            &single(),
            OutputFormat::Json,
            Some("out.json"),
            false,
            false,
        )
        .unwrap();
        assert_eq!(plan.sink, Sink::PerInput(vec![PathBuf::from("out.json")]));
    }

    #[test]
    fn batch_without_file_is_stdout_in_batch_form() {
        let plan = plan(
            &batch(&["a.wav", "b.wav"]),
            OutputFormat::Json,
            None,
            false,
            false,
        )
        .unwrap();
        assert_eq!(plan.sink, Sink::Stdout { batch: true });
    }

    #[test]
    fn batch_srt_without_file_is_an_error() {
        let error = plan(
            &batch(&["a.wav", "b.wav"]),
            OutputFormat::Srt,
            None,
            false,
            false,
        )
        .unwrap_err();
        assert!(matches!(error, OutputPlanError::SrtNeedsFile), "{error:?}");
    }

    #[test]
    fn merge_without_file_is_an_error() {
        let error = plan(
            &batch(&["a.wav", "b.wav"]),
            OutputFormat::Text,
            None,
            true,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(error, OutputPlanError::MergeNeedsFile),
            "{error:?}"
        );
    }

    #[test]
    fn existing_directory_means_stem_and_ext_inside_it() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().display().to_string();
        let plan = plan(
            &batch(&["a.wav", "b.wav"]),
            OutputFormat::Srt,
            Some(&file),
            false,
            false,
        )
        .unwrap();
        assert_eq!(
            plan.sink,
            Sink::PerInput(vec![dir.path().join("a.srt"), dir.path().join("b.srt")])
        );
    }

    #[test]
    fn only_an_existing_directory_is_a_directory() {
        let error = plan(
            &batch(&["a.wav", "b.wav"]),
            OutputFormat::Json,
            Some("out/"),
            false,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(error, OutputPlanError::PlaceholderlessFile),
            "{error:?}"
        );
    }

    #[test]
    fn directory_with_braces_in_its_name_is_not_a_template() {
        let dir = TempDir::new().unwrap();
        let braced = dir.path().join("{x} out");
        std::fs::create_dir(&braced).unwrap();
        let plan = plan(
            &batch(&["a.wav"]),
            OutputFormat::Text,
            Some(&braced.display().to_string()),
            false,
            false,
        )
        .unwrap();
        assert_eq!(plan.sink, Sink::PerInput(vec![braced.join("a.txt")]));
    }

    #[test]
    fn batch_with_placeholders_is_aligned_with_the_items() {
        let plan = plan(
            &batch(&["x/a.wav", "y/b.wav", "z/c.wav"]),
            OutputFormat::Text,
            Some("{index}_{stem}.{ext}"),
            false,
            false,
        )
        .unwrap();
        assert_eq!(
            plan.sink,
            Sink::PerInput(vec![
                PathBuf::from("1_a.txt"),
                PathBuf::from("2_b.txt"),
                PathBuf::from("3_c.txt"),
            ])
        );
    }

    #[test]
    fn two_inputs_rendering_to_one_path_collide() {
        let error = plan(
            &batch(&["x/a.wav", "y/a.wav"]),
            OutputFormat::Text,
            Some("{stem}.txt"),
            false,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(
                &error,
                OutputPlanError::Collision { path, first, second }
                    if path == Path::new("a.txt") && first == "x/a.wav" && second == "y/a.wav"
            ),
            "{error:?}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn collisions_ignore_case_on_windows() {
        let error = plan(
            &batch(&["x/a.wav", "y/A.wav"]),
            OutputFormat::Text,
            Some("{stem}.txt"),
            false,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(error, OutputPlanError::Collision { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn batch_file_without_placeholders_and_without_merge_is_an_error() {
        let error = plan(
            &batch(&["a.wav", "b.wav"]),
            OutputFormat::Text,
            Some("all.txt"),
            false,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(error, OutputPlanError::PlaceholderlessFile),
            "{error:?}"
        );
    }

    #[test]
    fn merge_with_a_literal_file_is_merged() {
        let plan = plan(
            &batch(&["a.wav", "b.wav"]),
            OutputFormat::Json,
            Some("all.jsonl"),
            true,
            false,
        )
        .unwrap();
        assert_eq!(plan.sink, Sink::Merged(PathBuf::from("all.jsonl")));
    }

    #[test]
    fn merge_with_a_template_is_an_error() {
        let error = plan(
            &batch(&["a.wav", "b.wav"]),
            OutputFormat::Text,
            Some("{stem}.txt"),
            true,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(error, OutputPlanError::MergeWithTemplate),
            "{error:?}"
        );
    }

    #[test]
    fn merge_with_srt_is_an_error() {
        let error = plan(
            &batch(&["a.wav", "b.wav"]),
            OutputFormat::Srt,
            Some("all.srt"),
            true,
            false,
        )
        .unwrap_err();
        assert!(matches!(error, OutputPlanError::MergeSrt), "{error:?}");
    }

    #[test]
    fn merge_in_single_form_is_an_error() {
        let error = plan(&single(), OutputFormat::Text, Some("all.txt"), true, false).unwrap_err();
        assert!(
            matches!(error, OutputPlanError::MergeInSingleForm),
            "{error:?}"
        );
    }

    #[test]
    fn skip_existing_marks_final_paths_that_exist_and_ignores_part_files() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"done").unwrap();
        std::fs::write(dir.path().join("b.txt.part"), b"half").unwrap();
        let file = format!(
            "{}{}{{stem}}.txt",
            dir.path().display(),
            std::path::MAIN_SEPARATOR
        );
        let plan = plan(
            &batch(&["a.wav", "b.wav", "c.wav"]),
            OutputFormat::Text,
            Some(&file),
            false,
            true,
        )
        .unwrap();
        assert_eq!(plan.skipped, vec![0]);
    }

    #[test]
    fn skip_existing_with_stdout_is_an_error() {
        let error = plan(
            &batch(&["a.wav", "b.wav"]),
            OutputFormat::Text,
            None,
            false,
            true,
        )
        .unwrap_err();
        assert!(
            matches!(error, OutputPlanError::SkipExistingNeedsPerInputFiles),
            "{error:?}"
        );
    }

    #[test]
    fn skip_existing_with_merged_is_an_error() {
        let error = plan(
            &batch(&["a.wav", "b.wav"]),
            OutputFormat::Text,
            Some("all.txt"),
            true,
            true,
        )
        .unwrap_err();
        assert!(
            matches!(error, OutputPlanError::SkipExistingNeedsPerInputFiles),
            "{error:?}"
        );
    }

    #[test]
    fn unknown_placeholder_surfaces_as_a_template_error() {
        let error = plan(
            &batch(&["a.wav", "b.wav"]),
            OutputFormat::Text,
            Some("{title}.txt"),
            false,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                OutputPlanError::Template {
                    source: OutputTemplateError::UnknownPlaceholder { .. }
                }
            ),
            "{error:?}"
        );
    }
}
