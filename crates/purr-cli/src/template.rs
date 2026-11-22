#![cfg_attr(not(test), expect(dead_code, reason = "used by the batch run"))]

use crate::cli::OutputFormat;
use crate::inputs::{InputItem, InputSource};
use percent_encoding::percent_decode_str;
use std::path::{Component, Path, PathBuf};
use strfmt::{FmtError, Formatter};
use url::Url;

const ALLOWED: &str = "{stem}, {name}, {dir}, {ext}, {index}";

#[oopsie::oopsie]
pub enum OutputTemplateError {
    #[oopsie(
        "Unknown placeholder '{name}' in the output name",
        code = "purr::output::unknown_placeholder",
        help = "The placeholders are {allowed}"
    )]
    UnknownPlaceholder { name: String, allowed: &'static str },
    #[oopsie(
        "Invalid output name '{template}'",
        code = "purr::output::invalid_template"
    )]
    Malformed { template: String, source: FmtError },
    #[oopsie(
        "Cannot expand '{placeholder}' for {input}: it has no usable file name",
        code = "purr::output::unnamed_input",
        help = "Use {{index}} in the output name"
    )]
    NameNeeded {
        placeholder: &'static str,
        input: String,
    },
    #[oopsie(
        "The path {} is not valid Unicode",
        path.display(),
        code = "purr::output::non_unicode_path"
    )]
    NonUnicodePath { path: PathBuf },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Placeholder {
    Stem,
    Name,
    Dir,
    Ext,
    Index,
}

impl Placeholder {
    fn from_key(key: &str) -> Option<Self> {
        match key {
            "stem" => Some(Self::Stem),
            "name" => Some(Self::Name),
            "dir" => Some(Self::Dir),
            "ext" => Some(Self::Ext),
            "index" => Some(Self::Index),
            _ => None,
        }
    }

    fn key(self) -> &'static str {
        match self {
            Self::Stem => "stem",
            Self::Name => "name",
            Self::Dir => "dir",
            Self::Ext => "ext",
            Self::Index => "index",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Form {
    Literal(PathBuf),
    Pattern(String),
    InDirectory(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputTemplate {
    form: Form,
}

impl OutputTemplate {
    pub fn in_directory(directory: PathBuf) -> Self {
        Self {
            form: Form::InDirectory(directory),
        }
    }

    pub fn parse(text: &str) -> Result<Self, OutputTemplateError> {
        let mut unknown = None;
        let mut placeholders = 0usize;
        let outcome = strfmt::strfmt_map(text, |formatter: Formatter| {
            if Placeholder::from_key(formatter.key).is_none() {
                unknown = Some(formatter.key.to_owned());
                return Err(FmtError::KeyError(formatter.key.to_owned()));
            }
            placeholders += 1;
            formatter.skip()
        });
        match outcome {
            Ok(unescaped) => Ok(Self {
                form: if placeholders == 0 {
                    Form::Literal(PathBuf::from(unescaped))
                } else {
                    Form::Pattern(text.to_owned())
                },
            }),
            Err(source) => Err(unknown.map_or_else(
                || OutputTemplateError::Malformed {
                    template: text.to_owned(),
                    source,
                },
                |name| OutputTemplateError::UnknownPlaceholder {
                    name,
                    allowed: ALLOWED,
                },
            )),
        }
    }

    pub fn has_placeholders(&self) -> bool {
        match &self.form {
            Form::Literal(_) => false,
            Form::Pattern(_) | Form::InDirectory(_) => true,
        }
    }

    pub fn literal(&self) -> Option<&Path> {
        match &self.form {
            Form::Literal(path) => Some(path),
            Form::Pattern(_) | Form::InDirectory(_) => None,
        }
    }

    pub fn render(
        &self,
        item: &InputItem,
        format: OutputFormat,
    ) -> Result<PathBuf, OutputTemplateError> {
        match &self.form {
            Form::Literal(path) => Ok(path.clone()),
            Form::Pattern(text) => render_pattern(text, item, format),
            Form::InDirectory(directory) => {
                let stem = value(Placeholder::Stem, item, format)?;
                Ok(directory.join(format!("{stem}.{}", extension(format))))
            }
        }
    }
}

fn render_pattern(
    text: &str,
    item: &InputItem,
    format: OutputFormat,
) -> Result<PathBuf, OutputTemplateError> {
    let mut failure = None;
    let outcome = strfmt::strfmt_map(text, |mut formatter: Formatter| {
        let Some(placeholder) = Placeholder::from_key(formatter.key) else {
            return Err(FmtError::KeyError(formatter.key.to_owned()));
        };
        match value(placeholder, item, format) {
            Ok(text) => formatter.str(&text),
            Err(error) => {
                failure = Some(error);
                Err(FmtError::KeyError(placeholder.key().to_owned()))
            }
        }
    });
    match outcome {
        Ok(rendered) => Ok(PathBuf::from(rendered)),
        Err(source) => Err(failure.unwrap_or_else(|| OutputTemplateError::Malformed {
            template: text.to_owned(),
            source,
        })),
    }
}

pub fn extension(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Text | OutputFormat::Txt => "txt",
        OutputFormat::Json => "json",
        OutputFormat::Srt => "srt",
    }
}

fn value(
    placeholder: Placeholder,
    item: &InputItem,
    format: OutputFormat,
) -> Result<String, OutputTemplateError> {
    match placeholder {
        Placeholder::Ext => Ok(extension(format).to_owned()),
        Placeholder::Index => Ok(item.index.to_string()),
        Placeholder::Stem => named(placeholder, item, Path::file_stem),
        Placeholder::Name => named(placeholder, item, |name| Some(name.as_os_str())),
        Placeholder::Dir => directory(item),
    }
}

fn named(
    placeholder: Placeholder,
    item: &InputItem,
    pick: impl Fn(&Path) -> Option<&std::ffi::OsStr>,
) -> Result<String, OutputTemplateError> {
    let unnamed = || OutputTemplateError::NameNeeded {
        placeholder: placeholder.key(),
        input: item.display.clone(),
    };
    match &item.source {
        InputSource::File(path) => {
            let file_name = path.file_name().ok_or_else(unnamed)?;
            let picked = pick(Path::new(file_name)).ok_or_else(unnamed)?;
            picked
                .to_str()
                .map(str::to_owned)
                .ok_or_else(|| OutputTemplateError::NonUnicodePath { path: path.clone() })
        }
        InputSource::Url(url) => {
            let name = url_file_name(url).ok_or_else(unnamed)?;
            let picked = pick(Path::new(&name)).ok_or_else(unnamed)?;
            picked.to_str().map(str::to_owned).ok_or_else(unnamed)
        }
    }
}

fn directory(item: &InputItem) -> Result<String, OutputTemplateError> {
    match &item.source {
        InputSource::File(path) => match path.parent() {
            Some(parent) if parent.as_os_str().is_empty() => Ok(".".to_owned()),
            Some(parent) => parent
                .to_str()
                .map(str::to_owned)
                .ok_or_else(|| OutputTemplateError::NonUnicodePath { path: path.clone() }),
            None => Err(OutputTemplateError::NameNeeded {
                placeholder: Placeholder::Dir.key(),
                input: item.display.clone(),
            }),
        },
        InputSource::Url(_) => Err(OutputTemplateError::NameNeeded {
            placeholder: Placeholder::Dir.key(),
            input: item.display.clone(),
        }),
    }
}

fn url_file_name(url: &Url) -> Option<String> {
    let segment = url.path_segments()?.next_back()?;
    let decoded = percent_decode_str(segment).decode_utf8().ok()?;
    let mut components = Path::new(decoded.as_ref()).components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(name)), None) => name.to_str().map(str::to_owned),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn file_item(index: usize, path: &str) -> InputItem {
        InputItem {
            index,
            source: InputSource::File(PathBuf::from(path)),
            display: path.to_owned(),
        }
    }

    fn url_item(index: usize, url: &str) -> InputItem {
        InputItem {
            index,
            source: InputSource::Url(Url::parse(url).unwrap()),
            display: url.to_owned(),
        }
    }

    fn render(template: &str, item: &InputItem, format: OutputFormat) -> PathBuf {
        OutputTemplate::parse(template)
            .unwrap()
            .render(item, format)
            .unwrap()
    }

    #[test]
    fn stem_keeps_inner_dots() {
        let path = render("{stem}", &file_item(1, "a.b.wav"), OutputFormat::Text);
        assert_eq!(path, PathBuf::from("a.b"));
    }

    #[cfg(windows)]
    #[test]
    fn name_dir_index_ext_render() {
        let item = file_item(3, "D:\\x\\a.wav");
        assert_eq!(
            render("{dir}\\{stem}.{ext}", &item, OutputFormat::Srt),
            PathBuf::from("D:\\x\\a.srt")
        );
        assert_eq!(
            render("{name}", &item, OutputFormat::Srt),
            PathBuf::from("a.wav")
        );
        assert_eq!(
            render("{index}", &item, OutputFormat::Srt),
            PathBuf::from("3")
        );
    }

    #[test]
    fn name_and_index_render() {
        let item = file_item(3, "a.wav");
        assert_eq!(
            render("{name}", &item, OutputFormat::Srt),
            PathBuf::from("a.wav")
        );
        assert_eq!(
            render("{index}", &item, OutputFormat::Srt),
            PathBuf::from("3")
        );
    }

    #[test]
    fn dir_of_a_bare_file_name_is_the_current_directory() {
        let path = render(
            "{dir}/{stem}.{ext}",
            &file_item(1, "a.wav"),
            OutputFormat::Json,
        );
        assert_eq!(path, PathBuf::from("./a.json"));
    }

    #[test]
    fn ext_follows_format() {
        let item = file_item(1, "a.wav");
        let extensions: Vec<PathBuf> = [
            OutputFormat::Text,
            OutputFormat::Txt,
            OutputFormat::Json,
            OutputFormat::Srt,
        ]
        .into_iter()
        .map(|format| render("{ext}", &item, format))
        .collect();
        assert_eq!(
            extensions,
            vec![
                PathBuf::from("txt"),
                PathBuf::from("txt"),
                PathBuf::from("json"),
                PathBuf::from("srt"),
            ]
        );
    }

    #[test]
    fn unknown_placeholder_is_an_error() {
        let error = OutputTemplate::parse("{title}.txt").unwrap_err();
        assert!(
            matches!(
                &error,
                OutputTemplateError::UnknownPlaceholder { name, allowed }
                    if name == "title" && *allowed == ALLOWED
            ),
            "{error:?}"
        );
    }

    #[test]
    fn unbalanced_braces_are_an_error() {
        let error = OutputTemplate::parse("{stem.txt").unwrap_err();
        assert!(
            matches!(error, OutputTemplateError::Malformed { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn url_without_a_name_needs_index() {
        let item = url_item(2, "https://h/");
        let error = OutputTemplate::parse("{stem}.txt")
            .unwrap()
            .render(&item, OutputFormat::Text)
            .unwrap_err();
        assert!(
            matches!(
                &error,
                OutputTemplateError::NameNeeded { placeholder, .. } if *placeholder == "stem"
            ),
            "{error:?}"
        );
        assert_eq!(
            render("{index}.txt", &item, OutputFormat::Text),
            PathBuf::from("2.txt")
        );
    }

    #[test]
    fn url_with_a_name_uses_its_stem() {
        let item = url_item(1, "https://h/shows/morning%20show.v2.mp3?token=1");
        assert_eq!(
            render("{stem}.{ext}", &item, OutputFormat::Srt),
            PathBuf::from("morning show.v2.srt")
        );
        assert_eq!(
            render("{name}", &item, OutputFormat::Srt),
            PathBuf::from("morning show.v2.mp3")
        );
    }

    #[test]
    fn url_name_with_an_encoded_separator_is_unusable() {
        let item = url_item(1, "https://h/a%2F..%2Fb.mp3");
        let error = OutputTemplate::parse("{stem}")
            .unwrap()
            .render(&item, OutputFormat::Text)
            .unwrap_err();
        assert!(
            matches!(error, OutputTemplateError::NameNeeded { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn dir_of_a_url_is_an_error() {
        let error = OutputTemplate::parse("{dir}")
            .unwrap()
            .render(&url_item(1, "https://h/a.mp3"), OutputFormat::Text)
            .unwrap_err();
        assert!(
            matches!(error, OutputTemplateError::NameNeeded { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn no_placeholders_is_literal() {
        let template = OutputTemplate::parse("out/all.txt").unwrap();
        assert!(!template.has_placeholders());
        assert_eq!(template.literal(), Some(Path::new("out/all.txt")));
        assert_eq!(
            template
                .render(&file_item(1, "a.wav"), OutputFormat::Text)
                .unwrap(),
            PathBuf::from("out/all.txt")
        );
    }

    #[test]
    fn in_directory_renders_stem_and_ext_inside_it() {
        let template = OutputTemplate::in_directory(PathBuf::from("out {x}"));
        assert!(template.has_placeholders());
        assert_eq!(template.literal(), None);
        assert_eq!(
            template
                .render(&file_item(1, "sub/a.b.wav"), OutputFormat::Srt)
                .unwrap(),
            PathBuf::from("out {x}").join("a.b.srt")
        );
    }

    #[test]
    fn in_directory_needs_a_name() {
        let error = OutputTemplate::in_directory(PathBuf::from("out"))
            .render(&url_item(1, "https://h/"), OutputFormat::Text)
            .unwrap_err();
        assert!(
            matches!(error, OutputTemplateError::NameNeeded { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn placeholders_are_detected() {
        let template = OutputTemplate::parse("{stem}.txt").unwrap();
        assert!(template.has_placeholders());
        assert_eq!(template.literal(), None);
    }

    #[test]
    fn doubled_braces_are_literal() {
        let template = OutputTemplate::parse("{{x}}.txt").unwrap();
        assert!(!template.has_placeholders());
        assert_eq!(template.literal(), Some(Path::new("{x}.txt")));
    }

    #[test]
    fn non_ascii_and_space_file_names_render_unchanged() {
        let item = file_item(1, "my caf\u{e9} \u{65e5}\u{672c}.wav");
        assert_eq!(
            render("{stem}.{ext}", &item, OutputFormat::Text),
            PathBuf::from("my caf\u{e9} \u{65e5}\u{672c}.txt")
        );
    }
}
