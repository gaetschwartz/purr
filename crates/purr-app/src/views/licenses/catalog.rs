//! The third-party licences the app is built from, as generated into `assets/licenses.json`.

use serde::Deserialize;

const GENERATED: &str = include_str!("../../../assets/licenses.json");

/// A crate that is distributed under a licence.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(from = "(String, String)")]
pub struct Package {
    pub name: String,
    pub version: String,
}

impl From<(String, String)> for Package {
    fn from((name, version): (String, String)) -> Self {
        Self { name, version }
    }
}

/// One licence text, with its copyright notice, and the crates it applies to.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Notice {
    pub text: String,
    pub crates: Vec<Package>,
}

impl Notice {
    /// The crates whose name contains `query`, ignoring case; all of them when `query` is blank.
    pub fn matching(&self, query: &str) -> Vec<&Package> {
        let query = query.trim().to_lowercase();
        self.crates
            .iter()
            .filter(|package| package.name.to_lowercase().contains(&query))
            .collect()
    }
}

/// A licence and the distinct texts crates ship it with.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Licence {
    pub id: String,
    pub name: String,
    pub texts: Vec<Notice>,
}

impl Licence {
    /// The number of crates released under this licence.
    pub fn crate_count(&self) -> usize {
        self.texts.iter().map(|notice| notice.crates.len()).sum()
    }
}

/// Reads the licences generated for this build.
pub fn generated() -> Result<Vec<Licence>, serde_json::Error> {
    parse(GENERATED)
}

fn parse(json: &str) -> Result<Vec<Licence>, serde_json::Error> {
    serde_json::from_str(json)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"[
        {"id":"MIT","name":"MIT License","texts":[
            {"text":"Permission is hereby granted","crates":[["serde","1.0.2"],["Tokio","1.4.0"]]},
            {"text":"Copyright someone else","crates":[["adler32","1.2.0"]]}]}
    ]"#;

    fn package(name: &str, version: &str) -> Package {
        Package {
            name: name.to_owned(),
            version: version.to_owned(),
        }
    }

    #[test]
    fn parses_licences_with_their_texts_and_crates() {
        assert_eq!(
            parse(FIXTURE).unwrap(),
            vec![Licence {
                id: "MIT".to_owned(),
                name: "MIT License".to_owned(),
                texts: vec![
                    Notice {
                        text: "Permission is hereby granted".to_owned(),
                        crates: vec![package("serde", "1.0.2"), package("Tokio", "1.4.0")],
                    },
                    Notice {
                        text: "Copyright someone else".to_owned(),
                        crates: vec![package("adler32", "1.2.0")],
                    },
                ],
            }]
        );
    }

    #[test]
    fn counts_the_crates_of_every_text() {
        assert_eq!(parse(FIXTURE).unwrap()[0].crate_count(), 3);
    }

    #[test]
    fn filters_crates_by_name_ignoring_case() {
        let licence = parse(FIXTURE).unwrap().remove(0);
        assert_eq!(
            licence.texts[0].matching(" TOK "),
            vec![&package("Tokio", "1.4.0")]
        );
        assert_eq!(licence.texts[0].matching("").len(), 2);
        assert_eq!(licence.texts[1].matching("serde"), Vec::<&Package>::new());
    }

    #[test]
    fn the_generated_file_is_valid() {
        assert_eq!(generated().is_ok(), true);
    }
}
