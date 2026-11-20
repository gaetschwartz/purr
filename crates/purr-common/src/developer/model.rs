use crate::{
    model::{family::ModelFamily, format::ModelFormat},
    web_model::{attributes::OnnxDtype, variant::TableStorage},
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The id of a model loaded outside the catalog; it never collides with a catalog id.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DeveloperModelId(String);

impl DeveloperModelId {
    const PREFIX: &'static str = "dev-";

    /// An id made of the lowercase hex digits of `unique`.
    #[must_use]
    pub fn from_unique(unique: &str) -> Self {
        let digits: String = unique
            .chars()
            .filter(char::is_ascii_hexdigit)
            .map(|digit| digit.to_ascii_lowercase())
            .take(12)
            .collect();
        Self(format!("{}{digits}", Self::PREFIX))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether `id` names a developer model.
    #[must_use]
    pub fn is_developer(id: &str) -> bool {
        id.starts_with(Self::PREFIX)
    }
}

/// Where a browser model's files come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrowserOrigin {
    /// A Hugging Face repository at a branch, tag or commit.
    HuggingFace { repo: String, revision: String },
    /// Files the user picked from their disk; the browser keeps its own copy.
    LocalFiles,
}

/// What the browser does to the ONNX files before it loads them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Conversion {
    /// The streaming converter: external weights and the tied embedding fix, with the table
    /// stored as `table`.
    Streaming { table: TableStorage },
    /// The files exactly as they were given.
    AsIs,
}

impl Conversion {
    /// How the converter stores the tied table; none when the files are used as they are.
    #[must_use]
    pub const fn table(&self) -> Option<TableStorage> {
        match self {
            Self::Streaming { table } => Some(*table),
            Self::AsIs => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrowserModel {
    pub origin: BrowserOrigin,
    pub encoder_dtype: OnnxDtype,
    pub decoder_dtype: OnnxDtype,
    pub conversion: Conversion,
    pub english_only: bool,
}

/// A model file the desktop engines load from where it is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeModel {
    pub path: PathBuf,
    pub format: ModelFormat,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeveloperSource {
    Browser(BrowserModel),
    Native(NativeModel),
}

/// A model the user loads without the catalog: no manifest entry and no hash checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeveloperModel {
    pub id: DeveloperModelId,
    pub name: String,
    /// How the picker groups it.
    pub family: ModelFamily,
    pub source: DeveloperSource,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::family::WhisperSize;

    #[test]
    fn an_id_keeps_only_hex_digits_and_says_it_is_a_developer_model() {
        let id = DeveloperModelId::from_unique("A1b2-c3d4-ZZ-e5f6-0123456789");
        assert_eq!(id.as_str(), "dev-a1b2c3d4e5f6");
        assert!(DeveloperModelId::is_developer(id.as_str()));
        assert!(!DeveloperModelId::is_developer("base"));
    }

    #[test]
    fn a_model_survives_its_settings_file() {
        let model = DeveloperModel {
            id: DeveloperModelId::from_unique("0123456789ab"),
            name: "tiny".to_owned(),
            family: ModelFamily::Whisper(WhisperSize::Tiny),
            source: DeveloperSource::Browser(BrowserModel {
                origin: BrowserOrigin::HuggingFace {
                    repo: "onnx-community/whisper-tiny".to_owned(),
                    revision: "main".to_owned(),
                },
                encoder_dtype: OnnxDtype::Fp32,
                decoder_dtype: OnnxDtype::Q4,
                conversion: Conversion::Streaming {
                    table: TableStorage::Fp16,
                },
                english_only: false,
            }),
        };
        let json = serde_json::to_string(&model).unwrap();
        assert_eq!(
            serde_json::from_str::<DeveloperModel>(&json).unwrap(),
            model
        );
    }
}
