use serde_with::{DeserializeFromStr, SerializeDisplay};
use strum::{Display, EnumString};

/// Languages a model can transcribe.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Display,
    EnumString,
    SerializeDisplay,
    DeserializeFromStr,
)]
pub enum ModelLanguage {
    #[strum(serialize = "multilingual")]
    Multilingual,
    #[strum(serialize = "english-only")]
    EnglishOnly,
}
