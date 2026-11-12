//! The user's Hugging Face access token.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// A Hugging Face access token, trimmed and never empty.
///
/// It prints as a placeholder everywhere; read the value with [`HuggingFaceToken::expose`] only
/// to build the `Authorization` header of a request to huggingface.co.
#[derive(Clone, PartialEq, Eq)]
pub struct HuggingFaceToken(String);

impl HuggingFaceToken {
    /// The token typed in `input`, or `None` when it is empty or only whitespace.
    #[must_use]
    pub fn new(input: &str) -> Option<Self> {
        let trimmed = input.trim();
        (!trimmed.is_empty()).then(|| Self(trimmed.to_owned()))
    }

    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// Whether the token can travel in an HTTP header value.
    #[must_use]
    pub fn is_header_safe(&self) -> bool {
        self.0.bytes().all(|byte| byte.is_ascii_graphic())
    }

    /// For `deserialize_with`: a missing, null, empty or blank value is no token.
    pub fn deserialize_optional<'de, D>(deserializer: D) -> Result<Option<Self>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Option::<String>::deserialize(deserializer)?.and_then(|raw| Self::new(&raw)))
    }
}

impl fmt::Debug for HuggingFaceToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("HuggingFaceToken(<redacted>)")
    }
}

impl Serialize for HuggingFaceToken {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Holder {
        #[serde(default, deserialize_with = "HuggingFaceToken::deserialize_optional")]
        token: Option<HuggingFaceToken>,
    }

    #[test]
    fn surrounding_whitespace_is_trimmed() {
        let token = HuggingFaceToken::new("  hf_abc \n");
        assert_eq!(token.as_ref().map(HuggingFaceToken::expose), Some("hf_abc"));
    }

    #[test]
    fn an_empty_or_blank_input_is_no_token() {
        assert_eq!(HuggingFaceToken::new(""), None);
        assert_eq!(HuggingFaceToken::new(" \t\r\n "), None);
    }

    #[test]
    fn the_debug_form_hides_the_value() {
        let token = HuggingFaceToken::new("hf_secret");
        assert_eq!(
            format!("{token:?}"),
            "Some(HuggingFaceToken(<redacted>))".to_owned()
        );
    }

    #[test]
    fn only_visible_ascii_fits_in_a_header() {
        let safe = |input| HuggingFaceToken::new(input).map(|token| token.is_header_safe());
        assert_eq!(safe("hf_AbC-123"), Some(true));
        assert_eq!(safe("hf abc"), Some(false));
        assert_eq!(safe("hf_\u{e9}"), Some(false));
        assert_eq!(safe("hf_\u{7}"), Some(false));
    }

    #[test]
    fn a_token_survives_json_and_toml() {
        let holder = Holder {
            token: HuggingFaceToken::new("hf_abc"),
        };
        let json = serde_json::to_string(&holder).unwrap();
        assert_eq!(json, r#"{"token":"hf_abc"}"#);
        assert_eq!(serde_json::from_str::<Holder>(&json).unwrap(), holder);
        let toml = toml::to_string(&holder).unwrap();
        assert_eq!(toml::from_str::<Holder>(&toml).unwrap(), holder);
    }

    #[test]
    fn a_missing_null_or_blank_value_loads_as_no_token() {
        let none = Holder { token: None };
        assert_eq!(serde_json::from_str::<Holder>("{}").unwrap(), none);
        assert_eq!(
            serde_json::from_str::<Holder>(r#"{"token":null}"#).unwrap(),
            none
        );
        assert_eq!(
            serde_json::from_str::<Holder>(r#"{"token":"   "}"#).unwrap(),
            none
        );
        assert_eq!(toml::from_str::<Holder>("token = \"\"").unwrap(), none);
    }

    #[test]
    fn a_stored_token_is_trimmed_on_load() {
        let loaded: Holder = toml::from_str("token = \" hf_abc \"").unwrap();
        assert_eq!(loaded.token, HuggingFaceToken::new("hf_abc"));
    }

    #[test]
    fn no_token_writes_nothing() {
        assert_eq!(toml::to_string(&Holder { token: None }).unwrap(), "");
    }
}
