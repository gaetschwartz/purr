use super::{
    model::DeveloperModel,
    session_options::{SessionOptions, SessionOptionsError, parse},
};
use serde::{Deserialize, Serialize};

/// The settings of developer mode.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeveloperSettings {
    pub enabled: bool,
    /// ONNX Runtime session options as the JSON text the user typed, merged into the browser's own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_options: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<DeveloperModel>,
}

impl DeveloperSettings {
    /// The session options to apply; none while developer mode is off.
    pub fn active_session_options(&self) -> Result<SessionOptions, SessionOptionsError> {
        if !self.enabled {
            return Ok(SessionOptions::new());
        }
        parse(self.session_options.as_deref().unwrap_or_default())
    }

    /// The models to offer; none while developer mode is off.
    #[must_use]
    pub fn active_models(&self) -> &[DeveloperModel] {
        if self.enabled { &self.models } else { &[] }
    }

    #[must_use]
    pub fn model(&self, id: &str) -> Option<&DeveloperModel> {
        self.active_models()
            .iter()
            .find(|model| model.id.as_str() == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::Settings;

    #[test]
    fn settings_written_before_developer_mode_read_as_off() {
        let mut value = serde_json::to_value(Settings::default()).unwrap();
        let removed = value.as_object_mut().unwrap().remove("developer");
        assert!(removed.is_some());
        let settings: Settings = serde_json::from_value(value).unwrap();
        assert!(!settings.developer.enabled);
        assert_eq!(settings.developer, DeveloperSettings::default());
    }

    #[test]
    fn the_options_apply_only_while_the_mode_is_on() {
        let mut settings = DeveloperSettings {
            session_options: Some(r#"{"logSeverityLevel":2}"#.to_owned()),
            ..DeveloperSettings::default()
        };
        assert_eq!(settings.active_session_options(), Ok(SessionOptions::new()));
        settings.enabled = true;
        assert_eq!(settings.active_session_options().unwrap().len(), 1);
    }

    #[test]
    fn settings_survive_a_toml_file() {
        let settings = Settings {
            developer: DeveloperSettings {
                enabled: true,
                session_options: Some("{\"a\": [1, null]}".to_owned()),
                models: Vec::new(),
            },
            ..Settings::default()
        };
        let text = toml::to_string(&settings).unwrap();
        assert_eq!(toml::from_str::<Settings>(&text).unwrap(), settings);
    }
}
