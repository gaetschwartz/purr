use crate::settings_store::{status::LoadState, use_settings};
use dioxus::prelude::*;
use purr_common::language::choice::LanguageChoice;

/// Starts the session with the default model and language, once the settings have loaded.
pub fn use_launch_defaults() {
    let settings = use_settings();
    let mut applied = use_signal(|| false);
    use_effect(move || {
        if settings.load_state() != LoadState::Loaded || *applied.peek() {
            return;
        }
        applied.set(true);
        let (model, default_model, language, default_language) = settings.with(|settings| {
            (
                settings.model.model_name.clone(),
                settings.model.default_model.clone(),
                settings.ui.language.clone(),
                settings.ui.default_language.clone(),
            )
        });
        if let Some(default_model) = default_model.filter(|default| *default != model)
            && let Err(error) = settings.set_model(default_model)
        {
            tracing::warn!("Could not start with the default model: {error}");
        }
        let language = LanguageChoice::from_setting(&language);
        if let Some(default_language) = default_language
            .map(|default| LanguageChoice::from_setting(&default))
            .filter(|default| *default != language)
            && let Err(error) = settings.set_language(default_language.setting())
        {
            tracing::warn!("Could not start with the default language: {error}");
        }
    });
}
