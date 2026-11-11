//! Deciding whether settings read from storage can be used.

use super::status::{LoadFailure, UpdateError};
use purr_common::{platform::PlatformError, settings::Settings};

/// The settings to run on, or why the stored ones must be left alone.
pub fn accept_loaded(loaded: Result<Settings, PlatformError>) -> Result<Settings, LoadFailure> {
    let mut settings = loaded.map_err(|error| LoadFailure::Unreadable(error.to_string()))?;
    let stored_version = settings.version;
    settings
        .migrate_from_version(stored_version)
        .map_err(LoadFailure::Unsupported)?;
    let errors = settings.validate();
    if errors.is_empty() {
        Ok(settings)
    } else {
        Err(LoadFailure::Invalid(errors))
    }
}

/// `current` changed by `change`, if the result is valid.
pub fn apply_change(
    current: &Settings,
    change: impl FnOnce(&mut Settings),
) -> Result<Settings, UpdateError> {
    let mut next = current.clone();
    change(&mut next);
    validated(next)
}

pub fn validated(settings: Settings) -> Result<Settings, UpdateError> {
    let errors = settings.validate();
    if errors.is_empty() {
        Ok(settings)
    } else {
        Err(UpdateError::Invalid(errors))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::settings::{CURRENT_VERSION, SettingsError};

    #[test]
    fn stored_settings_are_used_as_they_are() {
        let mut stored = Settings::default();
        "small".clone_into(&mut stored.model.model_name);
        assert_eq!(accept_loaded(Ok(stored.clone())), Ok(stored));
    }

    #[test]
    fn unreadable_storage_is_a_failure_not_the_defaults() {
        let failure = accept_loaded(Err(PlatformError::settings_persistence("bad json")));
        assert_eq!(
            failure,
            Err(LoadFailure::Unreadable(
                "Settings persistence error: bad json".to_owned()
            ))
        );
    }

    #[test]
    fn settings_from_a_newer_purr_are_left_alone() {
        let stored = Settings {
            version: 7,
            ..Settings::default()
        };
        assert_eq!(
            accept_loaded(Ok(stored)),
            Err(LoadFailure::Unsupported(
                SettingsError::UnsupportedVersion {
                    current: CURRENT_VERSION,
                    target: 7
                }
            ))
        );
    }

    #[test]
    fn version_zero_is_migrated() {
        let stored = Settings {
            version: 0,
            ..Settings::default()
        };
        assert_eq!(accept_loaded(Ok(stored)), Ok(Settings::default()));
    }

    #[test]
    fn version_one_is_migrated_and_keeps_what_the_user_chose() {
        let mut stored = Settings {
            version: 1,
            ..Settings::default()
        };
        "small".clone_into(&mut stored.model.model_name);
        stored.performance.gpu_acceleration = false;
        let mut expected = Settings::default();
        "small".clone_into(&mut expected.model.model_name);
        assert_eq!(accept_loaded(Ok(stored)), Ok(expected));
    }

    #[test]
    fn invalid_stored_values_are_a_failure() {
        let mut stored = Settings::default();
        stored.performance.max_concurrent_jobs = 0;
        assert_eq!(
            accept_loaded(Ok(stored)),
            Err(LoadFailure::Invalid(vec![
                SettingsError::InvalidConcurrentJobs { value: 0 }
            ]))
        );
    }

    #[test]
    fn an_invalid_change_is_refused() {
        let current = Settings::default();
        assert_eq!(
            apply_change(&current, |settings| settings.model.model_name.clear()),
            Err(UpdateError::Invalid(vec![SettingsError::EmptyModelName]))
        );
        let changed = apply_change(&current, |settings| settings.model.max_tokens = 512);
        assert_eq!(changed.map(|settings| settings.model.max_tokens), Ok(512));
    }
}
