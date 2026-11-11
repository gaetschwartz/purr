//! What the store reports about saving and loading.

use purr_common::settings::SettingsError;

/// Where the latest change stands on its way to storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaveStatus {
    /// Nothing changed since the settings were loaded.
    Idle,
    /// A change is being written or waits for the write in flight.
    Saving,
    /// Every change is stored; the last write ended at `at_millis` (Unix milliseconds).
    Saved { at_millis: u64 },
    /// The last write failed; the change stays in memory and is retried on the next change.
    Failed { error: String },
    /// The stored settings could not be read, so changes stay in memory until a reset.
    Paused,
}

/// Whether the stored settings are in memory yet.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadState {
    Loading,
    Loaded,
    /// The app runs on the defaults and leaves the stored settings untouched until a reset.
    Failed(LoadFailure),
}

/// Why the stored settings were not used.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LoadFailure {
    #[error("{0}")]
    Unreadable(String),
    #[error("{}", join(.0))]
    Invalid(Vec<SettingsError>),
    #[error("{0}")]
    Unsupported(SettingsError),
}

/// A change the store refused; nothing was applied or saved.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum UpdateError {
    #[error("{}", join(.0))]
    Invalid(Vec<SettingsError>),
}

fn join(errors: &[SettingsError]) -> String {
    errors
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}
