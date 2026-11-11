use crate::{components::toast::Toasts, settings_store::status::UpdateError};

/// Tells the user when the store refused a change; a change that was accepted needs no word.
pub fn report(toasts: Toasts, result: Result<(), UpdateError>) {
    if let Err(error) = result {
        toasts.error("That setting can't be used", error.to_string());
    }
}
