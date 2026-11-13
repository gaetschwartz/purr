//! Tells the user when a saved transcription could not keep its audio and they can do something.

use crate::components::toast::Toasts;
use purr_common::history::audio::{AudioDropReason, AudioOutcome};

/// Warns when the browser had no room for the audio; every other outcome is already logged by
/// the platform and needs nothing from the user.
pub fn announce(toasts: Toasts, outcome: &AudioOutcome) {
    match outcome {
        AudioOutcome::Dropped(AudioDropReason::StorageFull) => {
            toasts.warning(
                "Couldn't keep the audio",
                "The browser has no room left for it. Free up space, or clear the saved audio in Settings.",
            );
        }
        AudioOutcome::NotOffered
        | AudioOutcome::Kept
        | AudioOutcome::Dropped(AudioDropReason::OverBudget | AudioDropReason::Failed { .. }) => {}
    }
}
