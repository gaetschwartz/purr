//! Asks the browser to confirm before the page is closed or reloaded while work would be lost.

use crate::{
    platform::IS_WEB,
    transcribe::{run::RunOutcome, session::use_transcribe_session},
};
use dioxus::prelude::*;

const ARM: &str = "window.__purrLeaveGuard ??= (event) => { event.preventDefault(); event.returnValue = ''; }; window.addEventListener('beforeunload', window.__purrLeaveGuard);";
const DISARM: &str = "if (window.__purrLeaveGuard) window.removeEventListener('beforeunload', window.__purrLeaveGuard);";

/// While a transcription runs, closing or reloading the browser tab asks first, whatever screen
/// is showing; call once at the root.
pub fn use_leave_guard_during_runs() {
    let session = use_transcribe_session();
    use_effect(move || {
        let running = session
            .record()
            .is_some_and(|record| record.run.outcome == RunOutcome::Running);
        if IS_WEB {
            document::eval(if running { ARM } else { DISARM });
        }
    });
}
