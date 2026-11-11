//! Asks the browser to confirm before the page is closed or reloaded while work would be lost.

use crate::platform::IS_WEB;
use dioxus::prelude::*;

const ARM: &str = "window.__purrLeaveGuard ??= (event) => { event.preventDefault(); event.returnValue = ''; }; window.addEventListener('beforeunload', window.__purrLeaveGuard);";
const DISARM: &str = "if (window.__purrLeaveGuard) window.removeEventListener('beforeunload', window.__purrLeaveGuard);";

/// While the calling component is mounted, closing or reloading the browser tab asks first.
pub fn use_leave_guard() {
    use_hook(|| {
        if IS_WEB {
            document::eval(ARM);
        }
    });
    use_drop(|| {
        if IS_WEB {
            document::eval(DISARM);
        }
    });
}
