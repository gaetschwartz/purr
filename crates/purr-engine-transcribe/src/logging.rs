use core::ffi::{c_char, c_void};
use std::ffi::CStr;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing::Level;
use transcribe_cpp::sys;

pub const LOG_TARGET: &str = "purr_engine_transcribe::native";

static WANTED: AtomicBool = AtomicBool::new(false);
static INSTALL: Once = Once::new();

pub(crate) fn request() {
    WANTED.store(true, Ordering::Release);
}

/// Routes transcribe.cpp's log output into `tracing` if the engine was asked to. Calls the
/// library, so it only runs once the library is loaded.
pub(crate) fn install_if_requested() {
    if WANTED.load(Ordering::Acquire) {
        INSTALL.call_once(|| unsafe {
            sys::transcribe_log_set(Some(trampoline), std::ptr::null_mut());
        });
    }
}

const fn level_of(level: sys::transcribe_log_level) -> Option<Level> {
    match level {
        sys::transcribe_log_level::TRANSCRIBE_LOG_LEVEL_ERROR => Some(Level::ERROR),
        sys::transcribe_log_level::TRANSCRIBE_LOG_LEVEL_WARN => Some(Level::WARN),
        sys::transcribe_log_level::TRANSCRIBE_LOG_LEVEL_INFO
        | sys::transcribe_log_level::TRANSCRIBE_LOG_LEVEL_CONT => Some(Level::INFO),
        sys::transcribe_log_level::TRANSCRIBE_LOG_LEVEL_DEBUG => Some(Level::DEBUG),
        _ => None,
    }
}

fn emit(level: Level, text: &str) {
    match level {
        Level::ERROR => tracing::error!(target: LOG_TARGET, "{text}"),
        Level::WARN => tracing::warn!(target: LOG_TARGET, "{text}"),
        Level::INFO => tracing::info!(target: LOG_TARGET, "{text}"),
        _ => tracing::debug!(target: LOG_TARGET, "{text}"),
    }
}

unsafe extern "C" fn trampoline(
    level: sys::transcribe_log_level,
    text: *const c_char,
    _: *mut c_void,
) {
    if text.is_null() {
        return;
    }
    let Some(level) = level_of(level) else {
        return;
    };
    // SAFETY: transcribe.cpp passes a NUL-terminated string that outlives the call.
    let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    emit(level, text.trim_end_matches('\n'));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_map_onto_tracing_levels() {
        assert_eq!(
            level_of(sys::transcribe_log_level::TRANSCRIBE_LOG_LEVEL_ERROR),
            Some(Level::ERROR)
        );
        assert_eq!(
            level_of(sys::transcribe_log_level::TRANSCRIBE_LOG_LEVEL_CONT),
            Some(Level::INFO)
        );
    }

    #[test]
    fn a_message_without_a_level_is_dropped() {
        assert_eq!(
            level_of(sys::transcribe_log_level::TRANSCRIBE_LOG_LEVEL_NONE),
            None
        );
    }
}
