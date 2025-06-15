use core::ffi::{c_char, c_void};
use std::borrow::Cow;
use std::ffi::CStr;
use std::sync::Once;
use whisper_rs::whisper_rs_sys::{self, ggml_log_level};
use whisper_rs::GGMLLogLevel;

/// Redirect all whisper.cpp and GGML logs to logging hooks installed by whisper-rs.
///
/// This will stop most logs from being output to stdout/stderr and will bring them into
/// `log` or `tracing`, if the `log_backend` or `tracing_backend` features, respectively,
/// are enabled. If neither is enabled, this will essentially disable logging, as they won't
/// be output anywhere.
///
/// Note whisper.cpp and GGML do not reliably follow Rust logging conventions.
/// Use your logging crate's configuration to control how these logs will be output.
/// whisper-rs does not currently output any logs, but this may change in the future.
/// You should configure by module path and use `whisper_rs::ggml_logging_hook`,
/// and/or `whisper_rs::whisper_logging_hook`, to avoid possibly ignoring useful
/// `whisper-rs` logs in the future.
///
/// Safe to call multiple times. Only has an effect the first time.
/// (note this means installing your own logging handlers with unsafe functions after this call
/// is permanent and cannot be undone)
pub fn install_logging_hooks() {
    install_whisper_logging_hook();
    install_ggml_logging_hook();
}

macro_rules! generic_error {
    ($($expr:tt)*) => {
        tracing::error!($($expr)*);
    };
}

macro_rules! generic_warn {
    ($($expr:tt)*) => {
        tracing::warn!($($expr)*);
    }
}

macro_rules! generic_info {
    ($($expr:tt)*) => {
        tracing::info!($($expr)*);
    }
}

macro_rules! generic_debug {
    ($($expr:tt)*) => {
        tracing::debug!($($expr)*);
    }
}

macro_rules! generic_trace {
    ($($expr:tt)*) => {
        tracing::trace!($($expr)*);
    }
}

static WHISPER_LOG_TRAMPOLINE_INSTALL: Once = Once::new();
pub(crate) fn install_whisper_logging_hook() {
    WHISPER_LOG_TRAMPOLINE_INSTALL.call_once(|| unsafe {
        whisper_rs_sys::whisper_log_set(Some(whisper_logging_trampoline), std::ptr::null_mut())
    });
}

unsafe extern "C" fn whisper_logging_trampoline(
    level: ggml_log_level,
    text: *const c_char,
    _: *mut c_void, // user_data
) {
    if text.is_null() {
        generic_error!("whisper_logging_trampoline: text is nullptr");
    }
    let level = GGMLLogLevel::from(level);

    // SAFETY: we must trust whisper.cpp that it will not pass us a string that does not satisfy
    // from_ptr's requirements.
    let log_str = unsafe { CStr::from_ptr(text) }.to_string_lossy();

    whisper_logging_trampoline_safe(level, log_str)
}

// this code essentially compiles down to a noop if neither feature is enabled
fn whisper_logging_trampoline_safe(level: GGMLLogLevel, text: Cow<str>) {
    match level {
        GGMLLogLevel::None => {
            // no clue what to do here, trace it?
            generic_trace!("{}", text.trim());
        }
        GGMLLogLevel::Info => {
            generic_info!("{}", text.trim());
        }
        GGMLLogLevel::Warn => {
            generic_warn!("{}", text.trim());
        }
        GGMLLogLevel::Error => {
            generic_error!("{}", text.trim());
        }
        GGMLLogLevel::Debug => {
            generic_debug!("{}", text.trim());
        }
        GGMLLogLevel::Cont => {
            // this means continue previous log
            // storing state to do this is a massive pain so it's just a lot easier to not
            // plus as far as i can tell it's not actually *used* anywhere
            // whisper splits at 1024 chars and doesn't actually change the kind
            // so technically this is unused
            generic_trace!("{}", text.trim());
        }
        GGMLLogLevel::Unknown(level) => {
            generic_warn!(
                "whisper_logging_trampoline: unknown log level {}: message: {}",
                level,
                text.trim()
            );
        }
    }
}

static GGML_LOG_TRAMPOLINE_INSTALL: Once = Once::new();
pub(crate) fn install_ggml_logging_hook() {
    GGML_LOG_TRAMPOLINE_INSTALL.call_once(|| unsafe {
        whisper_rs_sys::ggml_log_set(Some(ggml_logging_trampoline), std::ptr::null_mut())
    });
}

unsafe extern "C" fn ggml_logging_trampoline(
    level: ggml_log_level,
    text: *const c_char,
    _: *mut c_void, // user_data
) {
    if text.is_null() {
        generic_error!("ggml_logging_trampoline: text is nullptr");
    }
    let level = GGMLLogLevel::from(level);

    // SAFETY: we must trust ggml that it will not pass us a string that does not satisfy
    // from_ptr's requirements.
    let log_str = unsafe { CStr::from_ptr(text) }.to_string_lossy();

    ggml_logging_trampoline_safe(level, log_str)
}

// this code essentially compiles down to a noop if neither feature is enabled
fn ggml_logging_trampoline_safe(level: GGMLLogLevel, text: Cow<str>) {
    match level {
        GGMLLogLevel::None => {
            // no clue what to do here, trace it?
            generic_trace!("{}", text.trim());
        }
        GGMLLogLevel::Info => {
            generic_info!("{}", text.trim());
        }
        GGMLLogLevel::Warn => {
            generic_warn!("{}", text.trim());
        }
        GGMLLogLevel::Error => {
            generic_error!("{}", text.trim());
        }
        GGMLLogLevel::Debug => {
            generic_debug!("{}", text.trim());
        }
        GGMLLogLevel::Cont => {
            // this means continue previous log
            // storing state to do this is a massive pain so it's just a lot easier to not
            // plus as far as i can tell it's not actually *used* anywhere
            // ggml splits at 128 chars and doesn't actually change the kind of log
            // so technically this is unused
            generic_trace!("{}", text.trim());
        }
        GGMLLogLevel::Unknown(level) => {
            generic_warn!(
                "ggml_logging_trampoline: unknown log level {}: message: {}",
                level,
                text.trim()
            );
        }
    }
}
