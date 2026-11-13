//! Escape sequence support of the Windows console, which is off by default on older consoles.

use crate::color::Stream;

#[cfg(windows)]
mod sys {
    use core::ffi::c_void;

    pub const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;

    unsafe extern "system" {
        pub fn GetConsoleMode(handle: *mut c_void, mode: *mut u32) -> i32;
        pub fn SetConsoleMode(handle: *mut c_void, mode: u32) -> i32;
    }
}

/// Turns on escape sequence processing for a terminal stream; `false` when the console refuses.
#[cfg(windows)]
pub fn enable(stream: Stream) -> bool {
    use std::os::windows::io::AsRawHandle as _;

    let handle = match stream {
        Stream::Stdout => std::io::stdout().as_raw_handle(),
        Stream::Stderr => std::io::stderr().as_raw_handle(),
    };
    let mut mode = 0_u32;
    // SAFETY: the handle belongs to the standard stream and `mode` outlives both calls.
    unsafe {
        if sys::GetConsoleMode(handle, &raw mut mode) == 0 {
            return false;
        }
        sys::SetConsoleMode(handle, mode | sys::ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0
    }
}

#[cfg(not(windows))]
pub const fn enable(_stream: Stream) -> bool {
    true
}
