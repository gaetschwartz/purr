//! Failure hook of the MSVC delay-load helper.
//!
//! GPU runtime libraries are delay-loaded so a binary starts without them. When the Vulkan
//! loader is absent the hook answers its imports with [`vulkan_stand_in`]. Any other missing
//! import is a bug, so it aborts with a readable message.

use std::ffi::{CStr, c_char, c_void};

use windows::{
    Win32::System::LibraryLoader::{LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW},
    core::w,
};

use super::{backend::AccelBackend, runtime_library::required, vulkan_stand_in};

const NOTIFY_FAIL_LOAD_LIB: u32 = 3;
const NOTIFY_FAIL_GET_PROC: u32 = 4;

#[repr(C)]
pub struct DelayLoadProc {
    import_by_name: i32,
    name_or_ordinal: *const c_char,
}

#[repr(C)]
pub struct DelayLoadInfo {
    _size: u32,
    _descriptor: *const c_void,
    _thunk: *mut *const c_void,
    dll_name: *const c_char,
    proc: DelayLoadProc,
    _module: *mut c_void,
    _current_proc: *const c_void,
    last_error: u32,
}

type FailureHook = unsafe extern "system" fn(u32, *const DelayLoadInfo) -> *const c_void;

#[unsafe(no_mangle)]
#[allow(non_upper_case_globals, reason = "symbol name fixed by delayimp.lib")]
pub static __pfnDliFailureHook2: Option<FailureHook> = Some(failure_hook);

unsafe fn text(raw: *const c_char) -> String {
    if raw.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(raw) }
        .to_string_lossy()
        .into_owned()
}

fn is_vulkan_loader(dll: &str) -> bool {
    required(AccelBackend::Vulkan)
        .iter()
        .any(|loader| loader.eq_ignore_ascii_case(dll))
}

fn abort_missing(dll: &str, proc_name: &str, last_error: u32) -> ! {
    eprintln!(
        "purr: fatal: the library {dll} is required by {proc_name} but cannot be used (Windows error {last_error}); \
         the backend that needs it should have been skipped"
    );
    std::process::abort()
}

unsafe extern "system" fn failure_hook(
    notification: u32,
    info: *const DelayLoadInfo,
) -> *const c_void {
    let Some(info) = (unsafe { info.as_ref() }) else {
        return std::ptr::null();
    };
    let dll = unsafe { text(info.dll_name) };
    let proc_name = if info.proc.import_by_name == 0 {
        String::from("an ordinal import")
    } else {
        unsafe { text(info.proc.name_or_ordinal) }
    };
    match notification {
        NOTIFY_FAIL_LOAD_LIB if is_vulkan_loader(&dll) => {
            // a valid module whose GetProcAddress fails, which routes every import back here
            unsafe { LoadLibraryExW(w!("kernel32.dll"), None, LOAD_LIBRARY_SEARCH_SYSTEM32) }
                .map_or_else(
                    |_| abort_missing(&dll, &proc_name, info.last_error),
                    |module| module.0.cast_const(),
                )
        }
        NOTIFY_FAIL_GET_PROC if is_vulkan_loader(&dll) && info.proc.import_by_name != 0 => {
            vulkan_stand_in::for_export(&proc_name)
        }
        NOTIFY_FAIL_LOAD_LIB | NOTIFY_FAIL_GET_PROC => {
            abort_missing(&dll, &proc_name, info.last_error)
        }
        _ => std::ptr::null(),
    }
}
