/// Whether the dynamic library `name` can be loaded by this process.
///
/// Uses the default search order, exactly like the loader does for an import, and leaves the
/// library mapped so a later delay-loaded import binds to the module that was checked.
#[cfg(windows)]
#[must_use]
pub fn is_loadable(name: &str) -> bool {
    use windows::{
        Win32::System::LibraryLoader::{LOAD_LIBRARY_FLAGS, LoadLibraryExW},
        core::PCWSTR,
    };

    let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe { LoadLibraryExW(PCWSTR(wide.as_ptr()), None, LOAD_LIBRARY_FLAGS(0)) }.is_ok()
}

/// Whether the dynamic library `name` can be loaded by this process.
///
/// Uses the dynamic linker's search order and leaves the library mapped, so the libraries that
/// open it later get the module that was checked.
#[cfg(target_os = "linux")]
#[must_use]
pub fn is_loadable(name: &str) -> bool {
    let Ok(name) = std::ffi::CString::new(name) else {
        return false;
    };
    !unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) }.is_null()
}

#[cfg(not(any(windows, target_os = "linux")))]
#[must_use]
pub const fn is_loadable(_name: &str) -> bool {
    true
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn the_c_library_is_loadable() {
        assert!(is_loadable("libc.so.6"));
    }

    #[test]
    fn a_missing_library_is_not_loadable() {
        assert!(!is_loadable("libpurr-missing-library.so.1"));
    }

    #[test]
    fn a_name_with_a_nul_byte_is_not_loadable() {
        assert!(!is_loadable("libc.so.6\0"));
    }
}
