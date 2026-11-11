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

#[cfg(not(windows))]
#[must_use]
pub const fn is_loadable(_name: &str) -> bool {
    true
}
