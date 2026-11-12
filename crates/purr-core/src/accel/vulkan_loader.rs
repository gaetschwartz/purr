//! Forwarders for the Vulkan functions ggml calls directly rather than through its dispatcher.
//!
//! The link step redirects those calls here (see `purr-delayload`), so the binary does not need
//! the Vulkan loader to start. The loader is opened on first use; without it the forwarders
//! answer with [`vulkan_stand_in`].

use std::{
    ffi::{CStr, CString, c_char, c_void},
    ptr::NonNull,
    sync::OnceLock,
};

use super::{
    backend::AccelBackend,
    library_loader::{SystemLoader, open_all},
    runtime_dir::{Environment, search_dirs},
    runtime_library::required,
    vulkan_stand_in,
};

/// A library handle or a function address, both valid for the whole process.
#[derive(Clone, Copy)]
struct Address(NonNull<c_void>);

// SAFETY: `dlopen` handles and the addresses of loaded functions are process-wide and never
// released, so they can be shared with and used from any thread.
unsafe impl Send for Address {}
// SAFETY: as above.
unsafe impl Sync for Address {}

fn loader() -> Option<Address> {
    static LOADER: OnceLock<Option<Address>> = OnceLock::new();
    *LOADER.get_or_init(|| {
        let libraries = required(AccelBackend::Vulkan);
        let dirs: Vec<_> = search_dirs(&Environment::current())
            .into_iter()
            .map(|dir| dir.path)
            .collect();
        open_all(&SystemLoader, &libraries, &dirs).ok()?;
        let name = CString::new(*libraries.first()?).ok()?;
        let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        NonNull::new(handle).map(Address)
    })
}

fn export(name: &CStr) -> Option<Address> {
    let loader = loader()?;
    NonNull::new(unsafe { libc::dlsym(loader.0.as_ptr(), name.as_ptr()) }).map(Address)
}

fn cached_export(cell: &'static OnceLock<Address>, name: &CStr) -> *mut c_void {
    cell.get_or_init(|| {
        export(name).unwrap_or_else(|| {
            eprintln!(
                "purr: fatal: {} was called but the Vulkan loader cannot be used; \
                 the Vulkan backend should have been skipped",
                name.to_string_lossy()
            );
            std::process::abort()
        })
    })
    .0
    .as_ptr()
}

type GetInstanceProcAddr = unsafe extern "C" fn(*const c_void, *const c_char) -> *const c_void;
type GetPhysicalDeviceFeatures2 = unsafe extern "C" fn(*mut c_void, *mut c_void);
type CmdCopyBuffer =
    unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, u32, *const c_void);

#[unsafe(no_mangle)]
#[allow(non_snake_case, reason = "symbol name fixed by the linker's --wrap")]
unsafe extern "C" fn __wrap_vkGetInstanceProcAddr(
    instance: *const c_void,
    name: *const c_char,
) -> *const c_void {
    match export(c"vkGetInstanceProcAddr") {
        Some(real) => {
            let real =
                unsafe { std::mem::transmute::<*mut c_void, GetInstanceProcAddr>(real.0.as_ptr()) };
            unsafe { real(instance, name) }
        }
        None => vulkan_stand_in::proc_address_unavailable(instance, name),
    }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case, reason = "symbol name fixed by the linker's --wrap")]
unsafe extern "C" fn __wrap_vkGetPhysicalDeviceFeatures2(
    physical_device: *mut c_void,
    features: *mut c_void,
) {
    static REAL: OnceLock<Address> = OnceLock::new();
    let real = cached_export(&REAL, c"vkGetPhysicalDeviceFeatures2");
    let real = unsafe { std::mem::transmute::<*mut c_void, GetPhysicalDeviceFeatures2>(real) };
    unsafe { real(physical_device, features) }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case, reason = "symbol name fixed by the linker's --wrap")]
unsafe extern "C" fn __wrap_vkCmdCopyBuffer(
    command_buffer: *mut c_void,
    source: *mut c_void,
    destination: *mut c_void,
    region_count: u32,
    regions: *const c_void,
) {
    static REAL: OnceLock<Address> = OnceLock::new();
    let real = cached_export(&REAL, c"vkCmdCopyBuffer");
    let real = unsafe { std::mem::transmute::<*mut c_void, CmdCopyBuffer>(real) };
    unsafe { real(command_buffer, source, destination, region_count, regions) }
}
