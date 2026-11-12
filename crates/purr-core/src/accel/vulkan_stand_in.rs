//! Answers for Vulkan calls made while the Vulkan loader is absent.
//!
//! ggml registers every compiled-in backend the first time any device is enumerated, and
//! registering Vulkan calls into the loader, so those calls cannot be gated from outside ggml.
//! Without a loader they get Vulkan's own "initialization failed" status and ggml skips the
//! backend.

use std::ffi::{c_char, c_void};

const VK_ERROR_INITIALIZATION_FAILED: i32 = -3;

extern "system" fn vulkan_call_unavailable() -> i32 {
    VK_ERROR_INITIALIZATION_FAILED
}

/// Stand-in for `vkGetInstanceProcAddr` and `vkGetDeviceProcAddr`.
pub extern "system" fn proc_address_unavailable(
    _object: *const c_void,
    _name: *const c_char,
) -> *const c_void {
    vulkan_call_unavailable as *const c_void
}

/// Stand-in for the loader export `proc_name`.
#[must_use]
pub fn for_export(proc_name: &str) -> *const c_void {
    match proc_name {
        "vkGetInstanceProcAddr" | "vkGetDeviceProcAddr" => {
            proc_address_unavailable as *const c_void
        }
        _ => vulkan_call_unavailable as *const c_void,
    }
}
