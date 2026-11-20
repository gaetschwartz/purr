//! C interface of the browser build: the host feeds bytes in and takes the two outputs.

use crate::convert::{Options, convert};
use std::io::{Read, Write};

const LOCATION_CAPACITY: usize = 256;
const SINK_DATA: u32 = 0;
const SINK_GRAPH: u32 = 1;

#[link(wasm_import_module = "purr")]
unsafe extern "C" {
    fn purr_read(ptr: *mut u8, len: usize) -> isize;
    fn purr_write(sink: u32, ptr: *const u8, len: usize) -> i32;
    fn purr_location(ptr: *mut u8, capacity: usize) -> usize;
    fn purr_report_error(ptr: *const u8, len: usize);
}

struct HostInput;

impl Read for HostInput {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        // SAFETY: the host writes at most `buf.len()` bytes at `buf.as_mut_ptr()`.
        let read = unsafe { purr_read(buf.as_mut_ptr(), buf.len()) };
        usize::try_from(read).map_err(std::io::Error::other)
    }
}

struct HostSink(u32);

impl Write for HostSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // SAFETY: the host reads `buf.len()` bytes from `buf.as_ptr()` before returning.
        let status = unsafe { purr_write(self.0, buf.as_ptr(), buf.len()) };
        if status == 0 {
            Ok(buf.len())
        } else {
            Err(std::io::Error::other(
                "the host refused the converted bytes",
            ))
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Converts the model the host supplies; returns 0, or 1 after reporting the failure.
#[unsafe(no_mangle)]
pub extern "C" fn purr_convert() -> i32 {
    let mut location = vec![0_u8; LOCATION_CAPACITY];
    // SAFETY: the host writes at most `capacity` bytes and returns how many.
    let length = unsafe { purr_location(location.as_mut_ptr(), location.len()) };
    location.truncate(length);
    let outcome = String::from_utf8(location)
        .map_err(|error| error.to_string())
        .and_then(|location| {
            convert(
                &mut HostInput,
                &mut HostSink(SINK_DATA),
                &mut HostSink(SINK_GRAPH),
                &Options {
                    data_location: &location,
                },
            )
            .map_err(|error| error.to_string())
        });
    match outcome {
        Ok(_) => 0,
        Err(message) => {
            // SAFETY: the host reads `len` bytes from `ptr` before returning.
            unsafe { purr_report_error(message.as_ptr(), message.len()) };
            1
        }
    }
}
