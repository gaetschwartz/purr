//! The only module that touches the process's standard streams (`clippy.toml` bans
//! `std::io::stdout` and `std::io::stderr` everywhere else).
#![allow(clippy::disallowed_methods)]

use crate::color::Stream;
use std::io::{self, IsTerminal as _, Write as _};

/// Where text finally lands.
pub trait Sink: Send + Sync {
    fn write(&self, stream: Stream, text: &str) -> io::Result<()>;
}

/// The process's own stdout and stderr; every write is flushed so partial lines show at once.
pub struct ProcessSink;

impl Sink for ProcessSink {
    fn write(&self, stream: Stream, text: &str) -> io::Result<()> {
        match stream {
            Stream::Stdout => {
                let mut out = io::stdout().lock();
                out.write_all(text.as_bytes())?;
                out.flush()
            }
            Stream::Stderr => {
                let mut err = io::stderr().lock();
                err.write_all(text.as_bytes())?;
                err.flush()
            }
        }
    }
}

pub fn is_terminal(stream: Stream) -> bool {
    match stream {
        Stream::Stdout => io::stdout().is_terminal(),
        Stream::Stderr => io::stderr().is_terminal(),
    }
}

pub fn stdin_is_terminal() -> bool {
    io::stdin().is_terminal()
}

#[cfg(windows)]
pub fn raw_handle(stream: Stream) -> std::os::windows::io::RawHandle {
    use std::os::windows::io::AsRawHandle as _;

    match stream {
        Stream::Stdout => io::stdout().as_raw_handle(),
        Stream::Stderr => io::stderr().as_raw_handle(),
    }
}

#[cfg(test)]
pub mod memory {
    use super::{Sink, Stream};
    use parking_lot::Mutex;
    use std::io;
    use std::sync::Arc;

    /// Keeps what was written to each stream, in order.
    #[derive(Clone, Default)]
    pub struct MemorySink {
        stdout: Arc<Mutex<String>>,
        stderr: Arc<Mutex<String>>,
    }

    impl MemorySink {
        pub fn stdout(&self) -> String {
            self.stdout.lock().clone()
        }

        pub fn stderr(&self) -> String {
            self.stderr.lock().clone()
        }
    }

    impl Sink for MemorySink {
        fn write(&self, stream: Stream, text: &str) -> io::Result<()> {
            match stream {
                Stream::Stdout => self.stdout.lock().push_str(text),
                Stream::Stderr => self.stderr.lock().push_str(text),
            }
            Ok(())
        }
    }
}
