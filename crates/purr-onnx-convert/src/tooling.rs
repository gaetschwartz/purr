//! Helpers for the manifest generator and the real-model tests: fetching the pinned original
//! files and measuring what the converter makes of them.

use crate::convert::{Options, Summary, TablePrecision, convert};
use oopsie::{Welp, WelpResultExt as _};
use sha2::{Digest as _, Sha256};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const COPY_BUFFER_BYTES: usize = 1 << 20;

/// Length and SHA-256 (lower-case hex) of a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Measured {
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conversion {
    pub graph: Measured,
    pub data: Measured,
    pub summary: Summary,
}

struct HashingWriter<W: Write> {
    inner: W,
    hasher: Sha256,
    written: u64,
}

impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
            written: 0,
        }
    }

    fn measured(self) -> Measured {
        Measured {
            size_bytes: self.written,
            sha256: hex::encode(self.hasher.finalize()),
        }
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buf)?;
        if let Some(accepted) = buf.get(..written) {
            self.hasher.update(accepted);
        }
        self.written += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn measure_stream(reader: &mut dyn Read) -> Result<Measured, Welp> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    let mut size_bytes = 0_u64;
    loop {
        let read = reader.read(&mut buffer).welp()?;
        let Some(filled) = buffer.get(..read).filter(|filled| !filled.is_empty()) else {
            break;
        };
        hasher.update(filled);
        size_bytes += read as u64;
    }
    Ok(Measured {
        size_bytes,
        sha256: hex::encode(hasher.finalize()),
    })
}

pub fn measure_file(path: &Path) -> Result<Measured, Welp> {
    measure_stream(
        &mut File::open(path).with_welp_context(|_| format!("opening {}", path.display()))?,
    )
}

/// Downloads `url` to `destination` unless a file with the expected digest is already there.
pub fn fetch_verified(url: &str, expected: &Measured, destination: &Path) -> Result<(), Welp> {
    if destination.is_file() && measure_file(destination)? == *expected {
        return Ok(());
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).welp()?;
    }
    let mut response = ureq::get(url)
        .call()
        .with_welp_context(|_| format!("GET {url}"))?;
    let partial = destination.with_extension("partial");
    let mut sink = HashingWriter::new(BufWriter::new(File::create(&partial).welp()?));
    let mut body = response.body_mut().as_reader();
    std::io::copy(&mut body, &mut sink).with_welp_context(|_| format!("downloading {url}"))?;
    sink.flush().welp()?;
    let received = sink.measured();
    if received != *expected {
        return Err(Welp::new(format!(
            "{url}: received {} bytes with sha256 {}, expected {} bytes with sha256 {}",
            received.size_bytes, received.sha256, expected.size_bytes, expected.sha256
        )));
    }
    std::fs::rename(&partial, destination).welp()?;
    Ok(())
}

/// Where the data file of `model_name` is named in the converted graph.
#[must_use]
pub fn data_name(model_name: &str) -> String {
    format!("{model_name}_data")
}

/// Converts `input` and measures both outputs. With `output_dir` the outputs are also written
/// there under their final names.
pub fn convert_file(
    input: &Path,
    output_dir: Option<&Path>,
    table: TablePrecision,
) -> Result<Conversion, Welp> {
    let name = input
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Welp::new(format!("{} has no file name", input.display())))?;
    let data_name = data_name(name);
    let open = |file: &str| -> Result<Box<dyn Write>, Welp> {
        match output_dir {
            Some(dir) => {
                std::fs::create_dir_all(dir).welp()?;
                Ok(Box::new(BufWriter::new(
                    File::create(dir.join(file)).welp()?,
                )))
            }
            None => Ok(Box::new(std::io::sink())),
        }
    };
    let mut data = HashingWriter::new(open(&data_name)?);
    let mut graph = HashingWriter::new(open(name)?);
    let mut reader = BufReader::new(File::open(input).welp()?);
    let summary = convert(
        &mut reader,
        &mut data,
        &mut graph,
        &Options {
            data_location: &data_name,
            table,
        },
    )
    .welp()?;
    data.flush().welp()?;
    graph.flush().welp()?;
    Ok(Conversion {
        graph: graph.measured(),
        data: data.measured(),
        summary,
    })
}

/// Cache folder for downloaded originals, shared by every build profile of the workspace.
#[must_use]
pub fn cache_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("target")
        .join("web-model-cache")
}
