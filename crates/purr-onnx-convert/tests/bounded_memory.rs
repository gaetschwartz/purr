mod common;

use common::counting_allocator::{CountingAllocator, measure};
use common::{Decoder, EmbeddingValues, TestResult};
use purr_onnx_convert::convert::{CHUNK_BYTES, Options, TablePrecision, convert};
use std::io::Write;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

struct Discard(u64);

impl Write for Discard {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn converting_a_model_never_holds_more_than_a_chunk_of_its_weights() -> TestResult {
    let decoder = Decoder {
        filler_tensors: 40,
        filler_bytes: 2 * CHUNK_BYTES + 12_345,
        ..Decoder::small()
    };
    let input = decoder.bytes()?;
    assert!(input.len() > 80 * 1024 * 1024);

    let (mut data, mut graph) = (Discard(0), Discard(0));
    let (summary, usage) = measure(|| {
        convert(
            &mut input.as_slice(),
            &mut data,
            &mut graph,
            &Options {
                data_location: "decoder.onnx_data",
                table: TablePrecision::Full,
            },
        )
    });
    let summary = summary?;

    assert!(summary.data_bytes > 80 * 1024 * 1024);
    assert_eq!(data.0, summary.data_bytes);
    assert!(usage.largest_allocation <= CHUNK_BYTES);
    assert!(
        usage.peak_above_start <= 2 * CHUNK_BYTES,
        "peak {} bytes for {} bytes of weights",
        usage.peak_above_start,
        summary.data_bytes
    );
    Ok(())
}

#[test]
fn narrowing_a_table_never_holds_more_than_a_chunk_of_it() -> TestResult {
    let decoder = Decoder {
        vocabulary: 40_000,
        width: 64,
        filler_tensors: 0,
        embedding_values: EmbeddingValues::Floats { largest: 3.0 },
        ..Decoder::small()
    };
    let input = decoder.bytes()?;
    assert!(input.len() > 8 * CHUNK_BYTES);

    let (mut data, mut graph) = (Discard(0), Discard(0));
    let (summary, usage) = measure(|| {
        convert(
            &mut input.as_slice(),
            &mut data,
            &mut graph,
            &Options {
                data_location: "decoder.onnx_data",
                table: TablePrecision::Half,
            },
        )
    });
    let summary = summary?;

    assert_eq!(summary.half_precision_tables, 1);
    assert_eq!(data.0, summary.data_bytes);
    assert!(summary.data_bytes < 40_000 * 64 * 3);
    assert!(usage.largest_allocation <= CHUNK_BYTES);
    assert!(
        usage.peak_above_start <= 2 * CHUNK_BYTES,
        "peak {} bytes for {} bytes of weights",
        usage.peak_above_start,
        summary.data_bytes
    );
    Ok(())
}
