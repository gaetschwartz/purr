//! One streaming pass over an ONNX model: big initializers go to a data sink, everything else
//! stays in a small graph that is written last.

use crate::error::ConvertError;
use crate::fields::{Field, Remainder, next_field, read_failed};
use crate::proto::onnx_proto3::tensor_proto::DataLocation;
use crate::proto::onnx_proto3::{GraphProto, ModelProto, StringStringEntryProto, TensorProto};
use crate::tied_embedding;
use protobuf::rt::WireType;
use protobuf::{CodedInputStream, EnumOrUnknown, Message, MessageField};
use std::io::{Read, Write};

/// Initializers of at least this many bytes move to the data file.
pub const EXTERNAL_THRESHOLD_BYTES: u32 = 1024;
/// Largest piece of a tensor held in memory at once.
pub const CHUNK_BYTES: usize = 1 << 20;
pub const DATA_ALIGNMENT: u64 = 4096;

const MODEL_GRAPH: u32 = 7;
const GRAPH_INITIALIZER: u32 = 5;
const TENSOR_RAW_DATA: u32 = 9;

#[derive(Debug, Clone, Copy)]
pub struct Options<'a> {
    /// Name the graph gives the data file: the base name transformers.js asks for.
    pub data_location: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Summary {
    pub data_bytes: u64,
    pub graph_bytes: u64,
    pub external_tensors: usize,
    pub tied_embedding_branches: usize,
}

struct DataSink<'w> {
    writer: &'w mut dyn Write,
    position: u64,
    chunk: Vec<u8>,
}

impl DataSink<'_> {
    fn write(&mut self, bytes: &[u8]) -> Result<(), ConvertError> {
        self.writer
            .write_all(bytes)
            .map_err(|source| ConvertError::Write {
                sink: "data",
                source,
            })?;
        self.position += bytes.len() as u64;
        Ok(())
    }

    fn pad_to_alignment(&mut self) -> Result<(), ConvertError> {
        const ZEROS: [u8; DATA_ALIGNMENT as usize] = [0; DATA_ALIGNMENT as usize];
        let missing = (DATA_ALIGNMENT - self.position % DATA_ALIGNMENT) % DATA_ALIGNMENT;
        self.write(ZEROS.get(..missing as usize).unwrap_or_default())
    }

    /// Copies `length` bytes of the input to the data file, returning where they start.
    fn copy_from(
        &mut self,
        input: &mut CodedInputStream,
        length: u32,
    ) -> Result<u64, ConvertError> {
        self.pad_to_alignment()?;
        let offset = self.position;
        let mut remaining = length as usize;
        while remaining > 0 {
            let take = remaining.min(CHUNK_BYTES);
            let mut chunk = std::mem::take(&mut self.chunk);
            chunk.clear();
            input
                .read_raw_bytes_into(take as u32, &mut chunk)
                .map_err(read_failed)?;
            let written = self.write(&chunk);
            self.chunk = chunk;
            written?;
            remaining -= take;
        }
        Ok(offset)
    }
}

fn external_entry(key: &str, value: String) -> StringStringEntryProto {
    let mut entry = StringStringEntryProto::new();
    key.clone_into(&mut entry.key);
    entry.value = value;
    entry
}

fn read_initializer(
    input: &mut CodedInputStream,
    sink: &mut DataSink,
    options: &Options,
) -> Result<TensorProto, ConvertError> {
    let length = input.read_raw_varint32().map_err(read_failed)?;
    let limit = input.push_limit(u64::from(length)).map_err(read_failed)?;
    let mut rest = Remainder::<TensorProto>::new();
    let mut placed: Option<(u64, u32)> = None;
    while let Some(field) = next_field(input)? {
        if field.number == TENSOR_RAW_DATA && field.wire_type == WireType::LengthDelimited {
            let size = input.read_raw_varint32().map_err(read_failed)?;
            if size >= EXTERNAL_THRESHOLD_BYTES {
                placed = Some((sink.copy_from(input, size)?, size));
            } else {
                let bytes = input.read_raw_bytes(size).map_err(read_failed)?;
                rest.keep_bytes(TENSOR_RAW_DATA, bytes);
            }
        } else {
            rest.keep(input, &field)?;
        }
    }
    input.pop_limit(limit);

    let mut tensor = rest.finish()?;
    if tensor.data_location.enum_value() == Ok(DataLocation::EXTERNAL) {
        return Err(ConvertError::AlreadyExternal { name: tensor.name });
    }
    if let Some((offset, size)) = placed {
        tensor.data_location = EnumOrUnknown::new(DataLocation::EXTERNAL);
        tensor.external_data = vec![
            external_entry("location", options.data_location.to_owned()),
            external_entry("offset", offset.to_string()),
            external_entry("length", size.to_string()),
        ];
    }
    Ok(tensor)
}

fn read_graph(
    input: &mut CodedInputStream,
    sink: &mut DataSink,
    options: &Options,
) -> Result<GraphProto, ConvertError> {
    let length = input.read_raw_varint32().map_err(read_failed)?;
    let limit = input.push_limit(u64::from(length)).map_err(read_failed)?;
    let mut rest = Remainder::<GraphProto>::new();
    let mut initializers = Vec::new();
    while let Some(field) = next_field(input)? {
        if field.number == GRAPH_INITIALIZER && field.wire_type == WireType::LengthDelimited {
            initializers.push(read_initializer(input, sink, options)?);
        } else {
            rest.keep(input, &field)?;
        }
    }
    input.pop_limit(limit);

    let mut graph = rest.finish()?;
    graph.initializer = initializers;
    Ok(graph)
}

fn is_graph(field: &Field) -> bool {
    field.number == MODEL_GRAPH && field.wire_type == WireType::LengthDelimited
}

/// Splits the model on `input` into a data file and a graph file, and fixes the tied
/// embedding where the model has one. Writes all of the data first, then the graph.
pub fn convert(
    input: &mut dyn Read,
    data: &mut dyn Write,
    graph: &mut dyn Write,
    options: &Options,
) -> Result<Summary, ConvertError> {
    let mut stream = CodedInputStream::new(input);
    let mut sink = DataSink {
        writer: data,
        position: 0,
        chunk: Vec::new(),
    };
    let mut rest = Remainder::<ModelProto>::new();
    let mut main_graph = None;
    while let Some(field) = next_field(&mut stream)? {
        if is_graph(&field) {
            main_graph = Some(read_graph(&mut stream, &mut sink, options)?);
        } else {
            rest.keep(&mut stream, &field)?;
        }
    }

    let mut model = rest.finish()?;
    model.graph = MessageField::some(main_graph.ok_or(ConvertError::NoGraph)?);
    let external_tensors = model
        .graph
        .initializer
        .iter()
        .filter(|tensor| tensor.data_location.enum_value() == Ok(DataLocation::EXTERNAL))
        .count();
    let tied_embedding_branches = tied_embedding::rewrite(&mut model);

    let encoded = model
        .write_to_bytes()
        .map_err(|source| ConvertError::Encode { source })?;
    graph
        .write_all(&encoded)
        .map_err(|source| ConvertError::Write {
            sink: "graph",
            source,
        })?;
    Ok(Summary {
        data_bytes: sink.position,
        graph_bytes: encoded.len() as u64,
        external_tensors,
        tied_embedding_branches,
    })
}
