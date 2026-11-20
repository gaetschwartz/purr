//! One streaming pass over an ONNX model: big initializers go to a data sink, everything else
//! stays in a small graph that is written last.

use crate::attention_fusion;
use crate::error::ConvertError;
use crate::fields::{Field, Remainder, next_field, read_failed};
use crate::proto::onnx_proto3::tensor_proto::{DataLocation, DataType};
use crate::proto::onnx_proto3::{GraphProto, ModelProto, StringStringEntryProto, TensorProto};
use crate::tied_embedding;
use half::f16;
use protobuf::rt::WireType;
use protobuf::{CodedInputStream, Enum as _, EnumOrUnknown, Message, MessageField};
use std::collections::BTreeSet;
use std::io::{Read, Write};

/// Initializers of at least this many bytes move to the data file.
pub const EXTERNAL_THRESHOLD_BYTES: u32 = 1024;
/// Largest piece of a tensor held in memory at once.
pub const CHUNK_BYTES: usize = 1 << 20;
pub const DATA_ALIGNMENT: u64 = 4096;

const MODEL_GRAPH: u32 = 7;
const GRAPH_INITIALIZER: u32 = 5;
const TENSOR_RAW_DATA: u32 = 9;

/// How the embedding table shared by a decoder's lookup and output head is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TablePrecision {
    Full,
    /// Rounded to half precision while it streams through; the graph reads it as such.
    Half,
}

#[derive(Debug, Clone, Copy)]
pub struct Options<'a> {
    /// Name the graph gives the data file: the base name transformers.js asks for.
    pub data_location: &'a str,
    pub table: TablePrecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Summary {
    pub data_bytes: u64,
    pub graph_bytes: u64,
    pub external_tensors: usize,
    pub tied_embedding_branches: usize,
    pub half_precision_tables: usize,
    pub fused_attention_blocks: usize,
}

struct DataSink<'w> {
    writer: &'w mut dyn Write,
    position: u64,
    chunk: Vec<u8>,
    half_chunk: Vec<u8>,
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

    /// Copies the `f32` elements in the next `length` bytes of the input to the data file as
    /// `f16`, returning where they start and how many bytes they take.
    fn copy_half_from(
        &mut self,
        input: &mut CodedInputStream,
        length: u32,
        name: &str,
    ) -> Result<(u64, u32), ConvertError> {
        if !length.is_multiple_of(4) {
            return Err(ConvertError::HalfTableSize {
                name: name.to_owned(),
            });
        }
        self.pad_to_alignment()?;
        let offset = self.position;
        let mut remaining = length as usize;
        while remaining > 0 {
            let take = remaining.min(CHUNK_BYTES);
            let mut chunk = std::mem::take(&mut self.chunk);
            let mut half = std::mem::take(&mut self.half_chunk);
            chunk.clear();
            half.clear();
            let read = input
                .read_raw_bytes_into(take as u32, &mut chunk)
                .map_err(read_failed);
            let written = read
                .and_then(|()| narrow(&chunk, &mut half, name))
                .and_then(|()| self.write(&half));
            self.chunk = chunk;
            self.half_chunk = half;
            written?;
            remaining -= take;
        }
        Ok((offset, length / 2))
    }
}

fn narrow(full: &[u8], half: &mut Vec<u8>, name: &str) -> Result<(), ConvertError> {
    let (elements, remainder) = full.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(ConvertError::HalfTableSize {
            name: name.to_owned(),
        });
    }
    for bytes in elements {
        let value = f32::from_le_bytes(*bytes);
        let narrowed = f16::from_f32(value);
        if value.is_finite() && !narrowed.is_finite() {
            return Err(ConvertError::HalfOverflow {
                name: name.to_owned(),
            });
        }
        half.extend_from_slice(&narrowed.to_le_bytes());
    }
    Ok(())
}

fn external_entry(key: &str, value: String) -> StringStringEntryProto {
    let mut entry = StringStringEntryProto::new();
    key.clone_into(&mut entry.key);
    entry.value = value;
    entry
}

/// Whether the tensor gathered so far is a table to store as half precision: a planned name with
/// two dimensions and the full-precision elements that `size` bytes hold.
fn is_half_table(so_far: &TensorProto, plan: &BTreeSet<String>, size: u32) -> bool {
    let elements = match so_far.dims.as_slice() {
        [rows, columns] => rows.checked_mul(*columns),
        _ => None,
    };
    plan.contains(&so_far.name)
        && so_far.data_type == DataType::FLOAT.value()
        && elements.and_then(|elements| elements.checked_mul(4)) == Some(i64::from(size))
}

fn read_initializer(
    input: &mut CodedInputStream,
    sink: &mut DataSink,
    options: &Options,
    plan: &BTreeSet<String>,
    converted: &mut BTreeSet<String>,
) -> Result<TensorProto, ConvertError> {
    let length = input.read_raw_varint32().map_err(read_failed)?;
    let limit = input.push_limit(u64::from(length)).map_err(read_failed)?;
    let mut rest = Remainder::<TensorProto>::new();
    let mut placed: Option<(u64, u32)> = None;
    let mut narrowed = false;
    while let Some(field) = next_field(input)? {
        if field.number == TENSOR_RAW_DATA && field.wire_type == WireType::LengthDelimited {
            let size = input.read_raw_varint32().map_err(read_failed)?;
            if size >= EXTERNAL_THRESHOLD_BYTES {
                let so_far = if plan.is_empty() {
                    None
                } else {
                    Some(rest.peek()?)
                };
                placed = Some(match so_far {
                    Some(so_far) if is_half_table(&so_far, plan, size) => {
                        narrowed = true;
                        sink.copy_half_from(input, size, &so_far.name)?
                    }
                    Some(_) | None => (sink.copy_from(input, size)?, size),
                });
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
    if narrowed {
        tensor.data_type = DataType::FLOAT16.value();
        converted.insert(tensor.name.clone());
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

struct ReadGraph {
    graph: GraphProto,
    planned: BTreeSet<String>,
    converted: BTreeSet<String>,
}

fn plan_tables(
    rest: &Remainder<GraphProto>,
    options: &Options,
) -> Result<BTreeSet<String>, ConvertError> {
    match options.table {
        TablePrecision::Full => Ok(BTreeSet::new()),
        TablePrecision::Half => Ok(tied_embedding::half_precision_candidates(rest.peek()?)),
    }
}

fn read_graph(
    input: &mut CodedInputStream,
    sink: &mut DataSink,
    options: &Options,
) -> Result<ReadGraph, ConvertError> {
    let length = input.read_raw_varint32().map_err(read_failed)?;
    let limit = input.push_limit(u64::from(length)).map_err(read_failed)?;
    let mut rest = Remainder::<GraphProto>::new();
    let mut initializers = Vec::new();
    let mut planned: Option<BTreeSet<String>> = None;
    let mut converted = BTreeSet::new();
    while let Some(field) = next_field(input)? {
        if field.number == GRAPH_INITIALIZER && field.wire_type == WireType::LengthDelimited {
            let plan = match &mut planned {
                Some(plan) => plan,
                empty => empty.insert(plan_tables(&rest, options)?),
            };
            initializers.push(read_initializer(
                input,
                sink,
                options,
                plan,
                &mut converted,
            )?);
        } else {
            rest.keep(input, &field)?;
        }
    }
    input.pop_limit(limit);

    let mut graph = rest.finish()?;
    graph.initializer = initializers;
    Ok(ReadGraph {
        graph,
        planned: planned.unwrap_or_default(),
        converted,
    })
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
        half_chunk: Vec::new(),
    };
    let mut rest = Remainder::<ModelProto>::new();
    let mut read_graph_result: Option<ReadGraph> = None;
    while let Some(field) = next_field(&mut stream)? {
        if is_graph(&field) {
            read_graph_result = Some(read_graph(&mut stream, &mut sink, options)?);
        } else {
            rest.keep(&mut stream, &field)?;
        }
    }

    let mut model = rest.finish()?;
    let ReadGraph {
        graph: main_graph,
        planned,
        converted,
    } = read_graph_result.ok_or(ConvertError::NoGraph)?;
    if options.table == TablePrecision::Half {
        let mut nodes = GraphProto::new();
        nodes.node.clone_from(&main_graph.node);
        nodes.output.clone_from(&main_graph.output);
        if tied_embedding::half_precision_candidates(nodes) != planned {
            return Err(ConvertError::TablesAfterWeights);
        }
    }
    model.graph = MessageField::some(main_graph);
    let external_tensors = model
        .graph
        .initializer
        .iter()
        .filter(|tensor| tensor.data_location.enum_value() == Ok(DataLocation::EXTERNAL))
        .count();
    let tied_embedding_branches = tied_embedding::rewrite(&mut model, &converted)?;
    let fused_attention_blocks = attention_fusion::fuse(&mut model);

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
        half_precision_tables: converted.len(),
        fused_attention_blocks,
    })
}
