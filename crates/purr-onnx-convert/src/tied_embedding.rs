//! Rewrites the language-model head of a merged decoder that shares its embedding table.
//!
//! Each `If` branch transposes the table for the output `MatMul`, and ONNX Runtime folds that
//! `Transpose` into a copy of the table per branch. A `Gemm` with `transB` reads the table as it
//! is, so nothing is left to fold.
//!
//! A table that the converter stored as half precision is read by the `Gather` and the `Gemm`
//! in half precision as well, and only their activations are cast: casting the table itself would
//! be constant-folded back into a full-precision copy.

use crate::error::ConvertError;
use crate::proto::onnx_proto3::attribute_proto::AttributeType;
use crate::proto::onnx_proto3::tensor_proto::DataType;
use crate::proto::onnx_proto3::{AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto};
use protobuf::{Enum as _, EnumOrUnknown, MessageField};
use std::collections::{BTreeMap, BTreeSet};

const FIRST_OPSET_WITH_OPTIONAL_GEMM_BIAS: i64 = 11;
const HEAD_PREFIX: &str = "purr_tied_";

#[derive(Clone, Copy)]
struct Table {
    vocabulary: i64,
    width: i64,
}

struct Match {
    transpose: usize,
    matmul: usize,
    table: String,
    shape: Table,
}

type ShapeOf<'a> = &'a dyn Fn(&str) -> Option<Table>;

/// Applies the rewrite to every `If` branch that has the pattern, returning how many it changed.
/// Tables in `half` are stored as half precision and get the half-precision form.
pub fn rewrite(model: &mut ModelProto, half: &BTreeSet<String>) -> Result<usize, ConvertError> {
    let opset = model
        .opset_import
        .iter()
        .find(|opset| opset.domain.is_empty() || opset.domain == "ai.onnx")
        .map_or(0, |opset| opset.version);
    let Some(graph) = model.graph.as_mut() else {
        return Ok(0);
    };
    let rewritten = if opset < FIRST_OPSET_WITH_OPTIONAL_GEMM_BIAS {
        0
    } else {
        let tables: BTreeMap<String, Table> = graph
            .initializer
            .iter()
            .filter_map(|tensor| match tensor.dims.as_slice() {
                [vocabulary, width] => Some((
                    tensor.name.clone(),
                    Table {
                        vocabulary: *vocabulary,
                        width: *width,
                    },
                )),
                _ => None,
            })
            .collect();
        rewrite_graph(graph, &|name| tables.get(name).copied(), half)
    };
    if let Some(name) = half.iter().find(|name| !only_gathers_read(graph, name)) {
        return Err(ConvertError::HalfTableStillRead { name: name.clone() });
    }
    Ok(rewritten)
}

/// The tables of a graph's nodes that can be stored as half precision: those whose every reader
/// the rewrite turns into a half-precision one. The initializers need not be known yet.
#[must_use]
pub fn half_precision_candidates(mut nodes: GraphProto) -> BTreeSet<String> {
    let mut candidates = BTreeSet::new();
    for branch in branches(&nodes) {
        let anywhere: ShapeOf = &|_| {
            Some(Table {
                vocabulary: 0,
                width: 0,
            })
        };
        let mut scratch = branch.clone();
        while let Some(found) = find_pattern(&scratch, anywhere) {
            candidates.insert(found.table.clone());
            apply(&mut scratch, &found, 0, false);
        }
    }
    rewrite_graph(
        &mut nodes,
        &|_| {
            Some(Table {
                vocabulary: 0,
                width: 0,
            })
        },
        &candidates,
    );
    candidates
        .into_iter()
        .filter(|name| only_gathers_read(&nodes, name))
        .collect()
}

fn branches(graph: &GraphProto) -> impl Iterator<Item = &GraphProto> {
    graph
        .node
        .iter()
        .filter(|node| node.op_type == "If")
        .flat_map(|node| node.attribute.iter())
        .filter_map(|attribute| attribute.g.as_ref())
}

fn rewrite_graph(graph: &mut GraphProto, shape_of: ShapeOf, half: &BTreeSet<String>) -> usize {
    let mut rewritten = 0;
    for node in graph.node.iter_mut().filter(|node| node.op_type == "If") {
        for attribute in &mut node.attribute {
            if let Some(branch) = attribute.g.as_mut() {
                while let Some(found) = find_pattern(branch, shape_of) {
                    let in_half = half.contains(&found.table);
                    apply(branch, &found, rewritten, in_half);
                    rewritten += 1;
                }
            }
        }
    }
    let mut serial = 0;
    for table in half {
        cast_gathers(graph, table, &mut serial);
    }
    rewritten
}

fn is_plain_transpose(node: &NodeProto) -> bool {
    node.attribute
        .iter()
        .all(|attribute| attribute.name != "perm" || attribute.ints.as_slice() == [1, 0])
}

fn find_pattern(branch: &GraphProto, shape_of: ShapeOf) -> Option<Match> {
    for (transpose, node) in branch.node.iter().enumerate() {
        let [table_name] = node.input.as_slice() else {
            continue;
        };
        let [transposed] = node.output.as_slice() else {
            continue;
        };
        if node.op_type != "Transpose" || !is_plain_transpose(node) {
            continue;
        }
        let Some(shape) = shape_of(table_name) else {
            continue;
        };
        let gathers_table = branch
            .node
            .iter()
            .any(|other| other.op_type == "Gather" && other.input.first() == Some(table_name));
        if !gathers_table {
            continue;
        }
        let mut consumers = branch
            .node
            .iter()
            .enumerate()
            .filter(|(_, other)| other.input.contains(transposed));
        let (Some((matmul, consumer)), None) = (consumers.next(), consumers.next()) else {
            continue;
        };
        let is_head = consumer.op_type == "MatMul"
            && consumer.input.get(1) == Some(transposed)
            && consumer.input.first() != Some(transposed)
            && consumer.output.len() == 1;
        let leaves_branch = branch
            .output
            .iter()
            .any(|output| output.name == *transposed);
        if is_head && !leaves_branch {
            return Some(Match {
                transpose,
                matmul,
                table: table_name.clone(),
                shape,
            });
        }
    }
    None
}

fn node(
    op_type: &str,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attributes: Vec<AttributeProto>,
) -> NodeProto {
    let mut node = NodeProto::new();
    op_type.clone_into(&mut node.op_type);
    name.clone_into(&mut node.name);
    node.input = inputs.iter().map(|input| (*input).to_owned()).collect();
    node.output = outputs.iter().map(|output| (*output).to_owned()).collect();
    node.attribute = attributes;
    node
}

fn int_attribute(name: &str, value: i64) -> AttributeProto {
    let mut attribute = AttributeProto::new();
    name.clone_into(&mut attribute.name);
    attribute.type_ = EnumOrUnknown::new(AttributeType::INT);
    attribute.i = value;
    attribute
}

fn cast(name: &str, input: &str, output: &str, to: DataType) -> NodeProto {
    node(
        "Cast",
        name,
        &[input],
        &[output],
        vec![int_attribute("to", i64::from(to.value()))],
    )
}

fn constant_ints(output: &str, values: &[i64]) -> NodeProto {
    let mut tensor = TensorProto::new();
    tensor.name = format!("{output}_value");
    tensor.data_type = DataType::INT64.value();
    tensor.dims = vec![i64::try_from(values.len()).unwrap_or(i64::MAX)];
    tensor.int64_data = values.to_vec();
    let mut attribute = AttributeProto::new();
    "value".clone_into(&mut attribute.name);
    attribute.type_ = EnumOrUnknown::new(AttributeType::TENSOR);
    attribute.t = MessageField::some(tensor);
    node("Constant", output, &[], &[output], vec![attribute])
}

fn apply(branch: &mut GraphProto, found: &Match, serial: usize, half: bool) {
    let Some(head) = branch.node.get(found.matmul) else {
        return;
    };
    let (Some(hidden), Some(logits)) = (head.input.first(), head.output.first()) else {
        return;
    };
    let (hidden, logits) = (hidden.clone(), logits.clone());
    let name = |part: &str| format!("{HEAD_PREFIX}{serial}_{part}");
    let (flat_shape, flat, flat_logits) =
        (name("flat_shape"), name("hidden_2d"), name("logits_2d"));
    let (hidden_shape, lead, zero, last, vocabulary, logits_shape) = (
        name("hidden_shape"),
        name("lead_dims"),
        name("zero"),
        name("minus_one"),
        name("vocabulary"),
        name("logits_shape"),
    );
    let (hidden_half, logits_half) = (name("hidden_half"), name("logits_half"));
    let (head_input, unflatten_output) = if half {
        (&hidden_half, &logits_half)
    } else {
        (&hidden, &logits)
    };
    let mut replacement = Vec::new();
    if half {
        replacement.push(cast(
            &name("hidden_to_half"),
            &hidden,
            &hidden_half,
            DataType::FLOAT16,
        ));
    }
    replacement.extend([
        constant_ints(&flat_shape, &[-1, found.shape.width]),
        node(
            "Reshape",
            &name("flatten"),
            &[head_input, &flat_shape],
            &[&flat],
            vec![],
        ),
        node(
            "Gemm",
            &name("lm_head"),
            &[&flat, &found.table],
            &[&flat_logits],
            vec![int_attribute("transB", 1)],
        ),
        node(
            "Shape",
            &name("shape"),
            &[&hidden],
            &[&hidden_shape],
            vec![],
        ),
        constant_ints(&zero, &[0]),
        constant_ints(&last, &[-1]),
        constant_ints(&vocabulary, &[found.shape.vocabulary]),
        node(
            "Slice",
            &name("lead"),
            &[&hidden_shape, &zero, &last],
            &[&lead],
            vec![],
        ),
        node(
            "Concat",
            &name("concat"),
            &[&lead, &vocabulary],
            &[&logits_shape],
            vec![int_attribute("axis", 0)],
        ),
        node(
            "Reshape",
            &name("unflatten"),
            &[&flat_logits, &logits_shape],
            &[unflatten_output],
            vec![],
        ),
    ]);
    if half {
        replacement.push(cast(
            &name("logits_to_full"),
            &logits_half,
            &logits,
            DataType::FLOAT,
        ));
    }
    branch.node.splice(found.matmul..=found.matmul, replacement);
    branch.node.remove(found.transpose);
}

fn subgraphs(node: &mut NodeProto) -> impl Iterator<Item = &mut GraphProto> {
    node.attribute.iter_mut().flat_map(|attribute| {
        attribute
            .g
            .as_mut()
            .into_iter()
            .chain(attribute.graphs.iter_mut())
    })
}

/// Splits every `Gather` of `table` in the graph and its subgraphs into a half-precision `Gather`
/// followed by a `Cast` that produces the original output.
fn cast_gathers(graph: &mut GraphProto, table: &str, serial: &mut usize) {
    let mut index = 0;
    while let Some(current) = graph.node.get_mut(index) {
        for subgraph in subgraphs(current) {
            cast_gathers(subgraph, table, serial);
        }
        let reads_table =
            current.op_type == "Gather" && current.input.first().map(String::as_str) == Some(table);
        if let (true, Some(output)) = (reads_table, current.output.first().cloned()) {
            let rows = format!("purr_tied_rows_{serial}_half");
            let cast_name = format!("purr_tied_rows_{serial}_to_full");
            *serial += 1;
            if let Some(slot) = current.output.first_mut() {
                slot.clone_from(&rows);
            }
            graph
                .node
                .insert(index + 1, cast(&cast_name, &rows, &output, DataType::FLOAT));
            index += 1;
        }
        index += 1;
    }
}

fn reads_as_half(node: &NodeProto, position: usize) -> bool {
    match (node.op_type.as_str(), position) {
        ("Gather", 0) => true,
        ("Gemm", 1) => node.name.starts_with(HEAD_PREFIX),
        _ => false,
    }
}

/// Whether the only readers of `table` in the graph and its subgraphs are the half-precision
/// lookups and heads this module writes, and no graph returns it.
fn only_gathers_read(graph: &GraphProto, table: &str) -> bool {
    if graph.output.iter().any(|output| output.name == table) {
        return false;
    }
    graph.node.iter().all(|node| {
        let readers_fit = node
            .input
            .iter()
            .enumerate()
            .all(|(position, input)| input != table || reads_as_half(node, position));
        readers_fit
            && node
                .attribute
                .iter()
                .flat_map(|attribute| attribute.g.iter().chain(attribute.graphs.iter()))
                .all(|subgraph| only_gathers_read(subgraph, table))
    })
}
