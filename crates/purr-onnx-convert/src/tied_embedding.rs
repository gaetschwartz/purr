//! Rewrites the language-model head of a merged decoder that shares its embedding table.
//!
//! Each `If` branch transposes the table for the output `MatMul`, and ONNX Runtime folds that
//! `Transpose` into a copy of the table per branch. A `Gemm` with `transB` reads the table as it
//! is, so nothing is left to fold.

use crate::proto::onnx_proto3::attribute_proto::AttributeType;
use crate::proto::onnx_proto3::tensor_proto::DataType;
use crate::proto::onnx_proto3::{AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto};
use protobuf::{Enum as _, EnumOrUnknown, MessageField};
use std::collections::BTreeMap;

const FIRST_OPSET_WITH_OPTIONAL_GEMM_BIAS: i64 = 11;

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

/// Applies the rewrite to every `If` branch that has the pattern, returning how many it changed.
pub fn rewrite(model: &mut ModelProto) -> usize {
    let opset = model
        .opset_import
        .iter()
        .find(|opset| opset.domain.is_empty() || opset.domain == "ai.onnx")
        .map_or(0, |opset| opset.version);
    if opset < FIRST_OPSET_WITH_OPTIONAL_GEMM_BIAS {
        return 0;
    }
    let Some(graph) = model.graph.as_mut() else {
        return 0;
    };
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

    let mut rewritten = 0;
    for node in graph.node.iter_mut().filter(|node| node.op_type == "If") {
        for attribute in &mut node.attribute {
            if let Some(branch) = attribute.g.as_mut() {
                while let Some(found) = find_pattern(branch, &tables) {
                    apply(branch, &found, rewritten);
                    rewritten += 1;
                }
            }
        }
    }
    rewritten
}

fn is_plain_transpose(node: &NodeProto) -> bool {
    node.attribute
        .iter()
        .all(|attribute| attribute.name != "perm" || attribute.ints.as_slice() == [1, 0])
}

fn find_pattern(branch: &GraphProto, tables: &BTreeMap<String, Table>) -> Option<Match> {
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
        let Some(shape) = tables.get(table_name) else {
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
                shape: *shape,
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

fn apply(branch: &mut GraphProto, found: &Match, serial: usize) {
    let Some(head) = branch.node.get(found.matmul) else {
        return;
    };
    let (Some(hidden), Some(logits)) = (head.input.first(), head.output.first()) else {
        return;
    };
    let (hidden, logits) = (hidden.clone(), logits.clone());
    let name = |part: &str| format!("purr_tied_{serial}_{part}");
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
    let replacement = vec![
        constant_ints(&flat_shape, &[-1, found.shape.width]),
        node(
            "Reshape",
            &name("flatten"),
            &[&hidden, &flat_shape],
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
            &[&logits],
            vec![],
        ),
    ];
    branch.node.splice(found.matmul..=found.matmul, replacement);
    branch.node.remove(found.transpose);
}
