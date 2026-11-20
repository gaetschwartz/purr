//! Replaces the self-attention of an encoder, written as separate operators, by one
//! `com.microsoft.MultiHeadAttention` node.
//!
//! The export computes the scores `q·kᵀ` and their softmax as two full `[batch * heads, S, S]`
//! tensors per layer, which ONNX Runtime keeps in memory. The fused node does not materialize
//! them. A block is recognised by the operators and edges it is made of and by the constants that
//! decide its meaning (the head count, the head size and the scale), never by node names, and a
//! graph without such blocks passes through untouched.

use crate::proto::onnx_proto3::attribute_proto::AttributeType;
use crate::proto::onnx_proto3::tensor_proto::DataType;
use crate::proto::onnx_proto3::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
};
use half::f16;
use protobuf::{Enum as _, EnumOrUnknown};
use std::collections::{HashMap, HashSet};

const MICROSOFT_DOMAIN: &str = "com.microsoft";
const MICROSOFT_OPSET: i64 = 1;
const FUSED_PREFIX: &str = "purr_mha_";
const SCALE_TOLERANCE: f64 = 2e-3;
const SPLIT_HEADS: [i64; 4] = [0, 2, 1, 3];
const KEY_TRANSPOSE: [i64; 3] = [0, 2, 1];
const KEY_TRANSPOSE_SPLIT: [i64; 4] = [0, 2, 3, 1];

/// What the shape of a `Reshape` is known to hold; dynamic entries are `None`.
type KnownInts = Vec<Option<i64>>;

struct Heads {
    count: i64,
    size: i64,
}

struct Split<'a> {
    source: &'a str,
    heads: Heads,
}

struct Block<'a> {
    merged: usize,
    query: &'a str,
    key: &'a str,
    value: &'a str,
    heads: Heads,
    scale: f64,
}

struct Index<'a> {
    graph: &'a GraphProto,
    producer: HashMap<&'a str, usize>,
    constants: HashMap<&'a str, &'a TensorProto>,
}

impl<'a> Index<'a> {
    fn new(graph: &'a GraphProto) -> Self {
        let mut producer = HashMap::new();
        let mut constants = HashMap::new();
        for tensor in &graph.initializer {
            constants.insert(tensor.name.as_str(), tensor);
        }
        for (position, node) in graph.node.iter().enumerate() {
            for output in &node.output {
                producer.insert(output.as_str(), position);
            }
            if node.op_type == "Constant"
                && let (Some(output), Some(tensor)) = (
                    node.output.first(),
                    node.attribute
                        .iter()
                        .find(|attribute| attribute.name == "value")
                        .and_then(|attribute| attribute.t.as_ref()),
                )
            {
                constants.insert(output.as_str(), tensor);
            }
        }
        Self {
            graph,
            producer,
            constants,
        }
    }

    fn node_of(&self, tensor: &str) -> Option<&'a NodeProto> {
        self.producer
            .get(tensor)
            .and_then(|position| self.graph.node.get(*position))
    }

    fn producing(&self, tensor: &str, op_type: &str) -> Option<&'a NodeProto> {
        self.node_of(tensor).filter(|node| node.op_type == op_type)
    }

    fn ints(&self, tensor: &str) -> Option<Vec<i64>> {
        self.constants.get(tensor).and_then(|t| tensor_ints(t))
    }

    fn scalar(&self, tensor: &str) -> Option<f64> {
        self.constants.get(tensor).and_then(|t| tensor_scalar(t))
    }

    /// The entries of the shape `tensor`, where it is a constant or a `Concat` of some.
    fn shape(&self, tensor: &str) -> Option<KnownInts> {
        if let Some(values) = self.ints(tensor) {
            return Some(values.into_iter().map(Some).collect());
        }
        let concat = self.producing(tensor, "Concat")?;
        concat
            .input
            .iter()
            .map(|input| match self.ints(input).as_deref() {
                Some([value]) => Some(Some(*value)),
                Some(_) => None,
                None => Some(None),
            })
            .collect()
    }

    fn reshaped(&self, tensor: &str) -> Option<(&'a NodeProto, KnownInts)> {
        let node = self.producing(tensor, "Reshape")?;
        let [_, shape] = node.input.as_slice() else {
            return None;
        };
        Some((node, self.shape(shape)?))
    }

    fn transposed(&self, tensor: &str, perm: &[i64]) -> Option<&'a NodeProto> {
        self.producing(tensor, "Transpose")
            .filter(|node| has_perm(node, perm))
    }

    /// Either layout of heads: folded into the batch or kept as an axis.
    fn split_heads(&self, tensor: &str) -> Option<Split<'a>> {
        self.split_heads_into_batch(tensor)
            .or_else(|| self.split_heads_apart(tensor))
    }

    /// `[batch, S, heads * size]` reshaped to `[batch, S, heads, size]` and transposed to
    /// `[batch, heads, S, size]`.
    fn split_heads_apart(&self, tensor: &str) -> Option<Split<'a>> {
        let transpose = self.transposed(tensor, &SPLIT_HEADS)?;
        self.heads_axis(transpose.input.first()?)
    }

    /// `[batch, S, heads * size]` reshaped to `[batch, S, heads, size]`.
    fn heads_axis(&self, tensor: &str) -> Option<Split<'a>> {
        let (split, split_shape) = self.reshaped(tensor)?;
        let [.., Some(count), Some(size)] = split_shape.as_slice() else {
            return None;
        };
        (*count > 0 && *size > 0).then(|| Split {
            source: split.input.first().map_or("", String::as_str),
            heads: Heads {
                count: *count,
                size: *size,
            },
        })
    }

    /// The key of a block, laid out for the product with the query: transposed to
    /// `[batch, heads, size, S]`, or split like the others and transposed to `[.., size, S]`.
    fn split_key(&self, tensor: &str) -> Option<Split<'a>> {
        if let Some(transpose) = self.transposed(tensor, &KEY_TRANSPOSE_SPLIT) {
            return self.heads_axis(transpose.input.first()?);
        }
        let transpose = self.transposed(tensor, &KEY_TRANSPOSE)?;
        self.split_heads_into_batch(transpose.input.first()?)
    }

    /// `[batch, S, heads * size]` reshaped to `[batch, S, heads, size]`, transposed to put the
    /// heads first and reshaped to `[batch * heads, S, size]`.
    fn split_heads_into_batch(&self, tensor: &str) -> Option<Split<'a>> {
        let (merge, merge_shape) = self.reshaped(tensor)?;
        let transpose = self.transposed(merge.input.first()?, &SPLIT_HEADS)?;
        let (split, split_shape) = self.reshaped(transpose.input.first()?)?;
        let [.., Some(count), Some(size)] = split_shape.as_slice() else {
            return None;
        };
        let [.., Some(merged_size)] = merge_shape.as_slice() else {
            return None;
        };
        (merged_size == size && *count > 0 && *size > 0).then(|| Split {
            source: split.input.first().map_or("", String::as_str),
            heads: Heads {
                count: *count,
                size: *size,
            },
        })
    }

    /// The block whose softmax is `softmax`, with the node that holds its merged heads.
    fn block(&self, softmax: &NodeProto) -> Option<Block<'a>> {
        let [scores_name] = softmax.input.as_slice() else {
            return None;
        };
        let [probabilities] = softmax.output.as_slice() else {
            return None;
        };
        let last_axis = softmax
            .attribute
            .iter()
            .any(|attribute| attribute.name == "axis" && attribute.i == -1);
        if !last_axis {
            return None;
        }
        let scores = self.producing(scores_name, "MatMul")?;
        let [query_side, key_side] = scores.input.as_slice() else {
            return None;
        };
        let weighted = self.sole_consumer(probabilities, "MatMul", 0)?;
        let [_, value_side] = weighted.input.as_slice() else {
            return None;
        };

        let scaled = self.split_heads(query_side)?;
        let scale_node = self.producing(scaled.source, "Mul")?;
        let (query, scale) = scale_node
            .input
            .iter()
            .enumerate()
            .find_map(|(at, input)| {
                let other = scale_node.input.get(1 - at)?;
                Some((other.as_str(), self.scalar(input)?))
            })?;
        let key = self.split_key(key_side)?;
        let value = self.split_heads(value_side)?;
        let heads = scaled.heads;
        let agree = |other: &Heads| other.count == heads.count && other.size == heads.size;
        if !agree(&key.heads) || !agree(&value.heads) {
            return None;
        }

        let merge_node = self.merge_heads(weighted.output.first()?, &heads)?;
        let merged = self
            .graph
            .node
            .iter()
            .position(|node| std::ptr::eq(node, merge_node))?;
        Some(Block {
            merged,
            query,
            key: key.source,
            value: value.source,
            heads,
            scale,
        })
    }

    /// The `Reshape` that merges the heads of `context` back into `[batch, S, heads * size]`,
    /// after regrouping them from the layout they were split into.
    fn merge_heads(&self, context: &str, heads: &Heads) -> Option<&'a NodeProto> {
        let moved = match self.sole_consumer(context, "Reshape", 0) {
            Some(regroup) => {
                let regrouped = self.shape(regroup.input.get(1)?)?;
                let [_, Some(count), _, Some(size)] = regrouped.as_slice() else {
                    return None;
                };
                if *count != heads.count || *size != heads.size {
                    return None;
                }
                self.sole_consumer(regroup.output.first()?, "Transpose", 0)?
            }
            None => self.sole_consumer(context, "Transpose", 0)?,
        };
        if !has_perm(moved, &SPLIT_HEADS) {
            return None;
        }
        let merge_node = self.sole_consumer(moved.output.first()?, "Reshape", 0)?;
        let merge_shape = self.shape(merge_node.input.get(1)?)?;
        let [.., Some(width)] = merge_shape.as_slice() else {
            return None;
        };
        (Some(*width) == heads.count.checked_mul(heads.size)).then_some(merge_node)
    }

    fn sole_consumer(&self, tensor: &str, op_type: &str, position: usize) -> Option<&'a NodeProto> {
        let mut readers = self
            .graph
            .node
            .iter()
            .filter(|node| node.input.iter().any(|input| input == tensor));
        let (Some(reader), None) = (readers.next(), readers.next()) else {
            return None;
        };
        let reads_there = reader.input.get(position).map(String::as_str) == Some(tensor);
        (reader.op_type == op_type && reads_there).then_some(reader)
    }
}

fn has_perm(node: &NodeProto, perm: &[i64]) -> bool {
    node.attribute
        .iter()
        .any(|attribute| attribute.name == "perm" && attribute.ints.as_slice() == perm)
}

fn tensor_ints(tensor: &TensorProto) -> Option<Vec<i64>> {
    if tensor.data_type == DataType::INT64.value() {
        if !tensor.int64_data.is_empty() {
            return Some(tensor.int64_data.clone());
        }
        let (words, rest) = tensor.raw_data.as_chunks::<8>();
        return rest
            .is_empty()
            .then(|| words.iter().map(|word| i64::from_le_bytes(*word)).collect());
    }
    if tensor.data_type == DataType::INT32.value() {
        if !tensor.int32_data.is_empty() {
            return Some(tensor.int32_data.iter().map(|v| i64::from(*v)).collect());
        }
        let (words, rest) = tensor.raw_data.as_chunks::<4>();
        return rest.is_empty().then(|| {
            words
                .iter()
                .map(|word| i64::from(i32::from_le_bytes(*word)))
                .collect()
        });
    }
    None
}

fn tensor_scalar(tensor: &TensorProto) -> Option<f64> {
    let one_element = tensor.dims.iter().product::<i64>() == 1;
    if !one_element {
        return None;
    }
    if tensor.data_type == DataType::FLOAT.value() {
        if let Some(value) = tensor.float_data.first() {
            return Some(f64::from(*value));
        }
        let (words, rest) = tensor.raw_data.as_chunks::<4>();
        return match (words, rest) {
            ([word], []) => Some(f64::from(f32::from_le_bytes(*word))),
            _ => None,
        };
    }
    if tensor.data_type == DataType::FLOAT16.value() {
        if let Some(bits) = tensor.int32_data.first() {
            return u16::try_from(*bits)
                .ok()
                .map(|bits| f64::from(f16::from_bits(bits)));
        }
        let (words, rest) = tensor.raw_data.as_chunks::<2>();
        return match (words, rest) {
            ([word], []) => Some(f64::from(f16::from_le_bytes(*word))),
            _ => None,
        };
    }
    None
}

fn attribute(name: &str, kind: AttributeType) -> AttributeProto {
    let mut attribute = AttributeProto::new();
    name.clone_into(&mut attribute.name);
    attribute.type_ = EnumOrUnknown::new(kind);
    attribute
}

fn fused_node(block: &Block, merged_output: &str, serial: usize) -> NodeProto {
    let mut node = NodeProto::new();
    "MultiHeadAttention".clone_into(&mut node.op_type);
    node.name = format!("{FUSED_PREFIX}{serial}");
    MICROSOFT_DOMAIN.clone_into(&mut node.domain);
    node.input = vec![
        block.query.to_owned(),
        block.key.to_owned(),
        block.value.to_owned(),
    ];
    node.output = vec![merged_output.to_owned()];
    let mut count = attribute("num_heads", AttributeType::INT);
    count.i = block.heads.count;
    node.attribute.push(count);
    let default_scale = 1.0 / (block.heads.size as f64).sqrt();
    if (block.scale - default_scale).abs() > default_scale * SCALE_TOLERANCE {
        let mut scale = attribute("scale", AttributeType::FLOAT);
        scale.f = block.scale as f32;
        node.attribute.push(scale);
    }
    node
}

fn subgraph_reads(graph: &GraphProto, reads: &mut Vec<String>) {
    for node in &graph.node {
        reads.extend(node.input.iter().filter(|input| !input.is_empty()).cloned());
        for attribute in &node.attribute {
            for inner in attribute.g.iter().chain(&attribute.graphs) {
                subgraph_reads(inner, reads);
            }
        }
    }
    reads.extend(graph.output.iter().map(|output| output.name.clone()));
}

/// Drops the nodes whose outputs the graph outputs do not depend on, directly or through the
/// outer-scope tensors its subgraphs read. The order of the others is kept.
fn remove_unreachable(graph: &mut GraphProto) {
    let mut producers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (position, node) in graph.node.iter().enumerate() {
        for output in &node.output {
            producers.entry(output.as_str()).or_default().push(position);
        }
    }
    let mut pending: Vec<String> = graph.output.iter().map(|o| o.name.clone()).collect();
    let mut reachable = HashSet::new();
    while let Some(tensor) = pending.pop() {
        for position in producers.get(tensor.as_str()).into_iter().flatten() {
            if !reachable.insert(*position) {
                continue;
            }
            let Some(node) = graph.node.get(*position) else {
                continue;
            };
            pending.extend(node.input.iter().filter(|input| !input.is_empty()).cloned());
            for attribute in &node.attribute {
                for inner in attribute.g.iter().chain(&attribute.graphs) {
                    subgraph_reads(inner, &mut pending);
                }
            }
        }
    }
    let mut position = 0;
    graph.node.retain(|_| {
        let kept = reachable.contains(&position);
        position += 1;
        kept
    });
}

/// Fuses every attention block of the model's main graph, returning how many it fused.
pub fn fuse(model: &mut ModelProto) -> usize {
    let Some(graph) = model.graph.as_mut() else {
        return 0;
    };
    let blocks: Vec<(usize, NodeProto)> = {
        let index = Index::new(graph);
        let mut seen = HashSet::new();
        graph
            .node
            .iter()
            .filter(|node| node.op_type == "Softmax")
            .filter_map(|softmax| index.block(softmax))
            .filter(|block| seen.insert(block.merged))
            .enumerate()
            .filter_map(|(serial, block)| {
                let output = graph.node.get(block.merged)?.output.first()?;
                Some((block.merged, fused_node(&block, output, serial)))
            })
            .collect()
    };
    if blocks.is_empty() {
        return 0;
    }
    let fused = blocks.len();
    for (position, node) in blocks {
        if let Some(slot) = graph.node.get_mut(position) {
            *slot = node;
        }
    }
    remove_unreachable(graph);
    let has_domain = model
        .opset_import
        .iter()
        .any(|opset| opset.domain == MICROSOFT_DOMAIN);
    if !has_domain {
        let mut opset = OperatorSetIdProto::new();
        MICROSOFT_DOMAIN.clone_into(&mut opset.domain);
        opset.version = MICROSOFT_OPSET;
        model.opset_import.push(opset);
    }
    fused
}
