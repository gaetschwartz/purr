#![allow(dead_code, reason = "each test binary uses a part of this module")]

pub mod counting_allocator;
pub mod encoder;

use protobuf::{EnumOrUnknown, Message, MessageField};
use purr_onnx_convert::proto::onnx_proto3::attribute_proto::AttributeType;
use purr_onnx_convert::proto::onnx_proto3::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
};

pub type TestResult = Result<(), Box<dyn std::error::Error>>;

pub const FLOAT: i32 = 1;

pub fn dim(size: usize) -> i64 {
    i64::try_from(size).unwrap_or(i64::MAX)
}

pub fn tensor(name: &str, dims: &[i64], raw: Vec<u8>) -> TensorProto {
    let mut tensor = TensorProto::new();
    name.clone_into(&mut tensor.name);
    tensor.dims = dims.to_vec();
    tensor.data_type = FLOAT;
    tensor.raw_data = raw;
    tensor
}

pub fn filler(name: &str, bytes: usize, seed: u8) -> TensorProto {
    let raw = (0..bytes)
        .map(|index| seed.wrapping_add((index % 251) as u8))
        .collect();
    tensor(name, &[dim(bytes / 4)], raw)
}

pub fn node(op_type: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    let mut node = NodeProto::new();
    op_type.clone_into(&mut node.op_type);
    node.input = inputs.iter().map(|name| (*name).to_owned()).collect();
    node.output = outputs.iter().map(|name| (*name).to_owned()).collect();
    node
}

fn branch(transposed_is_shared: bool) -> GraphProto {
    let mut graph = GraphProto::new();
    graph.node = vec![
        node("Gather", &["embedding", "ids"], &["embedded"]),
        node("Transpose", &["embedding"], &["embedding_t"]),
        node("MatMul", &["hidden", "embedding_t"], &["logits"]),
    ];
    if transposed_is_shared {
        graph
            .node
            .push(node("Identity", &["embedding_t"], &["copy"]));
    }
    graph
}

fn if_node(then_branch: GraphProto, else_branch: GraphProto) -> NodeProto {
    let mut node = node("If", &["cond"], &["logits"]);
    for (name, graph) in [("then_branch", then_branch), ("else_branch", else_branch)] {
        let mut attribute = AttributeProto::new();
        name.clone_into(&mut attribute.name);
        attribute.type_ = EnumOrUnknown::new(AttributeType::GRAPH);
        attribute.g = MessageField::some(graph);
        node.attribute.push(attribute);
    }
    node
}

pub const LEADING_VALUES: [f32; 4] = [1.0, 1.0 + 1.0 / 2048.0, 1.0 + 3.0 / 2048.0, -2.5];

pub struct Decoder {
    pub vocabulary: usize,
    pub width: usize,
    pub filler_tensors: usize,
    pub filler_bytes: usize,
    pub shared_transpose_in_else: bool,
    pub opset: i64,
    pub embedding_values: EmbeddingValues,
    pub extra_table_reader: bool,
}

#[derive(Clone, Copy)]
pub enum EmbeddingValues {
    Pattern,
    Floats { largest: f32 },
}

pub fn embedding_floats(count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            LEADING_VALUES
                .get(index)
                .map_or_else(|| (index as f32 * 0.37).sin() * 3.0, |value| *value)
        })
        .collect()
}

impl Decoder {
    pub fn small() -> Self {
        Self {
            vocabulary: 32,
            width: 16,
            filler_tensors: 3,
            filler_bytes: 5000,
            shared_transpose_in_else: false,
            opset: 14,
            embedding_values: EmbeddingValues::Pattern,
            extra_table_reader: false,
        }
    }

    pub fn model(&self) -> ModelProto {
        let embedding = filler("embedding", self.vocabulary * self.width * 4, 7);
        let mut embedding = embedding;
        embedding.dims = vec![dim(self.vocabulary), dim(self.width)];
        if let EmbeddingValues::Floats { largest } = self.embedding_values {
            let mut floats = embedding_floats(self.vocabulary * self.width);
            if let Some(last) = floats.last_mut() {
                *last = largest;
            }
            embedding.raw_data = floats
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
        }
        let mut graph = GraphProto::new();
        "decoder".clone_into(&mut graph.name);
        graph.initializer.push(embedding);
        graph
            .initializer
            .push(tensor("tiny", &[2], vec![1, 2, 3, 4, 5, 6, 7, 8]));
        for index in 0..self.filler_tensors {
            graph.initializer.push(filler(
                &format!("weight_{index}"),
                self.filler_bytes,
                index as u8,
            ));
        }
        let mut then_branch = branch(false);
        if self.extra_table_reader {
            then_branch
                .node
                .push(node("Identity", &["embedding"], &["table_copy"]));
        }
        graph.node = vec![if_node(then_branch, branch(self.shared_transpose_in_else))];
        let mut opset = OperatorSetIdProto::new();
        opset.version = self.opset;
        let mut model = ModelProto::new();
        model.ir_version = 9;
        "test".clone_into(&mut model.producer_name);
        model.opset_import = vec![opset];
        model.graph = MessageField::some(graph);
        model
    }

    pub fn bytes(&self) -> Result<Vec<u8>, protobuf::Error> {
        self.model().write_to_bytes()
    }
}
