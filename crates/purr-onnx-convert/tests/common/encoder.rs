use super::{dim, node, tensor};
use protobuf::{EnumOrUnknown, Message, MessageField};
use purr_onnx_convert::proto::onnx_proto3::attribute_proto::AttributeType;
use purr_onnx_convert::proto::onnx_proto3::tensor_proto::DataType;
use purr_onnx_convert::proto::onnx_proto3::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    ValueInfoProto,
};

pub const HEADS: i64 = 2;
pub const HEAD_SIZE: i64 = 4;
pub const WIDTH: i64 = HEADS * HEAD_SIZE;

#[derive(Clone, Copy)]
pub enum Precision {
    Full,
    Half,
}

#[derive(Clone, Copy)]
pub enum Breakage {
    None,
    SoftmaxOverFirstAxis,
    MergeWidth,
    ValueSplitHeads,
    NoFinalMerge,
}

pub struct Encoder {
    pub layers: usize,
    pub scale: f32,
    pub precision: Precision,
    pub breakage: Breakage,
    pub reads_scores_in_subgraph: bool,
    pub apart: bool,
}

fn ints(name: &str, values: &[i64]) -> TensorProto {
    let mut tensor = tensor(name, &[dim(values.len())], Vec::new());
    tensor.data_type = DataType::INT64 as i32;
    tensor.int64_data = values.to_vec();
    tensor
}

fn int_attribute(name: &str, value: i64) -> AttributeProto {
    let mut attribute = AttributeProto::new();
    name.clone_into(&mut attribute.name);
    attribute.type_ = EnumOrUnknown::new(AttributeType::INT);
    attribute.i = value;
    attribute
}

fn perm(values: &[i64]) -> AttributeProto {
    let mut attribute = AttributeProto::new();
    "perm".clone_into(&mut attribute.name);
    attribute.type_ = EnumOrUnknown::new(AttributeType::INTS);
    attribute.ints = values.to_vec();
    attribute
}

fn with(mut node: NodeProto, attribute: AttributeProto) -> NodeProto {
    node.attribute.push(attribute);
    node
}

fn value_info(name: &str) -> ValueInfoProto {
    let mut info = ValueInfoProto::new();
    name.clone_into(&mut info.name);
    info
}

impl Encoder {
    pub fn standard() -> Self {
        Self {
            layers: 2,
            scale: 0.5,
            precision: Precision::Full,
            breakage: Breakage::None,
            reads_scores_in_subgraph: false,
            apart: false,
        }
    }

    fn scalar(&self) -> TensorProto {
        let mut scale = tensor("scale", &[], Vec::new());
        match self.precision {
            Precision::Full => scale.raw_data = self.scale.to_le_bytes().to_vec(),
            Precision::Half => {
                scale.data_type = DataType::FLOAT16 as i32;
                scale.raw_data = half::f16::from_f32(self.scale).to_le_bytes().to_vec();
            }
        }
        scale
    }

    fn layer(&self, graph: &mut GraphProto, layer: usize, input: &str) -> String {
        let name = |part: &str| format!("l{layer}_{part}");
        let heads_size = |value: i64| match self.breakage {
            Breakage::ValueSplitHeads => value + 1,
            Breakage::None
            | Breakage::SoftmaxOverFirstAxis
            | Breakage::MergeWidth
            | Breakage::NoFinalMerge => value,
        };
        for part in ["wq", "wk", "wv", "wo"] {
            graph.initializer.push(tensor(
                &name(part),
                &[WIDTH, WIDTH],
                vec![0; (WIDTH * WIDTH) as usize * 4],
            ));
        }
        for part in ["bq", "bv", "bo"] {
            graph
                .initializer
                .push(tensor(&name(part), &[WIDTH], vec![0; WIDTH as usize * 4]));
        }
        let batch = name("batch");
        let steps = name("steps");
        let dims = name("dims");
        graph.node.extend([
            node("Shape", &[input], &[&dims]),
            node("Gather", &[&dims, "zero"], &[&batch]),
            node("Gather", &[&dims, "one"], &[&steps]),
            node("MatMul", &[input, &name("wq")], &[&name("q_mm")]),
            node("Add", &[&name("bq"), &name("q_mm")], &[&name("q_add")]),
            node("Mul", &[&name("q_add"), "scale"], &[&name("q_scaled")]),
            node("MatMul", &[input, &name("wk")], &[&name("k_mm")]),
            node("MatMul", &[input, &name("wv")], &[&name("v_mm")]),
            node("Add", &[&name("bv"), &name("v_mm")], &[&name("v_add")]),
            node("Mul", &[&batch, "heads"], &[&name("batch_heads")]),
        ]);

        let split = |graph: &mut GraphProto, source: &str, tag: &str, heads: i64| {
            let (shape_a, shape_b) = (
                name(&format!("{tag}_shape_a")),
                name(&format!("{tag}_shape_b")),
            );
            let (a, t, merged) = (
                name(&format!("{tag}_a")),
                name(&format!("{tag}_t")),
                name(&format!("{tag}_b")),
            );
            graph.node.extend([
                node(
                    "Concat",
                    &[&batch, "minus_one", &format!("heads_{heads}"), "head_size"],
                    &[&shape_a],
                ),
                with(
                    node("Reshape", &[source, &shape_a], &[&a]),
                    int_attribute("allowzero", 0),
                ),
            ]);
            graph
                .node
                .push(with(node("Transpose", &[&a], &[&t]), perm(&[0, 2, 1, 3])));
            if self.apart {
                return (t, a);
            }
            graph.node.extend([
                node(
                    "Concat",
                    &[&name("batch_heads"), "minus_one", "head_size"],
                    &[&shape_b],
                ),
                node("Reshape", &[&t, &shape_b], &[&merged]),
            ]);
            (merged, a)
        };
        for extra in [HEADS, HEADS + 1] {
            graph
                .initializer
                .push(ints(&format!("heads_{extra}"), &[extra]));
        }
        let (query, _) = split(graph, &name("q_scaled"), "q", HEADS);
        let (key, key_split) = split(graph, &name("k_mm"), "k", HEADS);
        let (value, _) = split(graph, &name("v_add"), "v", heads_size(HEADS));
        let key_t = name("k_transposed");
        let (scores, probs, context) = (name("scores"), name("probs"), name("context"));
        let softmax_axis = match self.breakage {
            Breakage::SoftmaxOverFirstAxis => 0,
            Breakage::None
            | Breakage::MergeWidth
            | Breakage::ValueSplitHeads
            | Breakage::NoFinalMerge => -1,
        };
        graph.node.extend([
            if self.apart {
                with(
                    node("Transpose", &[&key_split], &[&key_t]),
                    perm(&[0, 2, 3, 1]),
                )
            } else {
                with(node("Transpose", &[&key], &[&key_t]), perm(&[0, 2, 1]))
            },
            node("MatMul", &[&query, &key_t], &[&scores]),
            with(
                node("Softmax", &[&scores], &[&probs]),
                int_attribute("axis", softmax_axis),
            ),
            node("MatMul", &[&probs, &value], &[&context]),
        ]);

        let (regroup_shape, regrouped, moved, merge_shape, merged) = (
            name("regroup_shape"),
            name("regrouped"),
            name("moved"),
            name("merge_shape"),
            name("merged"),
        );
        let width = match self.breakage {
            Breakage::MergeWidth => WIDTH + 1,
            Breakage::None
            | Breakage::SoftmaxOverFirstAxis
            | Breakage::ValueSplitHeads
            | Breakage::NoFinalMerge => WIDTH,
        };
        graph.initializer.push(ints(&name("width"), &[width]));
        if self.apart {
            graph.node.push(with(
                node("Transpose", &[&context], &[&moved]),
                perm(&[0, 2, 1, 3]),
            ));
        } else {
            graph.node.extend([
                node(
                    "Concat",
                    &[&batch, "heads_2", &steps, "head_size"],
                    &[&regroup_shape],
                ),
                node("Reshape", &[&context, &regroup_shape], &[&regrouped]),
                with(
                    node("Transpose", &[&regrouped], &[&moved]),
                    perm(&[0, 2, 1, 3]),
                ),
            ]);
        }
        graph.node.push(node(
            "Concat",
            &[&batch, &steps, &name("width")],
            &[&merge_shape],
        ));
        let attention = match self.breakage {
            Breakage::NoFinalMerge => moved,
            Breakage::None
            | Breakage::SoftmaxOverFirstAxis
            | Breakage::MergeWidth
            | Breakage::ValueSplitHeads => {
                graph
                    .node
                    .push(node("Reshape", &[&moved, &merge_shape], &[&merged]));
                merged
            }
        };
        let projected = name("projected");
        graph.node.extend([
            node("MatMul", &[&attention, &name("wo")], &[&name("o_mm")]),
            node("Add", &[&name("bo"), &name("o_mm")], &[&projected]),
            node("Add", &[input, &projected], &[&name("out")]),
        ]);
        if self.reads_scores_in_subgraph {
            let mut branch = GraphProto::new();
            branch.node = vec![node("Identity", &[&scores], &["branch_out"])];
            branch.output = vec![value_info("branch_out")];
            let mut attribute = AttributeProto::new();
            "then_branch".clone_into(&mut attribute.name);
            attribute.type_ = EnumOrUnknown::new(AttributeType::GRAPH);
            attribute.g = MessageField::some(branch);
            let mut gate = node("If", &["condition"], &[&name("probe")]);
            gate.attribute.push(attribute);
            graph.node.push(gate);
            graph.output.push(value_info(&name("probe")));
        }
        name("out")
    }

    pub fn model(&self) -> ModelProto {
        let mut graph = GraphProto::new();
        "encoder".clone_into(&mut graph.name);
        graph.input = vec![value_info("x"), value_info("condition")];
        let mut zero = ints("zero", &[0]);
        zero.dims = vec![];
        let mut one = ints("one", &[1]);
        one.dims = vec![];
        graph.initializer.extend([
            zero,
            one,
            ints("minus_one", &[-1]),
            ints("head_size", &[HEAD_SIZE]),
            ints("heads", &[HEADS]),
            self.scalar(),
        ]);
        let mut current = "x".to_owned();
        for layer in 0..self.layers {
            current = self.layer(&mut graph, layer, &current);
        }
        graph.output.insert(0, value_info(&current));
        let mut opset = OperatorSetIdProto::new();
        opset.version = 14;
        let mut model = ModelProto::new();
        model.ir_version = 7;
        model.opset_import = vec![opset];
        model.graph = MessageField::some(graph);
        model
    }

    pub fn bytes(&self) -> Result<Vec<u8>, protobuf::Error> {
        self.model().write_to_bytes()
    }
}
