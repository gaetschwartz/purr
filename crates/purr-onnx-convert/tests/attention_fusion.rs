mod common;

use common::TestResult;
use common::encoder::{Breakage, Encoder, HEADS, Precision};
use protobuf::Message;
use purr_onnx_convert::convert::{Options, TablePrecision, convert};
use purr_onnx_convert::proto::onnx_proto3::{GraphProto, ModelProto, NodeProto};

struct Fused {
    blocks: usize,
    model: ModelProto,
}

fn run(encoder: &Encoder) -> Result<Fused, Box<dyn std::error::Error>> {
    let (mut data, mut graph) = (Vec::new(), Vec::new());
    let summary = convert(
        &mut encoder.bytes()?.as_slice(),
        &mut data,
        &mut graph,
        &Options {
            data_location: "encoder.onnx_data",
            table: TablePrecision::Full,
        },
    )?;
    Ok(Fused {
        blocks: summary.fused_attention_blocks,
        model: ModelProto::parse_from_bytes(&graph)?,
    })
}

fn op_types(graph: &GraphProto) -> Vec<&str> {
    graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect()
}

fn fused_nodes(model: &ModelProto) -> Vec<&NodeProto> {
    model
        .graph
        .node
        .iter()
        .filter(|node| node.op_type == "MultiHeadAttention")
        .collect()
}

fn attribute_f(node: &NodeProto, name: &str) -> Option<f32> {
    node.attribute
        .iter()
        .find(|attribute| attribute.name == name)
        .map(|attribute| attribute.f)
}

fn attribute_i(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute
        .iter()
        .find(|attribute| attribute.name == name)
        .map(|attribute| attribute.i)
}

fn unchanged(encoder: &Encoder) -> TestResult {
    let fused = run(encoder)?;
    assert_eq!(fused.blocks, 0);
    assert_eq!(
        op_types(&fused.model.graph),
        op_types(&encoder.model().graph)
    );
    assert_eq!(fused.model.opset_import, encoder.model().opset_import,);
    Ok(())
}

#[test]
fn every_attention_block_becomes_one_node_fed_by_the_projections() -> TestResult {
    let encoder = Encoder::standard();
    let fused = run(&encoder)?;
    assert_eq!(fused.blocks, encoder.layers);
    let nodes = fused_nodes(&fused.model);
    assert_eq!(nodes.len(), encoder.layers);
    for (layer, node) in nodes.iter().enumerate() {
        assert_eq!(node.domain, "com.microsoft");
        assert_eq!(
            node.input,
            [
                format!("l{layer}_q_add"),
                format!("l{layer}_k_mm"),
                format!("l{layer}_v_add"),
            ]
        );
        assert_eq!(node.output, [format!("l{layer}_merged")]);
        assert_eq!(attribute_i(node, "num_heads"), Some(HEADS));
    }
    let remaining = op_types(&fused.model.graph);
    let leftovers = remaining
        .iter()
        .filter(|op| {
            matches!(
                **op,
                "Softmax" | "Transpose" | "Concat" | "Reshape" | "Shape" | "Gather"
            )
        })
        .count();
    assert_eq!(leftovers, 0);
    let domains: Vec<(&str, i64)> = fused
        .model
        .opset_import
        .iter()
        .map(|opset| (opset.domain.as_str(), opset.version))
        .collect();
    assert_eq!(domains, [("", 14), ("com.microsoft", 1)]);
    Ok(())
}

#[test]
fn the_scale_of_the_export_is_the_default_of_the_fused_node() -> TestResult {
    let mut encoder = Encoder::standard();
    encoder.scale = 0.5;
    let fused = run(&encoder)?;
    assert_eq!(fused.blocks, encoder.layers);
    assert!(
        fused_nodes(&fused.model)
            .iter()
            .all(|node| attribute_f(node, "scale").is_none())
    );
    Ok(())
}

#[test]
fn half_precision_scales_are_read_too() -> TestResult {
    let mut encoder = Encoder::standard();
    encoder.precision = Precision::Half;
    let fused = run(&encoder)?;
    assert_eq!(fused.blocks, encoder.layers);
    Ok(())
}

#[test]
fn a_scale_that_is_not_the_default_is_passed_on() -> TestResult {
    let mut encoder = Encoder::standard();
    encoder.scale = 0.25;
    let fused = run(&encoder)?;
    assert_eq!(fused.blocks, encoder.layers);
    let scales: Vec<Option<f32>> = fused_nodes(&fused.model)
        .iter()
        .map(|node| attribute_f(node, "scale"))
        .collect();
    assert_eq!(scales, vec![Some(0.25); encoder.layers]);
    Ok(())
}

#[test]
fn a_softmax_over_another_axis_is_not_attention() -> TestResult {
    let mut encoder = Encoder::standard();
    encoder.breakage = Breakage::SoftmaxOverFirstAxis;
    unchanged(&encoder)
}

#[test]
fn heads_that_disagree_between_the_projections_are_not_fused() -> TestResult {
    let mut encoder = Encoder::standard();
    encoder.breakage = Breakage::ValueSplitHeads;
    unchanged(&encoder)
}

#[test]
fn a_merged_width_that_is_not_heads_times_size_is_not_fused() -> TestResult {
    let mut encoder = Encoder::standard();
    encoder.breakage = Breakage::MergeWidth;
    unchanged(&encoder)
}

#[test]
fn a_block_whose_heads_are_not_merged_back_is_not_fused() -> TestResult {
    let mut encoder = Encoder::standard();
    encoder.breakage = Breakage::NoFinalMerge;
    unchanged(&encoder)
}

#[test]
fn a_tensor_that_a_subgraph_reads_keeps_what_computes_it() -> TestResult {
    let mut encoder = Encoder::standard();
    encoder.layers = 1;
    encoder.reads_scores_in_subgraph = true;
    let fused = run(&encoder)?;
    assert_eq!(fused.blocks, 1);
    let remaining = op_types(&fused.model.graph);
    assert_eq!(remaining.iter().filter(|op| **op == "Softmax").count(), 0);
    let scores = fused
        .model
        .graph
        .node
        .iter()
        .filter(|node| node.output.iter().any(|output| output == "l0_scores"))
        .count();
    assert_eq!(scores, 1);
    assert_eq!(remaining.iter().filter(|op| **op == "If").count(), 1);
    Ok(())
}

#[test]
fn heads_kept_as_an_axis_are_fused_too() -> TestResult {
    let mut encoder = Encoder::standard();
    encoder.apart = true;
    let fused = run(&encoder)?;
    assert_eq!(fused.blocks, encoder.layers);
    let nodes = fused_nodes(&fused.model);
    assert_eq!(nodes.len(), encoder.layers);
    assert_eq!(nodes[0].input, ["l0_q_add", "l0_k_mm", "l0_v_add"]);
    Ok(())
}
