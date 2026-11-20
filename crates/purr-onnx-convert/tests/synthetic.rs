mod common;

use common::{Decoder, TestResult};
use protobuf::Message;
use purr_onnx_convert::convert::{DATA_ALIGNMENT, Options, Summary, convert};
use purr_onnx_convert::proto::onnx_proto3::tensor_proto::DataLocation;
use purr_onnx_convert::proto::onnx_proto3::{GraphProto, ModelProto};

const LOCATION: &str = "decoder.onnx_data";

struct Converted {
    summary: Summary,
    data: Vec<u8>,
    graph: Vec<u8>,
}

fn run(input: &[u8]) -> Result<Converted, Box<dyn std::error::Error>> {
    let (mut data, mut graph) = (Vec::new(), Vec::new());
    let summary = convert(
        &mut &*input,
        &mut data,
        &mut graph,
        &Options {
            data_location: LOCATION,
        },
    )?;
    Ok(Converted {
        summary,
        data,
        graph,
    })
}

fn op_types(graph: &GraphProto) -> Vec<&str> {
    graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect()
}

fn branches(model: &ModelProto) -> Vec<&GraphProto> {
    model
        .graph
        .node
        .iter()
        .filter(|node| node.op_type == "If")
        .flat_map(|node| node.attribute.iter().map(|attribute| &*attribute.g))
        .collect()
}

#[test]
fn big_initializers_move_to_aligned_data_and_small_ones_stay() -> TestResult {
    let decoder = Decoder::small();
    let original = decoder.model();
    let converted = run(&decoder.bytes()?)?;
    let model = ModelProto::parse_from_bytes(&converted.graph)?;

    assert_eq!(
        model.graph.initializer.len(),
        original.graph.initializer.len()
    );
    let mut external = 0;
    for (placed, source) in model
        .graph
        .initializer
        .iter()
        .zip(&original.graph.initializer)
    {
        assert_eq!(placed.name, source.name);
        assert_eq!(placed.dims, source.dims);
        if source.raw_data.len() < 1024 {
            assert_eq!(placed.raw_data, source.raw_data);
            assert_eq!(placed.data_location.enum_value(), Ok(DataLocation::DEFAULT));
            continue;
        }
        external += 1;
        assert_eq!(placed.raw_data, Vec::<u8>::new());
        assert_eq!(
            placed.data_location.enum_value(),
            Ok(DataLocation::EXTERNAL)
        );
        let entry = |key: &str| {
            placed
                .external_data
                .iter()
                .find(|entry| entry.key == key)
                .map(|entry| entry.value.clone())
        };
        assert_eq!(entry("location").as_deref(), Some(LOCATION));
        let offset: usize = entry("offset").ok_or("no offset")?.parse()?;
        let length: usize = entry("length").ok_or("no length")?.parse()?;
        assert_eq!(offset as u64 % DATA_ALIGNMENT, 0);
        assert_eq!(length, source.raw_data.len());
        assert_eq!(
            converted.data.get(offset..offset + length),
            Some(source.raw_data.as_slice())
        );
    }
    assert_eq!(external, 4);
    assert_eq!(converted.summary.external_tensors, external);
    assert_eq!(converted.summary.data_bytes, converted.data.len() as u64);
    assert_eq!(converted.summary.graph_bytes, converted.graph.len() as u64);
    Ok(())
}

#[test]
fn tied_embedding_loses_its_transpose_in_every_branch() -> TestResult {
    let converted = run(&Decoder::small().bytes()?)?;
    assert_eq!(converted.summary.tied_embedding_branches, 2);
    let model = ModelProto::parse_from_bytes(&converted.graph)?;
    let branches = branches(&model);
    assert_eq!(branches.len(), 2);
    for branch in branches {
        let ops = op_types(branch);
        assert!(!ops.contains(&"Transpose"));
        assert!(!ops.contains(&"MatMul"));
        assert_eq!(ops.iter().filter(|op| **op == "Gemm").count(), 1);
        let gemm = branch
            .node
            .iter()
            .find(|node| node.op_type == "Gemm")
            .ok_or("no Gemm")?;
        assert_eq!(gemm.input.get(1).map(String::as_str), Some("embedding"));
        assert_eq!(gemm.attribute.len(), 1);
        let transpose_b = gemm.attribute.first().ok_or("no attribute")?;
        assert_eq!((transpose_b.name.as_str(), transpose_b.i), ("transB", 1));
        assert!(branch.node.iter().any(|node| node.op_type == "Gather"));
        let last = branch.node.last().ok_or("empty branch")?;
        assert_eq!(
            (last.op_type.as_str(), last.output.clone()),
            ("Reshape", vec!["logits".to_owned()])
        );
    }
    Ok(())
}

#[test]
fn a_transpose_with_another_consumer_is_left_alone() -> TestResult {
    let decoder = Decoder {
        shared_transpose_in_else: true,
        ..Decoder::small()
    };
    let converted = run(&decoder.bytes()?)?;
    assert_eq!(converted.summary.tied_embedding_branches, 1);
    let model = ModelProto::parse_from_bytes(&converted.graph)?;
    let branches = branches(&model);
    assert_eq!(
        branches
            .iter()
            .map(|branch| op_types(branch).contains(&"Transpose"))
            .collect::<Vec<_>>(),
        [false, true]
    );
    Ok(())
}

#[test]
fn an_opset_without_optional_gemm_bias_is_left_alone() -> TestResult {
    let decoder = Decoder {
        opset: 9,
        ..Decoder::small()
    };
    let converted = run(&decoder.bytes()?)?;
    assert_eq!(converted.summary.tied_embedding_branches, 0);
    Ok(())
}

#[test]
fn a_model_without_the_pattern_only_changes_where_its_weights_live() -> TestResult {
    let mut model = Decoder::small().model();
    model.graph.mut_or_insert_default().node.clear();
    let converted = run(&model.write_to_bytes()?)?;
    assert_eq!(converted.summary.tied_embedding_branches, 0);
    let rewritten = ModelProto::parse_from_bytes(&converted.graph)?;
    assert_eq!(rewritten.graph.node.len(), 0);
    assert_eq!(rewritten.opset_import, model.opset_import);
    Ok(())
}

#[test]
fn the_same_model_always_converts_to_the_same_bytes() -> TestResult {
    let input = Decoder::small().bytes()?;
    let (first, second) = (run(&input)?, run(&input)?);
    assert_eq!(first.graph, second.graph);
    assert_eq!(first.data, second.data);
    Ok(())
}

#[test]
fn a_truncated_model_is_an_error() -> TestResult {
    let input = Decoder::small().bytes()?;
    let cut = input.get(..input.len() / 2).ok_or("empty input")?;
    assert!(run(cut).is_err());
    Ok(())
}

#[test]
fn a_model_without_a_graph_is_an_error() {
    assert!(run(&[]).is_err());
}
