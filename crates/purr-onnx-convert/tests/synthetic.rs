mod common;

use common::{Decoder, EmbeddingValues, LEADING_VALUES, TestResult};
use half::f16;
use protobuf::Message;
use purr_onnx_convert::convert::{DATA_ALIGNMENT, Options, Summary, TablePrecision, convert};
use purr_onnx_convert::error::ConvertError;
use purr_onnx_convert::proto::onnx_proto3::tensor_proto::DataLocation;
use purr_onnx_convert::proto::onnx_proto3::{GraphProto, ModelProto};

const LOCATION: &str = "decoder.onnx_data";

struct Converted {
    summary: Summary,
    data: Vec<u8>,
    graph: Vec<u8>,
}

fn run(input: &[u8]) -> Result<Converted, Box<dyn std::error::Error>> {
    Ok(run_with(input, TablePrecision::Full)?)
}

fn run_with(input: &[u8], table: TablePrecision) -> Result<Converted, ConvertError> {
    let (mut data, mut graph) = (Vec::new(), Vec::new());
    let summary = convert(
        &mut &*input,
        &mut data,
        &mut graph,
        &Options {
            data_location: LOCATION,
            table,
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

fn half_decoder() -> Decoder {
    Decoder {
        embedding_values: EmbeddingValues::Floats { largest: 3.0 },
        ..Decoder::small()
    }
}

fn entry(tensor: &purr_onnx_convert::proto::onnx_proto3::TensorProto, key: &str) -> Option<usize> {
    tensor
        .external_data
        .iter()
        .find(|entry| entry.key == key)
        .and_then(|entry| entry.value.parse().ok())
}

#[test]
fn a_half_table_is_stored_rounded_to_nearest_even_and_declared_as_half() -> TestResult {
    let decoder = half_decoder();
    let original = decoder.model();
    let source = original
        .graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "embedding")
        .ok_or("no embedding")?;
    let converted = run_with(&decoder.bytes()?, TablePrecision::Half)?;
    assert_eq!(converted.summary.half_precision_tables, 1);
    let model = ModelProto::parse_from_bytes(&converted.graph)?;
    let table = model
        .graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "embedding")
        .ok_or("no embedding")?;
    assert_eq!(table.data_type, 10);
    assert_eq!(table.dims, source.dims);
    let (offset, length) = (
        entry(table, "offset").ok_or("no offset")?,
        entry(table, "length").ok_or("no length")?,
    );
    assert_eq!(offset as u64 % DATA_ALIGNMENT, 0);
    assert_eq!(length, source.raw_data.len() / 2);
    let stored = converted
        .data
        .get(offset..offset + length)
        .ok_or("data too short")?;
    let expected: Vec<u8> = source
        .raw_data
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|bytes| f16::from_f32(f32::from_le_bytes(*bytes)).to_le_bytes())
        .collect();
    assert_eq!(stored, expected.as_slice());
    let leading: Vec<u16> = stored
        .as_chunks::<2>()
        .0
        .iter()
        .take(LEADING_VALUES.len())
        .map(|bytes| u16::from_le_bytes(*bytes))
        .collect();
    assert_eq!(leading, [0x3C00, 0x3C00, 0x3C02, 0xC100]);
    Ok(())
}

#[test]
fn a_half_table_is_read_by_half_precision_lookups_and_head_in_every_branch() -> TestResult {
    let converted = run_with(&half_decoder().bytes()?, TablePrecision::Half)?;
    assert_eq!(converted.summary.tied_embedding_branches, 2);
    let model = ModelProto::parse_from_bytes(&converted.graph)?;
    for branch in branches(&model) {
        let ops = op_types(branch);
        assert!(!ops.contains(&"Transpose"));
        assert!(!ops.contains(&"MatMul"));
        let gather = branch
            .node
            .iter()
            .position(|node| node.op_type == "Gather")
            .ok_or("no Gather")?;
        let gather_node = branch.node.get(gather).ok_or("no Gather")?;
        assert_eq!(
            gather_node.input.first().map(String::as_str),
            Some("embedding")
        );
        let rows = gather_node.output.first().ok_or("no rows")?;
        let after = branch.node.get(gather + 1).ok_or("no node after Gather")?;
        assert_eq!(after.op_type, "Cast");
        assert_eq!(after.input, std::slice::from_ref(rows));
        assert_eq!(after.output, ["embedded".to_owned()]);
        assert_eq!(after.attribute.first().map(|a| a.i), Some(1));

        let gemm = branch
            .node
            .iter()
            .position(|node| node.op_type == "Gemm")
            .ok_or("no Gemm")?;
        let gemm_node = branch.node.get(gemm).ok_or("no Gemm")?;
        assert_eq!(
            gemm_node.input.get(1).map(String::as_str),
            Some("embedding")
        );
        let first_cast = branch
            .node
            .iter()
            .filter(|node| node.op_type == "Cast")
            .map(|node| {
                (
                    node.input.clone(),
                    node.attribute.first().map_or(0, |a| a.i),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(first_cast.len(), 3);
        assert_eq!(first_cast.get(1).map(|cast| cast.1), Some(10));
        assert_eq!(
            first_cast.get(1).map(|cast| cast.0.clone()),
            Some(vec!["hidden".to_owned()])
        );
        let last = branch.node.last().ok_or("empty branch")?;
        assert_eq!(
            (
                last.op_type.as_str(),
                last.output.clone(),
                last.attribute.first().map(|a| a.i)
            ),
            ("Cast", vec!["logits".to_owned()], Some(1))
        );
    }
    Ok(())
}

#[test]
fn a_table_another_node_reads_stays_in_full_precision() -> TestResult {
    let decoder = Decoder {
        extra_table_reader: true,
        ..half_decoder()
    };
    let converted = run_with(&decoder.bytes()?, TablePrecision::Half)?;
    assert_eq!(converted.summary.half_precision_tables, 0);
    assert_eq!(converted.summary.tied_embedding_branches, 2);
    let model = ModelProto::parse_from_bytes(&converted.graph)?;
    let table = model
        .graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "embedding")
        .ok_or("no embedding")?;
    assert_eq!(table.data_type, 1);
    assert_eq!(entry(table, "length"), Some(32 * 16 * 4));
    assert!(
        branches(&model)
            .iter()
            .all(|branch| !op_types(branch).contains(&"Cast"))
    );
    Ok(())
}

#[test]
fn a_model_without_the_pattern_has_no_half_table() -> TestResult {
    let mut model = half_decoder().model();
    model.graph.mut_or_insert_default().node.clear();
    let converted = run_with(&model.write_to_bytes()?, TablePrecision::Half)?;
    assert_eq!(converted.summary.half_precision_tables, 0);
    assert_eq!(
        converted.summary.data_bytes,
        run(&model.write_to_bytes()?)?.summary.data_bytes
    );
    Ok(())
}

#[test]
fn values_half_precision_cannot_hold_are_an_error() -> TestResult {
    let decoder = Decoder {
        embedding_values: EmbeddingValues::Floats { largest: 1.0e10 },
        ..Decoder::small()
    };
    assert!(matches!(
        run_with(&decoder.bytes()?, TablePrecision::Half),
        Err(ConvertError::HalfOverflow { .. })
    ));
    Ok(())
}

#[test]
fn nodes_serialized_after_the_weights_are_an_error_when_a_table_would_be_narrowed() -> TestResult {
    let model = half_decoder().model();
    let mut weights = GraphProto::new();
    weights.initializer = model.graph.initializer.clone();
    let mut nodes = GraphProto::new();
    nodes.node = model.graph.node.clone();
    let mut graph = weights.write_to_bytes()?;
    graph.extend(nodes.write_to_bytes()?);

    let mut head = model;
    head.graph.clear();
    let mut bytes = head.write_to_bytes()?;
    {
        let mut output = protobuf::CodedOutputStream::vec(&mut bytes);
        output.write_bytes(7, &graph)?;
        output.flush()?;
    }
    assert!(matches!(
        run_with(&bytes, TablePrecision::Half),
        Err(ConvertError::TablesAfterWeights)
    ));
    assert!(run_with(&bytes, TablePrecision::Full).is_ok());
    Ok(())
}
