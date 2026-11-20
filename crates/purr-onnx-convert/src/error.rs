//! Failures of an ONNX conversion.

#[oopsie::oopsie]
pub enum ConvertError {
    #[oopsie(
        "Reading the ONNX model failed: {source}",
        code = "purr::onnx_convert::read"
    )]
    Read { source: protobuf::Error },

    #[oopsie(
        "Writing the converted {sink} failed: {source}",
        code = "purr::onnx_convert::write"
    )]
    Write {
        sink: &'static str,
        source: std::io::Error,
    },

    #[oopsie(
        "Encoding the converted model failed: {source}",
        code = "purr::onnx_convert::encode"
    )]
    Encode { source: protobuf::Error },

    #[oopsie(
        "The model has an unrecognised wire type {wire_type} on field {field}",
        code = "purr::onnx_convert::wire_type"
    )]
    WireType { field: u32, wire_type: u32 },

    #[oopsie(
        "The model has fields this converter does not know: {message}",
        code = "purr::onnx_convert::unknown_field"
    )]
    UnknownField { message: String },

    #[oopsie("The model has no graph", code = "purr::onnx_convert::no_graph")]
    NoGraph,

    #[oopsie(
        "Initializer {name} already lives in an external file",
        code = "purr::onnx_convert::already_external"
    )]
    AlreadyExternal { name: String },
}
