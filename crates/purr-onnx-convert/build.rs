fn main() {
    println!("cargo::rerun-if-changed=proto/onnx.proto3");
    protobuf_codegen::Codegen::new()
        .pure()
        .include("proto")
        .input("proto/onnx.proto3")
        .cargo_out_dir("onnx_proto")
        .run_from_script();
}
