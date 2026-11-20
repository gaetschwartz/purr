//! Streaming ONNX rewrite for the browser model install: initializers above a size threshold move
//! to an external data file, and a tied embedding stops being copied by ONNX Runtime.

mod attention_fusion;
pub mod convert;
pub mod error;
mod fields;
mod tied_embedding;

#[cfg(feature = "tooling")]
pub mod tooling;

#[cfg(target_arch = "wasm32")]
mod abi;

#[allow(
    clippy::all,
    clippy::pedantic,
    clippy::restriction,
    clippy::nursery,
    unused_qualifications,
    missing_docs,
    unreachable_pub,
    dead_code,
    mismatched_lifetime_syntaxes
)]
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx_proto/mod.rs"));
}
