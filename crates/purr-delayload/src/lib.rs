//! Build-script support for loading the GPU runtime libraries of purr binaries on demand, so a
//! binary starts on machines without them.

pub mod cuda;
pub mod emit;
pub mod error;
pub mod libraries;
pub mod target;
