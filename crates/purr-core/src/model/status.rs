//! State of a model file on disk.

use crate::model::digest::Sha256Digest;

/// Result of the cheap, size-only check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFileStatus {
    Missing,
    SizeMismatch { expected: u64, actual: u64 },
    Present,
}

/// Result of fully hashing a model file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelIntegrity {
    Missing,
    SizeMismatch {
        expected: u64,
        actual: u64,
    },
    ChecksumMismatch {
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    Verified,
}
