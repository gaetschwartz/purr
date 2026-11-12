//! A remote file pinned by size and SHA-256.

use crate::error::ModelIntegrityError;
use crate::model::digest::Sha256Digest;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedFile {
    /// What the file is called in diagnostics.
    pub name: String,
    pub url: String,
    pub size_bytes: u64,
    pub sha256: Sha256Digest,
}

impl PinnedFile {
    pub fn check_size(&self, actual: u64) -> Result<(), ModelIntegrityError> {
        if actual == self.size_bytes {
            Ok(())
        } else {
            Err(ModelIntegrityError::SizeMismatch {
                model_id: self.name.clone(),
                expected: self.size_bytes,
                actual,
            })
        }
    }

    pub fn check_digest(&self, actual: Sha256Digest) -> Result<(), ModelIntegrityError> {
        if actual == self.sha256 {
            Ok(())
        } else {
            Err(ModelIntegrityError::ChecksumMismatch {
                model_id: self.name.clone(),
                expected: self.sha256,
                actual,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::digest::Sha256Hasher;

    fn digest_of(bytes: &[u8]) -> Sha256Digest {
        let mut hasher = Sha256Hasher::new();
        hasher.update(bytes);
        hasher.finalize()
    }

    fn pinned(bytes: &[u8]) -> PinnedFile {
        PinnedFile {
            name: "fixture".to_owned(),
            url: "https://huggingface.co/owner/repo/resolve/abc/file".to_owned(),
            size_bytes: bytes.len() as u64,
            sha256: digest_of(bytes),
        }
    }

    #[test]
    fn a_matching_size_and_digest_pass() {
        let file = pinned(b"abc");
        assert_eq!(file.check_size(3), Ok(()));
        assert_eq!(file.check_digest(digest_of(b"abc")), Ok(()));
    }

    #[test]
    fn a_wrong_size_names_the_file_and_both_sizes() {
        let file = pinned(b"abc");
        assert_eq!(
            file.check_size(4),
            Err(ModelIntegrityError::SizeMismatch {
                model_id: "fixture".to_owned(),
                expected: 3,
                actual: 4,
            })
        );
        assert_eq!(
            file.check_size(2),
            Err(ModelIntegrityError::SizeMismatch {
                model_id: "fixture".to_owned(),
                expected: 3,
                actual: 2,
            })
        );
    }

    #[test]
    fn a_wrong_digest_names_the_file_and_both_digests() {
        let file = pinned(b"abc");
        assert_eq!(
            file.check_digest(digest_of(b"abd")),
            Err(ModelIntegrityError::ChecksumMismatch {
                model_id: "fixture".to_owned(),
                expected: digest_of(b"abc"),
                actual: digest_of(b"abd"),
            })
        );
    }
}
