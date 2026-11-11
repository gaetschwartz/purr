//! Streaming SHA-256 check of a model file against its manifest entry.

use crate::error::IntegrityError;
use purr_common::web_model::digest::Sha256Digest;
use purr_common::web_model::registry::WebModelFile;
use sha2::{Digest as _, Sha256};

/// Hashes a file chunk by chunk so it never has to sit in memory as a whole.
pub struct FileChecker<'a> {
    file: &'a WebModelFile,
    hasher: Sha256,
    received_bytes: u64,
}

impl<'a> FileChecker<'a> {
    #[must_use]
    pub fn new(file: &'a WebModelFile) -> Self {
        Self {
            file,
            hasher: Sha256::new(),
            received_bytes: 0,
        }
    }

    pub fn update(&mut self, chunk: &[u8]) {
        self.hasher.update(chunk);
        self.received_bytes = self.received_bytes.saturating_add(chunk.len() as u64);
    }

    pub fn finish(self) -> Result<(), IntegrityError> {
        if self.received_bytes != self.file.size_bytes {
            return Err(IntegrityError::SizeMismatch {
                path: self.file.path.clone(),
                expected: self.file.size_bytes,
                actual: self.received_bytes,
            });
        }
        let actual = hex::encode(self.hasher.finalize());
        let matches = actual
            .parse::<Sha256Digest>()
            .is_ok_and(|digest| digest == self.file.sha256);
        if matches {
            Ok(())
        } else {
            Err(IntegrityError::DigestMismatch {
                path: self.file.path.clone(),
                expected: self.file.sha256.to_string(),
                actual,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::web_model::file_role::ModelFileRole;

    const ABC_DIGEST: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn abc_file(size_bytes: u64, digest: &str) -> Result<WebModelFile, Box<dyn std::error::Error>> {
        Ok(WebModelFile {
            role: ModelFileRole::Config,
            path: "config.json".to_owned(),
            size_bytes,
            sha256: digest.parse()?,
        })
    }

    #[test]
    fn accepts_a_file_split_across_chunks() -> Result<(), Box<dyn std::error::Error>> {
        let file = abc_file(3, ABC_DIGEST)?;
        let mut checker = FileChecker::new(&file);
        checker.update(b"a");
        checker.update(b"bc");
        assert_eq!(checker.finish(), Ok(()));
        Ok(())
    }

    #[test]
    fn rejects_a_wrong_length() -> Result<(), Box<dyn std::error::Error>> {
        let file = abc_file(4, ABC_DIGEST)?;
        let mut checker = FileChecker::new(&file);
        checker.update(b"abc");
        assert_eq!(
            checker.finish(),
            Err(IntegrityError::SizeMismatch {
                path: "config.json".to_owned(),
                expected: 4,
                actual: 3,
            })
        );
        Ok(())
    }

    #[test]
    fn rejects_changed_content_of_the_right_length() -> Result<(), Box<dyn std::error::Error>> {
        let file = abc_file(3, ABC_DIGEST)?;
        let mut checker = FileChecker::new(&file);
        checker.update(b"abd");
        assert!(matches!(
            checker.finish(),
            Err(IntegrityError::DigestMismatch { .. })
        ));
        Ok(())
    }
}
