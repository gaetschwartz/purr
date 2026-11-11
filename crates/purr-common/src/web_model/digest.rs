//! SHA-256 digest value type.

use serde_with::{DeserializeFromStr, SerializeDisplay};
use std::fmt;
use std::str::FromStr;

const DIGEST_LEN: usize = 32;

/// A SHA-256 digest, written as 64 hexadecimal characters.
#[derive(Clone, Copy, PartialEq, Eq, Hash, SerializeDisplay, DeserializeFromStr)]
pub struct Sha256Digest([u8; DIGEST_LEN]);

#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("invalid SHA-256 digest: {source}")]
pub struct Sha256ParseError {
    #[source]
    source: hex::FromHexError,
}

impl Sha256Digest {
    pub const ZERO: Self = Self([0; DIGEST_LEN]);

    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&byte| byte == 0)
    }
}

impl FromStr for Sha256Digest {
    type Err = Sha256ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut bytes = [0u8; DIGEST_LEN];
        hex::decode_to_slice(s, &mut bytes).map_err(|source| Sha256ParseError { source })?;
        Ok(Self(bytes))
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sha256Digest({self})")
    }
}
