use std::fmt;

/// A number of bytes that displays with a binary unit prefix, such as `116.6 MB`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteSize(pub u64);

impl fmt::Display for ByteSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
        let mut size = self.0 as f64;
        let mut unit = 0;
        while size >= 1024.0 && unit < UNITS.len() - 1 {
            size /= 1024.0;
            unit += 1;
        }
        let name = UNITS.get(unit).copied().unwrap_or("B");
        if unit == 0 {
            write!(f, "{} {name}", self.0)
        } else {
            write!(f, "{size:.1} {name}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_sizes_are_whole_bytes() {
        assert_eq!(ByteSize(0).to_string(), "0 B");
        assert_eq!(ByteSize(1023).to_string(), "1023 B");
    }

    #[test]
    fn larger_sizes_use_one_decimal() {
        assert_eq!(ByteSize(1536).to_string(), "1.5 KB");
        assert_eq!(ByteSize(122_309_681).to_string(), "116.6 MB");
        assert_eq!(ByteSize(1_610_612_736).to_string(), "1.5 GB");
    }
}
