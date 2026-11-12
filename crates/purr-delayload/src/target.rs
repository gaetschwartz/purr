/// How the binaries being built can defer loading a GPU runtime library.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LinkTarget {
    /// The MSVC linker delay-loads DLL imports.
    WindowsMsvc,
    /// glibc can `dlopen` a library that the binary does not list as needed.
    LinuxGnu,
    /// Everything is linked eagerly.
    #[default]
    Other,
}

impl LinkTarget {
    #[must_use]
    pub fn new(os: &str, env: &str) -> Self {
        match (os, env) {
            ("windows", "msvc") => Self::WindowsMsvc,
            ("linux", "gnu") => Self::LinuxGnu,
            _ => Self::Other,
        }
    }

    #[must_use]
    pub fn from_cargo_env() -> Self {
        let cfg = |name: &str| std::env::var(name).unwrap_or_default();
        Self::new(&cfg("CARGO_CFG_TARGET_OS"), &cfg("CARGO_CFG_TARGET_ENV"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_the_targets_that_can_defer_loading() {
        assert_eq!(LinkTarget::new("windows", "msvc"), LinkTarget::WindowsMsvc);
        assert_eq!(LinkTarget::new("linux", "gnu"), LinkTarget::LinuxGnu);
    }

    #[test]
    fn other_targets_link_eagerly() {
        assert_eq!(LinkTarget::new("windows", "gnu"), LinkTarget::Other);
        assert_eq!(LinkTarget::new("linux", "musl"), LinkTarget::Other);
        assert_eq!(LinkTarget::new("macos", ""), LinkTarget::Other);
    }
}
