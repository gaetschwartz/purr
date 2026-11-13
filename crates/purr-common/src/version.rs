//! The version a build reports about itself.
//!
//! The release identifier comes from the crate version and the release channel. Git only supplies
//! the commit and its date, never the version.

use std::fmt;

const SHORT_HASH_LEN: usize = 7;

/// Which kind of build this is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    /// A local build.
    Dev,
    /// The `number`th automated build of the development branch.
    Canary { number: u64 },
    /// Built from a version tag.
    Stable,
}

impl Channel {
    /// Reads the channel from the values of `PURR_CHANNEL` and `PURR_CANARY_NUMBER` at compile
    /// time. Unset and empty both mean [`Channel::Dev`]. Anything unrecognised fails the build.
    #[allow(
        clippy::panic,
        reason = "a build with a malformed channel must not compile"
    )]
    pub const fn from_build_env(channel: Option<&str>, canary_number: Option<&str>) -> Self {
        let name = match channel {
            Some(name) => name,
            None => "",
        };
        if is_same(name, "") || is_same(name, "dev") {
            Self::Dev
        } else if is_same(name, "stable") {
            Self::Stable
        } else if is_same(name, "canary") {
            let Some(text) = canary_number else {
                panic!("PURR_CHANNEL=canary needs PURR_CANARY_NUMBER");
            };
            match parse_number(text) {
                Some(number) => Self::Canary { number },
                None => panic!("PURR_CANARY_NUMBER must be a plain number"),
            }
        } else {
            panic!("PURR_CHANNEL must be unset, dev, canary or stable")
        }
    }
}

/// The channel this crate was compiled for.
#[macro_export]
macro_rules! build_channel {
    () => {
        $crate::version::Channel::from_build_env(
            option_env!("PURR_CHANNEL"),
            option_env!("PURR_CANARY_NUMBER"),
        )
    };
}

const fn is_same(left: &str, right: &str) -> bool {
    let mut left = left.as_bytes();
    let mut right = right.as_bytes();
    loop {
        match (left, right) {
            ([], []) => return true,
            ([a, left_rest @ ..], [b, right_rest @ ..]) if *a == *b => {
                left = left_rest;
                right = right_rest;
            }
            _ => return false,
        }
    }
}

#[allow(clippy::cast_lossless, reason = "`u64::from` is not const")]
const fn parse_number(text: &str) -> Option<u64> {
    let mut rest = text.as_bytes();
    if rest.is_empty() {
        return None;
    }
    let mut number: u64 = 0;
    while let [digit @ b'0'..=b'9', tail @ ..] = rest {
        let Some(shifted) = number.checked_mul(10) else {
            return None;
        };
        let Some(next) = shifted.checked_add((*digit - b'0') as u64) else {
            return None;
        };
        number = next;
        rest = tail;
    }
    if rest.is_empty() { Some(number) } else { None }
}

/// The version of a build: the crate version qualified by the channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Release<'a> {
    pub crate_version: &'a str,
    pub channel: Channel,
}

impl fmt::Display for Release<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let version = self.crate_version;
        match self.channel {
            Channel::Stable => write!(f, "{version}"),
            Channel::Canary { number } => write!(f, "{version}-canary.{number}"),
            Channel::Dev => write!(f, "{version}-dev"),
        }
    }
}

/// The commit a build was made from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commit<'a> {
    short_hash: &'a str,
    date: &'a str,
    dirty: bool,
}

impl<'a> Commit<'a> {
    /// `date` is an ISO 8601 timestamp; only its day is kept. Without a hash, the build is not
    /// from a git checkout and there is no commit to report.
    pub fn new(hash: &'a str, date: &'a str, dirty: bool) -> Option<Self> {
        if hash.is_empty() {
            return None;
        }
        Some(Self {
            short_hash: hash.get(..SHORT_HASH_LEN).unwrap_or(hash),
            date: date.split(['T', ' ']).next().unwrap_or(date),
            dirty,
        })
    }
}

impl fmt::Display for Commit<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_hash)?;
        if self.dirty {
            f.write_str("-dirty")?;
        }
        if !self.date.is_empty() {
            write!(f, " {}", self.date)?;
        }
        Ok(())
    }
}

/// The release and, when known, the commit: `0.3.0-canary.123 (abc1234 2026-11-13)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildInfo<'a> {
    pub release: Release<'a>,
    pub commit: Option<Commit<'a>>,
}

impl fmt::Display for BuildInfo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.release)?;
        if let Some(commit) = &self.commit {
            write!(f, " ({commit})")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn release(channel: Channel) -> Release<'static> {
        Release {
            crate_version: "0.3.0",
            channel,
        }
    }

    fn info(channel: Channel, hash: &str, date: &str, dirty: bool) -> String {
        BuildInfo {
            release: release(channel),
            commit: Commit::new(hash, date, dirty),
        }
        .to_string()
    }

    #[test]
    fn stable_release_is_the_crate_version() {
        assert_eq!(release(Channel::Stable).to_string(), "0.3.0");
    }

    #[test]
    fn canary_release_carries_its_number() {
        assert_eq!(
            release(Channel::Canary { number: 123 }).to_string(),
            "0.3.0-canary.123"
        );
    }

    #[test]
    fn dev_release_is_marked() {
        assert_eq!(release(Channel::Dev).to_string(), "0.3.0-dev");
    }

    #[test]
    fn stable_build_info() {
        assert_eq!(
            info(
                Channel::Stable,
                "abc1234def5678901234567890123456789abcde",
                "2026-11-13T10:18:00+01:00",
                false
            ),
            "0.3.0 (abc1234 2026-11-13)"
        );
    }

    #[test]
    fn canary_build_info() {
        assert_eq!(
            info(
                Channel::Canary { number: 7 },
                "abc1234def5678901234567890123456789abcde",
                "2026-11-13T23:59:59Z",
                false
            ),
            "0.3.0-canary.7 (abc1234 2026-11-13)"
        );
    }

    #[test]
    fn dev_build_info() {
        assert_eq!(
            info(
                Channel::Dev,
                "abc1234def",
                "2026-11-13 10:18:00 +01:00",
                false
            ),
            "0.3.0-dev (abc1234 2026-11-13)"
        );
    }

    #[test]
    fn dirty_tree_is_marked_on_the_commit() {
        assert_eq!(
            info(
                Channel::Dev,
                "abc1234def",
                "2026-11-13T10:18:00+01:00",
                true
            ),
            "0.3.0-dev (abc1234-dirty 2026-11-13)"
        );
    }

    #[test]
    fn short_hash_is_kept_whole() {
        assert_eq!(
            info(Channel::Dev, "abc12", "2026-11-13", false),
            "0.3.0-dev (abc12 2026-11-13)"
        );
    }

    #[test]
    fn missing_date_leaves_only_the_hash() {
        assert_eq!(
            info(Channel::Stable, "abc1234def", "", false),
            "0.3.0 (abc1234)"
        );
    }

    #[test]
    fn outside_a_checkout_there_is_no_commit() {
        assert_eq!(info(Channel::Dev, "", "", true), "0.3.0-dev");
    }

    #[test]
    fn build_env_unset_or_empty_is_dev() {
        assert_eq!(Channel::from_build_env(None, None), Channel::Dev);
        assert_eq!(Channel::from_build_env(Some(""), Some("5")), Channel::Dev);
        assert_eq!(Channel::from_build_env(Some("dev"), None), Channel::Dev);
    }

    #[test]
    fn build_env_stable_ignores_the_number() {
        assert_eq!(
            Channel::from_build_env(Some("stable"), Some("5")),
            Channel::Stable
        );
    }

    #[test]
    fn build_env_canary_reads_the_number() {
        assert_eq!(
            Channel::from_build_env(Some("canary"), Some("1234")),
            Channel::Canary { number: 1234 }
        );
    }

    #[test]
    fn number_parser_accepts_digits_only() {
        assert_eq!(parse_number("0042"), Some(42));
        assert_eq!(parse_number(""), None);
        assert_eq!(parse_number("1a"), None);
        assert_eq!(parse_number("-1"), None);
        assert_eq!(parse_number("99999999999999999999999"), None);
    }
}
