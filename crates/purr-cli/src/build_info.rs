//! The version `purr --version` and `purr sys` report.

use purr_common::version::{BuildInfo, Channel, Commit, Release};
use shadow_rs::shadow;
use std::sync::LazyLock;

shadow!(build);

const CHANNEL: Channel = purr_common::build_channel!();

/// The crate version qualified by the release channel, then the commit and its day.
pub fn version() -> &'static str {
    static VERSION: LazyLock<String> = LazyLock::new(|| {
        BuildInfo {
            release: Release {
                crate_version: env!("CARGO_PKG_VERSION"),
                channel: CHANNEL,
            },
            commit: Commit::new(
                build::COMMIT_HASH,
                build::COMMIT_DATE_3339,
                !build::GIT_CLEAN,
            ),
        }
        .to_string()
    });
    &VERSION
}
