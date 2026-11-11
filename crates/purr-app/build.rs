use std::env;

const TRACKED_VARS: &[&str] = &[
    "VULKAN_SDK",
    "VK_SDK_PATH",
    "CARGO_MAKEFLAGS",
    "NUM_JOBS",
    "CARGO_ENCODED_RUSTFLAGS",
];

const BUILD_INFO_VARS: &[&str] = &["PROFILE", "OUT_DIR", "TARGET", "HOST", "OPT_LEVEL", "DEBUG"];

fn warn(line: impl std::fmt::Display) {
    println!("cargo::warning={line}");
}

fn collect_prefixed(prefix: &str) -> Vec<(String, String)> {
    let mut vars: Vec<_> = env::vars().filter(|(k, _)| k.starts_with(prefix)).collect();
    vars.sort();
    vars
}

fn main() {
    for var in TRACKED_VARS {
        println!("cargo::rerun-if-env-changed={var}");
    }

    warn("=== BUILD ENVIRONMENT FOR purr-app ===");
    for var in BUILD_INFO_VARS {
        warn(format_args!("{var}: {}", env::var(var).unwrap_or_default()));
    }

    for (key, value) in collect_prefixed("CARGO_FEATURE_") {
        warn(format_args!("{key} = {value}"));
    }
    for (key, value) in collect_prefixed("CARGO_CFG_") {
        warn(format_args!("{key} = {value}"));
    }
    for var in TRACKED_VARS {
        warn(format_args!(
            "{var}: {}",
            env::var(var).unwrap_or_else(|_| "NOT SET".to_string())
        ));
    }
}
