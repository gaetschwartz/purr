use std::env;
use std::fs::File;
use std::io::Write;

fn main() {
    // Create timestamped filename
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let filename = format!("{}/build_env_{}.txt", manifest_dir, timestamp);
    let mut file = File::create(&filename).unwrap();

    writeln!(file, "=== BUILD ENVIRONMENT FOR purr-app ===").unwrap();
    writeln!(file, "Timestamp: {}", timestamp).unwrap();
    writeln!(file, "PWD: {:?}", env::current_dir()).unwrap();
    writeln!(file, "\n=== CARGO BUILD INFO ===").unwrap();
    writeln!(file, "PROFILE: {}", env::var("PROFILE").unwrap_or_default()).unwrap();
    writeln!(file, "OUT_DIR: {}", env::var("OUT_DIR").unwrap_or_default()).unwrap();
    writeln!(file, "TARGET: {}", env::var("TARGET").unwrap_or_default()).unwrap();
    writeln!(file, "HOST: {}", env::var("HOST").unwrap_or_default()).unwrap();
    writeln!(file, "OPT_LEVEL: {}", env::var("OPT_LEVEL").unwrap_or_default()).unwrap();
    writeln!(file, "DEBUG: {}", env::var("DEBUG").unwrap_or_default()).unwrap();
    writeln!(file, "\n=== CARGO FEATURES ===").unwrap();
    writeln!(file, "CARGO_CFG_FEATURE: {}", env::var("CARGO_CFG_FEATURE").unwrap_or_default()).unwrap();

    // List all CARGO_FEATURE_* variables
    let mut features: Vec<String> = env::vars()
        .filter(|(k, _)| k.starts_with("CARGO_FEATURE_"))
        .map(|(k, v)| format!("{} = {}", k, v))
        .collect();
    features.sort();
    for feat in features {
        writeln!(file, "{}", feat).unwrap();
    }

    writeln!(file, "\n=== CARGO CFG ===").unwrap();
    let mut cfg_vars: Vec<String> = env::vars()
        .filter(|(k, _)| k.starts_with("CARGO_CFG_"))
        .map(|(k, v)| format!("{} = {}", k, v))
        .collect();
    cfg_vars.sort();
    for var in cfg_vars {
        writeln!(file, "{}", var).unwrap();
    }

    writeln!(file, "\n=== CRITICAL ENV VARS ===").unwrap();
    writeln!(file, "VULKAN_SDK: {}", env::var("VULKAN_SDK").unwrap_or("NOT SET".to_string())).unwrap();
    writeln!(file, "VK_SDK_PATH: {}", env::var("VK_SDK_PATH").unwrap_or("NOT SET".to_string())).unwrap();
    writeln!(file, "CARGO_MAKEFLAGS: {}", env::var("CARGO_MAKEFLAGS").unwrap_or("NOT SET".to_string())).unwrap();
    writeln!(file, "NUM_JOBS: {}", env::var("NUM_JOBS").unwrap_or("NOT SET".to_string())).unwrap();
    writeln!(file, "CARGO_ENCODED_RUSTFLAGS: {}", env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or("NOT SET".to_string())).unwrap();

    writeln!(file, "\n=== ALL ENVIRONMENT VARIABLES ===").unwrap();
    let mut all_vars: Vec<(String, String)> = env::vars().collect();
    all_vars.sort_by(|a, b| a.0.cmp(&b.0));
    for (key, value) in all_vars {
        writeln!(file, "{} = {}", key, value).unwrap();
    }

    println!("cargo:warning=Environment captured to: {}", filename);
    println!("cargo:warning=Build timestamp: {}", timestamp);
}
