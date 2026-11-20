use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

const NPM: &str = if cfg!(windows) { "npm.cmd" } else { "npm" };

fn npm(args: &[&str]) -> Result<(), Box<dyn Error>> {
    let status = Command::new(NPM).args(args).status().map_err(|e| {
        format!(
            "failed to run `{NPM} {}`: {e}; Node.js and npm are required to build the web worker",
            args.join(" ")
        )
    })?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`{NPM} {}` failed with {status}", args.join(" ")).into())
    }
}

fn modified(path: &str) -> Option<SystemTime> {
    fs::metadata(path).and_then(|meta| meta.modified()).ok()
}

/// Whether `node_modules` was installed from the current lockfile.
fn dependencies_are_current() -> bool {
    match (
        modified("node_modules/.package-lock.json"),
        modified("package-lock.json"),
    ) {
        (Some(installed), Some(locked)) => installed >= locked,
        _ => false,
    }
}

const CONVERTER_PACKAGE: &str = "purr-onnx-convert";
const CONVERTER_TARGET: &str = "wasm32-unknown-unknown";
const CONVERTER_ARTIFACT: &str = "purr_onnx_convert.wasm";

/// Compiles the ONNX converter to its own small WebAssembly module, apart from the app's build:
/// its own target directory avoids the build lock, and the app's flags would not suit it.
fn build_converter() -> Result<(), Box<dyn Error>> {
    let target_dir = PathBuf::from(env::var("OUT_DIR")?).join("converter-target");
    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let mut command = Command::new(cargo);
    command
        .args(["build", "--locked", "--release", "--lib"])
        .args(["--package", CONVERTER_PACKAGE, "--target", CONVERTER_TARGET])
        .arg("--target-dir")
        .arg(&target_dir)
        .arg("--jobs")
        .arg(env::var("NUM_JOBS").unwrap_or_else(|_| "4".to_owned()))
        .env("CARGO_PROFILE_RELEASE_OPT_LEVEL", "s")
        .env("CARGO_PROFILE_RELEASE_LTO", "true")
        .env("CARGO_PROFILE_RELEASE_CODEGEN_UNITS", "1")
        .env("CARGO_PROFILE_RELEASE_PANIC", "abort")
        .env("CARGO_PROFILE_RELEASE_STRIP", "symbols");
    for inherited in [
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "CARGO_BUILD_TARGET",
        "CARGO_BUILD_RUSTFLAGS",
        "CARGO_PRIMARY_PACKAGE",
    ] {
        command.env_remove(inherited);
    }
    let status = command
        .status()
        .map_err(|e| format!("failed to run cargo to build {CONVERTER_PACKAGE}: {e}"))?;
    if !status.success() {
        return Err(format!(
            "building {CONVERTER_PACKAGE} for {CONVERTER_TARGET} failed with {status}"
        )
        .into());
    }
    let built = target_dir
        .join(CONVERTER_TARGET)
        .join("release")
        .join(CONVERTER_ARTIFACT);
    fs::create_dir_all("dist")?;
    fs::copy(&built, Path::new("dist").join(CONVERTER_ARTIFACT))?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    for input in [
        "js",
        "package.json",
        "package-lock.json",
        "rollup.config.js",
        "../purr-onnx-convert/src",
        "../purr-onnx-convert/proto",
        "../purr-onnx-convert/build.rs",
        "../purr-onnx-convert/Cargo.toml",
    ] {
        println!("cargo::rerun-if-changed={input}");
    }

    if !dependencies_are_current() {
        npm(&["ci", "--no-audit", "--no-fund"])?;
    }
    npm(&["run", "build"])?;
    build_converter()?;

    for output in ["dist/worker.js", "dist/convert_worker.js"] {
        if !Path::new(output).is_file() {
            return Err(format!("rollup did not produce {output}").into());
        }
    }
    Ok(())
}
