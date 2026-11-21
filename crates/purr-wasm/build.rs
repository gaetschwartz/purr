use std::error::Error;
use std::fs;
use std::path::Path;
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

fn emit_converter_abi() -> Result<(), Box<dyn Error>> {
    let path = "../purr-onnx-convert/ABI_VERSION";
    println!("cargo::rerun-if-changed={path}");
    let version: u32 = fs::read_to_string(path)?.trim().parse()?;
    println!("cargo::rustc-env=PURR_CONVERTER_ABI={version}");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    for input in [
        "js",
        "package.json",
        "package-lock.json",
        "rollup.config.js",
    ] {
        println!("cargo::rerun-if-changed={input}");
    }

    if !dependencies_are_current() {
        npm(&["ci", "--no-audit", "--no-fund"])?;
    }
    npm(&["run", "build"])?;
    emit_converter_abi()?;

    for output in ["dist/worker.js", "dist/convert_worker.js"] {
        if !Path::new(output).is_file() {
            return Err(format!("rollup did not produce {output}").into());
        }
    }
    Ok(())
}
