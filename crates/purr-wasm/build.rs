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

fn main() -> Result<(), Box<dyn Error>> {
    for input in ["js", "package.json", "package-lock.json", "rollup.config.js"] {
        println!("cargo::rerun-if-changed={input}");
    }

    if !dependencies_are_current() {
        npm(&["ci", "--no-audit", "--no-fund"])?;
    }
    npm(&["run", "build"])?;

    if !Path::new("dist/worker.js").is_file() {
        return Err("rollup did not produce dist/worker.js".into());
    }
    Ok(())
}
