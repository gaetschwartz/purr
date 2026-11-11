use std::error::Error;
use std::path::Path;
use std::process::Command;

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

fn main() -> Result<(), Box<dyn Error>> {
    for input in [
        "js",
        "package.json",
        "package-lock.json",
        "rollup.config.js",
    ] {
        println!("cargo::rerun-if-changed={input}");
    }

    npm(&["ci", "--no-audit", "--no-fund"])?;
    npm(&["run", "build"])?;

    if !Path::new("dist/worker.js").is_file() {
        return Err("rollup did not produce dist/worker.js".into());
    }
    Ok(())
}
