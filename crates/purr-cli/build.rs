use std::{path::Path, process::Command, process::ExitCode};

use shadow_rs::{BuildPattern, ShadowBuilder};

fn git_path(name: &str) -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--git-path", name])
        .output()
        .ok()?;
    let path = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    Path::new(&path).exists().then_some(path)
}

fn main() -> ExitCode {
    let if_path_changed = ["src", "Cargo.toml"]
        .into_iter()
        .map(str::to_owned)
        .chain(git_path("HEAD"))
        .chain(git_path("logs/HEAD"))
        .collect();
    if let Err(error) = ShadowBuilder::builder()
        .build_pattern(BuildPattern::Custom {
            if_path_changed,
            if_env_changed: Vec::new(),
        })
        .build()
    {
        println!("cargo::error={error}");
        return ExitCode::FAILURE;
    }
    purr_delayload::emit::report(purr_delayload::emit::emit_link_args())
}
