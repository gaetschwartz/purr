use std::process::ExitCode;

use shadow_rs::{BuildPattern, ShadowBuilder};

fn main() -> ExitCode {
    if let Err(error) = ShadowBuilder::builder()
        .build_pattern(BuildPattern::Lazy)
        .build()
    {
        println!("cargo::error={error}");
        return ExitCode::FAILURE;
    }
    purr_delayload::emit::report(purr_delayload::emit::emit_link_args())
}
