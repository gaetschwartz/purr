use std::{env, process::ExitCode};

fn main() -> ExitCode {
    println!("cargo::rerun-if-env-changed=DEP_TRANSCRIBE_CPP_LIB_DIR");
    let linux = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");
    if linux && let Some(lib_dir) = env::var_os("DEP_TRANSCRIBE_CPP_LIB_DIR") {
        println!(
            "cargo::rustc-link-arg=-Wl,-rpath,{}",
            lib_dir.to_string_lossy()
        );
    }
    purr_delayload::emit::report(purr_delayload::emit::emit_transcribe_engine())
}
