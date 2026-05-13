use build_target::target_os;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

fn main() {
    let os = target_os();
    println!("cargo:rustc-env=TARGET_OS={:?}", os);

    // Capture environment for debugging vulkan feature propagation
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let env_file = out_dir.join("build_env.txt");
    let mut f = BufWriter::new(File::create(&env_file).unwrap());

    writeln!(f, "=== BUILD ENVIRONMENT FOR purr-core ===").unwrap();
    writeln!(f, "PWD: {:?}", env::current_dir()).unwrap();
    writeln!(f, "\n=== ENVIRONMENT VARIABLES ===").unwrap();

    let env_vars: Vec<(String, String)> = env::vars().collect();
    for (key, value) in env_vars {
        writeln!(f, "{} = {:?}", key, value).unwrap();
    }

    println!(
        "cargo:warning=Environment captured to: {}",
        env_file.display()
    );
}
