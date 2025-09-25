use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let target_dir = Path::new(&out_dir).join("assets");

    // Create assets directory
    fs::create_dir_all(&target_dir).unwrap();

    // Check if we have Node.js and npm available
    if Command::new("npm").arg("--version").output().is_ok() {
        println!("cargo:rerun-if-changed=src/worker.js");
        println!("cargo:rerun-if-changed=package.json");
        println!("cargo:rerun-if-changed=rollup.config.js");

        // Install dependencies
        let install_status = Command::new("npm")
            .args(["install"])
            .status()
            .expect("Failed to run npm install");

        assert!(install_status.success(), "npm install failed");

        // Build the worker
        let build_status = Command::new("npm")
            .args(["run", "build"])
            .status()
            .expect("Failed to build worker");

        assert!(build_status.success(), "Worker build failed");

        // Copy built worker to output directory
        let worker_src = Path::new("dist/worker.js");
        let worker_dest = target_dir.join("worker.js");

        if worker_src.exists() {
            fs::copy(worker_src, worker_dest).expect("Failed to copy worker.js");
            println!("Worker built and copied successfully");
        } else {
            // Fallback: copy source worker directly
            let worker_src = Path::new("src/worker.js");
            let worker_dest = target_dir.join("worker.js");
            fs::copy(worker_src, worker_dest).expect("Failed to copy worker.js");
            println!("Warning: Using unbundled worker.js");
        }
    } else {
        println!("Warning: npm not found, copying worker.js without bundling");

        // Copy source worker directly
        let worker_src = Path::new("src/worker.js");
        let worker_dest = target_dir.join("worker.js");
        fs::copy(worker_src, worker_dest).expect("Failed to copy worker.js");
    }

    // Tell cargo to link the assets
    println!(
        "cargo:rustc-env=WORKER_JS_PATH={}",
        target_dir.join("worker.js").display()
    );
}
