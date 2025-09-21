use build_target::Os;

pub fn main() {
    let mut whisper_backends = Vec::new();

    if build_target::target_os() == Os::MacOS && is_feature_enabled("coreml") {
        println!("cargo:warning=CoreML support enabled on macOS. Ensure your models are compatible with CoreML.");
        println!("cargo:rustc-cfg=feature=\"whisper-rs/coreml\"");
        whisper_backends.push("coreml");
    }
    if build_target::target_os() == Os::MacOS && !is_feature_enabled("no-metal") {
        println!("cargo:warning=Metal support enabled by default on macOS. To disable, enable the `no-metal` feature.");
        println!("cargo:rustc-cfg=feature=\"whisper-rs/metal\"");
        whisper_backends.push("metal");
    }
    if matches!(build_target::target_os(), Os::Linux | Os::Windows) && is_feature_enabled("cuda") {
        println!("cargo:warning=CUDA support enabled. Ensure your system has a compatible NVIDIA GPU and CUDA drivers installed.");
        println!("cargo:rustc-cfg=feature=\"whisper-rs/cuda\"");
        whisper_backends.push("cuda");
    }
    if matches!(build_target::target_os(), Os::Linux | Os::Windows) && is_feature_enabled("vulkan")
    {
        println!("cargo:warning=Vulkan support enabled. Ensure your system has a compatible GPU and Vulkan drivers installed.");
        println!("cargo:rustc-cfg=feature=\"whisper-rs/vulkan\"");
        whisper_backends.push("vulkan");
    }

    println!(
        "cargo:rustc-env=WHISPER_RS_BACKENDS={}",
        whisper_backends.join(",")
    );
}

fn is_feature_enabled(feature: &str) -> bool {
    std::env::var(format!("CARGO_FEATURE_{}", feature.to_uppercase()).replace('-', "_")).is_ok()
}
