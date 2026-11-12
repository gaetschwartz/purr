# CI

Two workflows run on every push. Neither one waits for the other.

## `ci.yml`: checks

Triggers: pushes to any branch, pull requests, manual runs. A newer run on the same ref cancels the older one.

| Job | Runner | What it does |
|---|---|---|
| `fmt` | ubuntu-24.04 | `cargo fmt --all --check` |
| `test (linux)` | ubuntu-24.04 | `cargo clippy` (reports warnings without failing on them), then `cargo nextest run --workspace` with static FFmpeg. Tests that need a model skip themselves. `.config/nextest.toml` filters out the `slow_` tests. |

## `build.yml`: artifacts and releases

Triggers: pushes to any branch and to tags, the 02:00 UTC schedule, and manual runs.

```
ffmpeg (linux)   ─┬─ cli (linux, cpu | vulkan | cuda | all)
                  └─ desktop (linux, all)
ffmpeg (macos)   ─┬─ cli (macos, cpu | metal)
                  └─ desktop (macos, metal)
ffmpeg (windows) ─┬─ cli (windows, cpu | cuda-vulkan)      (+ cuda, vulkan on demand)
                  └─ desktop (windows, cuda-vulkan)
web ─ deploy web (vercel)
all cli/desktop/web ─┬─ release (nightly) ─ publish packages (nightly)     schedule, or manual with nightly
                     ├─ release (tag)     ─ publish packages (stable)      v* tags, stable only for non-prerelease
                     └─ prune caches                                       develop only
```

- Each `cli` and `desktop` job builds one artifact family. They all run in parallel once their platform's `ffmpeg` job is done.
- The `ffmpeg` jobs only check the cache when it hits. On a miss they build FFmpeg with vcpkg and save it, so the build jobs only restore it.
- The single-backend Windows CLIs (`cuda` alone, `vulkan` alone) never ship. They only build on a manual run with `windows_single_backends`.
- Concurrency: a push to a branch cancels the previous run on that branch. Tags, scheduled runs and manual runs are never cancelled. Scheduled and manual runs have their own group, so a push to `develop` cannot replace a pending nightly.

### Artifacts consumed downstream

`scripts/package-release-assets.sh` reads these artifact names:

- `purr-cli-*`: `purr-cli-linux-x64-{cpu,vulkan,cuda,all}`, `purr-cli-macos-arm64-metal`, `purr-cli-windows-x64-cuda-vulkan`
- `purr-desktop-*`: `purr-desktop-linux-x64-all`, `purr-desktop-macos-arm64-metal`, `purr-desktop-windows-x64-cuda-vulkan`
- `purr-web`

The script only reads names in this list. The other build variants upload as `purr-dev-cli-*` and stay out of releases. `publish-packages.yml` reads `SHA256SUMS.txt` from the published release.

### Toolchains

| What | Linux | Windows |
|---|---|---|
| Runner | `ubuntu-24.04` | `windows-2025-vs2026`: Visual Studio 2026 (18) with the C++ tools, LLVM (libclang for bindgen), Windows SDK 10.0.26100, CMake 4. Nothing is installed with Chocolatey. `setup-msvc` fails early if the image changes. |
| CUDA `CUDA_VERSION` | NVIDIA apt repository: `cuda-nvcc`, `cuda-cudart-dev`, `libcublas-dev` and their dependencies | NVIDIA redistributable archives (checked against `redistrib_<version>.json`): `cuda_crt`, `cuda_cudart`, `cuda_nvcc`, `libnvvm`, `libcublas`, `cccl`, `visual_studio_integration`, about 530 MB. The MSBuild CUDA customizations are copied into Visual Studio, because the `cmake` crate configures ggml with the Visual Studio generator. |
| Vulkan `VULKAN_SDK_VERSION` | Ubuntu packages and the LunarG SDK tarball | LunarG SDK installer |
| FFmpeg | vcpkg `x64-linux` | vcpkg `x64-windows-static` (static CRT) |

To fall back to the full CUDA toolkit, set `method: installer` on `setup-cuda-windows` or `packages: cuda-toolkit` on `setup-cuda-linux`.

## Caches

All caches share the repository's 10 GB limit. Over the limit, GitHub evicts the least recently used entries.

| Cache | Key | Saved by |
|---|---|---|
| vcpkg FFmpeg binaries | `vcpkg-ffmpeg-<os>-<triplet>-<vcpkg commit>` | `ffmpeg (<os>)` on a miss |
| dx | `dioxus-cli-<os>-<version>` | the first job that misses |
| Rust (`Swatinem/rust-cache`): registry and `target`, including the ggml build of whisper-rs-sys | job id, variant, `GGML_*`, CUDA and Vulkan versions, `crt-static` on Windows, toolchain and `Cargo.lock` hashes | `develop` only. Jobs on other refs restore `develop`'s cache. |

- Rust caches are saved only for the jobs on the critical path: the CUDA and Vulkan CLIs, the desktop apps, web, macOS metal and `test`. The cheap CPU-only CLIs restore but do not save.
- On Windows, `CARGO_TARGET_DIR` is `D:\t` to stay under the MSBuild path length limit. rust-cache is given `. -> ../../../t` because it joins the target onto the workspace path.
- After each `develop` run, `prune caches` deletes every Rust cache that a newer entry of the same job and configuration has replaced.
- CUDA and the Vulkan SDK are downloaded on every run and not cached. They download in about a minute, and the cache space is worth more to the Rust caches.

## Running one variant by hand

- Actions → build → Run workflow builds every job on the chosen ref.
- `nightly`: also publish a nightly release.
- `windows_single_backends`: also build the Windows CLI with CUDA only and with Vulkan only.
- There is no input to run a single job. Re-run a failed job from the run page instead.

## First-run checklist

1. `ffmpeg (*)`: the Linux and macOS keys are new, so they build FFmpeg once and save it.
2. `cli (windows, cuda-vulkan)`, step `Install CUDA`: the components download, the checksums match, `nvcc --version` prints 13.5, and the customizations are copied to `...\18\Enterprise\MSBuild\Microsoft\VC\v180\BuildCustomizations`.
3. The same job's build gets through whisper-rs-sys. That proves CMake found CUDA through the Visual Studio generator.
4. `Prepare CLI artifact` finds `cudart64_13.dll`, `cublas64_13.dll` and `cublasLt64_13.dll` under `bin\x64`.
5. `setup-msvc` prints Visual Studio 18 and passes the libclang check. Then the import and start checks pass.
6. `cli (linux, cuda | all)`: the `Install CUDA` step lists `/usr/local/cuda/lib64/stubs/libcuda.so` and `libculibos.a`.
7. `web`: builds without FFmpeg or the GTK packages.
8. `test (linux)`: passes. If it fails for environment reasons, it is new coverage, not a regression.
9. Second run on `develop`: the Windows rust-cache step restores a multi-GB entry, not about 165 MB, and the CUDA builds take minutes instead of about 30.
10. `prune caches`: lists what it deleted, and the total in Actions → Caches stays under 10 GB.
