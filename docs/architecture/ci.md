# CI

Two workflows run on a push to `develop`. Neither one waits for the other. Other branches and pull requests only run `ci.yml`.

## `ci.yml`: checks

Triggers: pushes to any branch, pull requests, manual runs. A newer run on the same ref cancels the older one.

| Job | Runner | What it does |
|---|---|---|
| `fmt` | ubuntu-24.04 | `cargo fmt --all --check` |
| `test (linux)` | ubuntu-24.04 | `cargo clippy` (reports warnings without failing on them), then `cargo nextest run --workspace` with static FFmpeg. Tests that need a model skip themselves. `.config/nextest.toml` filters out the `slow_` tests. |

## `build.yml`: artifacts and releases

Triggers: pushes to `develop`, pushes of `v*` tags, and manual runs. Other branches and pull requests do not build artifacts.

```
ffmpeg (linux)   ─┬─ cli (linux, cpu | vulkan | cuda | all)
                  └─ desktop (linux, all)
ffmpeg (macos)   ─┬─ cli (macos, cpu | metal)
                  └─ desktop (macos, metal)
ffmpeg (windows) ─┬─ cli (windows, cpu | cuda-vulkan)      (+ cuda, vulkan on demand)
                  └─ desktop (windows, cuda-vulkan)
web ─ deploy web (vercel)
all cli/desktop/web ─┬─ release (canary)  ─ publish packages (canary)      push to develop
                     ├─ release (tag)     ─ publish packages (stable)      v* tags, stable only for non-prerelease
                     └─ prune caches                                       develop only
```

- Each `cli` and `desktop` job builds one artifact family. They all run in parallel once their platform's `ffmpeg` job is done.
- The `ffmpeg` jobs only check the cache when it hits. On a miss they build FFmpeg with vcpkg and save it, so the build jobs only restore it.
- The single-backend Windows CLIs (`cuda` alone, `vulkan` alone) never ship. They only build on a manual run with `windows_single_backends`.
- Concurrency: a push to `develop` cancels the previous run on that branch, so quick successive pushes only publish the newest commit. Tags and manual runs are never cancelled.
- A release job only starts when every build job succeeded. `ci.yml` is a separate workflow and is not waited for.
- `PURR_CHANNEL` and `PURR_CANARY_NUMBER` are set for the whole workflow: `stable` on `v*` tags, `canary` with the run number on `develop`, unset (a dev build) otherwise. They make `purr --version` report the same identifier as the release. See [releasing.md](releasing.md).
- The build scripts read the commit and its date from the checkout, which a shallow clone provides. Only the release jobs fetch the full history, for the changelog.

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

On Windows the `cmake` crate builds ggml with Ninja, set through `CMAKE_GENERATOR_x86_64_pc_windows_msvc`. That variable reaches only the `cmake` crate, not vcpkg. It takes the MSVC environment from the `cc` crate. Ninja runs nvcc on every core, while the Visual Studio generator compiles the CUDA sources one at a time.

Fallbacks:

- Full CUDA toolkit: set `method: installer` on `setup-cuda-windows`, or `packages: cuda-toolkit` on `setup-cuda-linux`.
- Visual Studio generator: delete the `CMAKE_GENERATOR_x86_64_pc_windows_msvc` lines. The CUDA build customizations are still installed for it.

## Caches

All caches share the repository's 10 GB limit. Over the limit, GitHub evicts the least recently used entries.

| Cache | Key | Saved by |
|---|---|---|
| vcpkg FFmpeg binaries | `vcpkg-ffmpeg-<os>-<vcpkg commit>`, plus the triplet on Windows | `ffmpeg (<os>)` on a miss |
| dx | `dioxus-cli-<os>-<version>` | the first job that misses |
| Rust (`Swatinem/rust-cache`): registry and `target`, including the ggml build of whisper-rs-sys | job id, variant, `GGML_*`, CUDA and Vulkan versions, `crt-static` and the CMake generator on Windows, toolchain and `Cargo.lock` hashes | `develop` only. Jobs on other refs restore `develop`'s cache. |

- Rust caches are saved only for the jobs on the critical path: the CUDA and Vulkan CLIs, the desktop apps, web, macOS metal and `test`. The cheap CPU-only CLIs restore but do not save.
- On Windows, `CARGO_TARGET_DIR` is `D:\t` to stay under the MSBuild path length limit. rust-cache is given `. -> ../../../t` because it joins the target onto the workspace path.
- After each `develop` run, `prune caches` deletes every Rust cache that a newer entry of the same job and configuration has replaced.
- CUDA and the Vulkan SDK are downloaded on every run and not cached. They download in about a minute, and the cache space is worth more to the Rust caches.

## Running one variant by hand

- Actions → build → Run workflow builds every job on the chosen ref.
- A manual run builds everything but never publishes a release. To publish, re-run the failed jobs of the `develop` push.
- `windows_single_backends`: also build the Windows CLI with CUDA only and with Vulkan only.
- There is no input to run a single job. Re-run a failed job from the run page instead.

## First-run checklist

1. `ffmpeg (*)`: each one finishes in seconds on a cache hit.
2. `cli (windows, cuda-vulkan)`, step `Install CUDA`: the components download, the checksums match, `nvcc --version` prints 13.5, and the customizations are copied to `...\18\Enterprise\MSBuild\Microsoft\VC\v180\BuildCustomizations`.
3. The same job's build gets through whisper-rs-sys with Ninja. That proves CMake found nvcc and cl from the `cc` environment. If it fails, apply the generator fallback above.
4. `Prepare CLI artifact` finds `cudart64_13.dll`, `cublas64_13.dll` and `cublasLt64_13.dll` under `bin\x64`.
5. `setup-msvc` prints Visual Studio 18 and passes the libclang check. Then the import and start checks pass.
6. `cli (linux, cuda | all)`: the `Install CUDA` step lists `/usr/local/cuda/lib64/stubs/libcuda.so` and `libculibos.a`.
7. `web`: builds without FFmpeg or the GTK packages.
8. `test (linux)`: passes. If it fails for environment reasons, it is new coverage, not a regression.
9. Second run on `develop`: the Windows rust-cache step restores a multi-GB entry, not about 165 MB, and the CUDA builds take minutes instead of about 30.
10. `prune caches`: lists what it deleted, and the total in Actions → Caches stays under 10 GB.
11. Every CLI job prints `purr --version` as `purr 0.1.0-canary.<run number> (<commit> <day>)` on `develop`. A `-dirty` suffix means the checkout changed during the build, for example an out-of-date `Cargo.lock`.
12. `release (canary)`: `Read the version from the built CLI` passes, the release appears as a pre-release with all assets, and at most 5 `canary-*` releases remain.
13. `publish packages (canary)`: the tap gains `purr@canary` and `purr-app@canary`, the bucket `purr-canary` and `purr-app-canary`.
