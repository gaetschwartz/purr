# CI

Two workflows run on a push to `develop`. Neither one waits for the other. Other branches and pull requests only run `ci.yml`.

## `ci.yml`: checks

Triggers: pushes to any branch, pull requests, manual runs. A newer run on the same ref cancels the older one.

| Job | Runner | What it does |
|---|---|---|
| `fmt` | ubuntu-24.04 | `cargo fmt --all --check` |
| `scripts` | ubuntu-24.04 | `scripts/test-prune-rust-caches.sh`, the tests of the cache pruning script |
| `test (linux)` | ubuntu-24.04 | `cargo clippy` (reports warnings without failing on them), then `cargo nextest run --workspace` with static FFmpeg. Tests that need a model skip themselves. `.config/nextest.toml` filters out the `slow_` tests. No Rust cache: about 10 minutes, against 3 to 4 with one. |

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
- `deploy web (vercel)` is part of this workflow, so it only runs for `develop` and `v*` tags.
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
| Rust | `setup-rust`, pinned to one version on every runner (see [Caches](#caches)) | same |
| Runner | `ubuntu-24.04` | `windows-2025-vs2026`: Visual Studio 2026 (18) with the C++ tools, LLVM (libclang for bindgen), Windows SDK 10.0.26100, CMake 4. Nothing is installed with Chocolatey. `setup-msvc` fails early if the image changes. |
| CUDA `CUDA_VERSION` | NVIDIA apt repository: `cuda-nvcc`, `cuda-cudart-dev`, `libcublas-dev` and their dependencies | NVIDIA redistributable archives (checked against `redistrib_<version>.json`): `cuda_crt`, `cuda_cudart`, `cuda_nvcc`, `libnvvm`, `libcublas`, `cccl`, `visual_studio_integration`, about 530 MB. The MSBuild CUDA customizations are copied into Visual Studio, because the `cmake` crate configures ggml with the Visual Studio generator. |
| Vulkan `VULKAN_SDK_VERSION` | Ubuntu packages and the LunarG SDK tarball | LunarG SDK installer |
| FFmpeg | vcpkg `x64-linux` | vcpkg `x64-windows-static` (static CRT) |

On Windows the `cmake` crate builds ggml with Ninja, set through `CMAKE_GENERATOR_x86_64_pc_windows_msvc`. That variable reaches only the `cmake` crate, not vcpkg. It takes the MSVC environment from the `cc` crate. Ninja runs nvcc on every core, while the Visual Studio generator compiles the CUDA sources one at a time.

Fallbacks:

- Full CUDA toolkit: set `method: installer` on `setup-cuda-windows`, or `packages: cuda-toolkit` on `setup-cuda-linux`.
- Visual Studio generator: delete the `CMAKE_GENERATOR_x86_64_pc_windows_msvc` lines. The CUDA build customizations are still installed for it.

## Caches

All caches share the repository's 10 GB limit. Over the limit, GitHub evicts the least recently used entries, and a lost Windows Rust cache costs about 90 minutes. So the budget is kept by holding exactly one entry per cache family.

| Cache | Key | Saved by |
|---|---|---|
| vcpkg FFmpeg binaries | `vcpkg-ffmpeg-<os>-<vcpkg commit>`, plus the triplet on Windows | `ffmpeg (<os>)` on a miss |
| dx | `dioxus-cli-<os>-<version>` | the first job that misses |
| Rust (`Swatinem/rust-cache`): registry and `target`, including the ggml build of whisper-rs-sys | `v0-rust-<cache-key>-<job id>-<os>-<arch>-<environment hash>-<lockfile hash>`. The cache key holds the variant, `GGML_*`, CUDA and Vulkan versions, `crt-static` and the CMake generator on Windows. The environment hash covers the installed Rust toolchains and `CARGO*` variables, the lockfile hash `Cargo.lock` and the manifests. | `develop` only, by the jobs in the table below. Jobs on other refs restore `develop`'s cache. |

The vcpkg and dx keys are exact, so they never pile up. The Rust keys change with every lockfile or toolchain change; each change would leave a second entry next to the old one.

### Rust cache families

A family is the set of Rust entries that differ only by the two trailing hashes. Each one holds one entry at steady state. Sizes are as listed by `gh cache list` (GiB). The times are a job without and with its cache.

| Job | Family size | Without / with cache |
|---|---|---|
| cli (linux, vulkan) | 0.47 | 8 / 1 to 2 min |
| cli (linux, cuda) | 0.81 | 61 / 2 to 3 min |
| cli (linux, all) | 0.88 | 66 / 3 min |
| desktop (linux, all) | 1.12 | 71 / 6 to 7 min |
| cli (macos, metal) | 0.37 | 6 / 2 min |
| desktop (macos, metal) | 0.53 | 8 to 9 / 3 min |
| cli (windows, cuda-vulkan) | 0.93 | 82 to 91 / 3 to 4 min |
| desktop (windows, cuda-vulkan) | 1.13 | 96 to 99 / 5 to 6 min |
| Rust total | 6.24 | |
| vcpkg FFmpeg (linux, macos, windows) | 0.10, 0.09, 0.11 | |
| dx (linux, macos, windows) | 0.03 each | |
| **Total** | **6.6** | 3.4 left for a transient second entry |

These jobs never save a Rust cache, because the cache costs more space than it saves time: the CPU-only CLIs (3 to 8 min), `web` (2 to 3 min against 1) and `test (linux)` (10 min against 3 to 4). The single-backend Windows CLIs of a manual run do not save either.

### Pruning

- The last step of each saving job, `prune-rust-cache`, deletes the family's entries on `develop` just before the post step of rust-cache saves the new one, so the old and the new entry never exist together. `scripts/prune-rust-caches.sh own-family` does it with `gh cache list` and `jq`.
- It does nothing unless the job saves (`cache-save` of `setup-rust`, the same value rust-cache gets as `save-if`) and rust-cache reported no exact hit. An exact hit saves nothing, so its entry stays. It runs only when every earlier step succeeded, which is also when rust-cache saves.
- A failed deletion marks the step failed but not the job: rust-cache saves only after a successful job, so failing it would lose the new entry too.
- The old entry is gone before the new one exists. A cancelled run, or a failed save, in that window leaves the family empty until the next run on `develop` has rebuilt it. A cancelled or failed run before the last step deletes nothing.
- `prune caches` in `build.yml` is the safety net, `scripts/prune-rust-caches.sh superseded`. After every `develop` run, even a partly failed one, it deletes all but the newest entry of each family, which clears any duplicate that slipped through, such as after a failed deletion. It never deletes the only entry of a family.
- Both modes match families by their keys, ignoring the two trailing hashes, and only touch `refs/heads/develop`. `scripts/test-prune-rust-caches.sh` checks them against synthetic cache lists.
- The jobs that prune have `actions: write` next to `contents: read`.

### Rust toolchain

CI is pinned to one Rust version, the `toolchain` default of `.github/actions/setup-rust/action.yml`. Developer toolchains are not affected: there is no `rust-toolchain.toml`.

- rust-cache hashes `rustc -vV` of every toolchain that `rustup` lists. The hosted images ship their own `stable`, so `setup-rust` uninstalls every toolchain except the pinned one and fails if `rustc` does not report the pinned version. Otherwise the keys would move whenever an image updates.
- To bump: change that one default to a full `X.Y.Z` and push. The first run on `develop` misses every Rust cache, with no fallback restore. The Windows CUDA builds take about 90 minutes then. Each job deletes its old entry before it saves the new one.
- A new Rust release changes nothing until the pin is bumped.

### Notes

- On Windows, `CARGO_TARGET_DIR` is `D:\t` to stay under the MSBuild path length limit. rust-cache is given `. -> ../../../t` because it joins the target onto the workspace path.
- CUDA and the Vulkan SDK are downloaded on every run and not cached. They download in about a minute, and the cache space is worth more to the Rust caches.
- `desktop (linux, all)` and `cli (linux, all)` are separate families although they share their features: one key would let whichever job saves first decide what the entry holds, and one is built with `dx`, the other with `cargo`.

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
10. Cache pruning: the last step of each saving job lists the entries it deleted, or says why it kept them. `prune caches` has nothing left to delete. `gh cache list` shows one Rust entry per family and the total stays near 6.6 GiB.
11. `setup-rust`, step `Keep only the pinned toolchain`: `rustup toolchain list` shows the pinned version alone. In the rust-cache log, `Rust Versions` names that one version.
12. Every CLI job prints `purr --version` as `purr 0.1.0-canary.<run number> (<commit> <day>)` on `develop`. A `-dirty` suffix means the checkout changed during the build, for example an out-of-date `Cargo.lock`.
13. `release (canary)`: `Read the version from the built CLI` passes, the release appears as a pre-release with all assets, and at most 5 `canary-*` releases remain.
14. `publish packages (canary)`: the tap gains `purr-canary` and `purr-app-canary`, the bucket `purr-canary` and `purr-app-canary`.
