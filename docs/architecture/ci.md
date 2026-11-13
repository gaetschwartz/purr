# CI

Two workflows run on a push to `develop`. Neither one waits for the other. Other branches and pull requests only run `ci.yml`.

## `ci.yml`: checks

Triggers: pushes to any branch, pull requests, manual runs. A newer run on the same ref cancels the older one.

| Job | Runner | What it does |
|---|---|---|
| `fmt` | ubuntu-24.04 | `cargo fmt --all --check` |
| `scripts` | ubuntu-24.04 | `scripts/test-prune-rust-caches.sh` and `scripts/test-drop-ggml-build.sh`, the tests of the cache pruning script and of the ggml build tree removal |
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
| Compiler cache (ccache): the C, C++ and CUDA objects of ggml | `ccache-<os>-v1-<run id>-<run attempt>`, one family per OS | the one writer job per OS, on `develop`, see [Compiler cache](#compiler-cache) |

The vcpkg and dx keys are exact, so they never pile up. The Rust keys change with every lockfile or toolchain change; each change would leave a second entry next to the old one. The compiler cache keys are unique per run on purpose: the writer saves the new entry first and then deletes the older ones of its family.

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
- `prune caches` in `build.yml` is the safety net, `scripts/prune-rust-caches.sh superseded`. After every `develop` run, even a partly failed one, it deletes all but the newest entry of each Rust and compiler cache family, which clears any duplicate that slipped through, such as after a failed deletion. It never deletes the only entry of a family.
- The modes match families by their keys, ignoring the two trailing hashes of a Rust key or the run id and attempt of a compiler cache key, and only touch `refs/heads/develop`. The `compiler-cache` mode is described under [Compiler cache](#compiler-cache). `scripts/test-prune-rust-caches.sh` checks all three against synthetic cache lists.
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

## Compiler cache

Compiling ggml (C, C++ and CUDA, inside the `whisper-rs-sys` build script) is the only expensive part of a cold build. The Rust caches already hold the result, but only as a whole: a new `Cargo.lock` hash, a Rust bump or a changed key starts them over, and the Linux jobs each store their own copy of nearly the same objects. ccache stores the compiled objects by content in a local directory that `actions/cache` saves and restores, so the objects survive those changes and are shared between jobs.

| Step | Where | What it does |
|---|---|---|
| `setup-compiler-cache` | after `setup-rust`, in every job that builds ggml | installs ccache, exports the configuration, restores the OS's newest entry |
| `finish-compiler-cache` | `if: always()`, before `prune-rust-cache` | prints the statistics and the step summary table, and in the writer job saves the entry |

### How cmake reaches ccache

- `whisper-rs-sys` forwards every `CMAKE_*`, `GGML_*` and `WHISPER_*` environment variable to cmake as `-D`. cmake also initialises `CMAKE_<LANG>_COMPILER_LAUNCHER` from the environment. The action exports `CMAKE_C_COMPILER_LAUNCHER`, `CMAKE_CXX_COMPILER_LAUNCHER` and `CMAKE_CUDA_COMPILER_LAUNCHER` as `ccache`. No dependency is patched. Because cmake reads the environment itself, the nested `vulkan-shaders-gen` build inherits it too.
- ggml's own hook (`GGML_CCACHE`) switches itself off when a launcher is set, so the explicit variables are the only mechanism. The hosted images ship neither ccache nor sccache.
- The Makefile and Ninja generators honour the launchers, for all three languages. The Visual Studio generator does not, which is another reason the Windows jobs stay on Ninja.
- Rust crates never go through ccache: there is no `RUSTC_WRAPPER`. `SCCACHE_DISABLE` in `.cargo/config.toml` is not read by anything (sccache has no such variable). Cargo's `[env]` does not override a variable that is already set unless `force = true`, and the switch does not rely on it.
- The cache is turned on by environment variables of the CI job only. A developer build never sees them.

### Why ccache and not sccache

ccache rewrites the absolute paths under `CCACHE_BASEDIR` to paths relative to the compiler's working directory before it hashes them and runs the compiler. ggml's build tree and its source copy both sit under cargo's `OUT_DIR` (`target/<profile>/build/whisper-rs-sys-<hash>/out`). The `<hash>`, the profile directory (`release`, `r`) and any `--target` directory differ between jobs and change with `Cargo.lock` or Rust bumps, but the relative paths from the build directory do not, so the keys do not depend on them. `__FILE__` matters here: `GGML_ABORT` in `ggml.h` expands it into the preprocessed text of every file that asserts, and the CUDA error macros in `common.cuh` do too. sccache only strips a base directory prefix from what it hashes. A probe on this repository's toolchain (cmake, Ninja, `cl`, nvcc 13.5) hit across two different directories when the sources did not use `__FILE__` and missed all of them as soon as they did.

| OS | Tool | Notes |
|---|---|---|
| Linux (gcc, nvcc) | ccache 4.14.2 | llama.cpp builds its CUDA 13.5 Linux releases with ccache |
| macOS (AppleClang) | ccache 4.14.2 | `compiler_check` runs `cc --version`, because `/usr/bin/cc` is the same shim for every Xcode |
| Windows (`cl`, nvcc, Ninja) | ccache 4.14.2 | llama.cpp builds Windows CUDA 13.5 with MSVC and ccache. Unproven for this repository: read the first Windows summary. |

ccache is a pinned download from the release page, checked against a SHA-256 in the action. If the download, the checksum or the version check fails, the step warns and the job builds without the cache.

### Configuration

| Variable | Value | Why |
|---|---|---|
| `CCACHE_DIR` | `$RUNNER_TEMP/ccache` | the directory `actions/cache` saves |
| `CCACHE_BASEDIR` | `$GITHUB_WORKSPACE` (Linux, macOS), `D:\t` (Windows) | covers every cmake build tree and source copy, and no toolchain or SDK |
| `CCACHE_NOHASHDIR` | `1` | the working directory is `OUT_DIR/build`, whose name changes. Release builds carry no debug info, so nothing else records it. |
| `CCACHE_COMPILERCHECK` | `content` (`%compiler% --version` on macOS) | tool installs have new modification times on every run, ccache's default compares them |
| `CCACHE_MAXSIZE` | 2G Linux and Windows, 1G macOS | caps the entry. ccache evicts its least recently used objects beyond it. |

### What is cached

- Cached: every `.c`, `.cpp` and `.cu` compilation of ggml and whisper.cpp, including the generated `ggml-vulkan` shader sources (a compile of an array).
- Not cached: linking and archiving, the `glslc` and `vulkan-shaders-gen` steps that generate those shader sources, the Metal library (embedded, not compiled at build time) and anything ccache calls uncacheable, which the statistics list.
- Unproven for this repository: ccache with `cl` and with nvcc under MSVC on Windows, and the `/showIncludes` replay Ninja needs. If the Windows summary shows no hits, the cause is in that job's log, and nothing else is affected.

### Which jobs share objects

A key covers the compiler binary, the arguments (defines included), the preprocessed source and the relative paths. Jobs share an object when ggml is configured identically.

| Objects | Shared by | Differ because |
|---|---|---|
| `ggml-base`, `ggml-cpu` | every job of one OS (Linux: cpu, vulkan, cuda, all, desktop) | expected identical: `GGML_NATIVE=OFF` and the same flags everywhere |
| CUDA (`ggml-cuda`, about 160 files) | `cli (linux, all)` and `desktop (linux, all)`; `cli (windows, cuda-vulkan)`, `desktop (windows, cuda-vulkan)` and the manual `cuda` CLI | `cli (linux, cuda)` passes `GGML_CUDA_NO_VMM=OFF`, `all` and `desktop` pass `ON`, which is a define on every CUDA command line. The Linux `cuda` job restores the cache but gets no CUDA hits from it and adds none. |
| Vulkan | every job with Vulkan | nothing |
| `ggml` (backend registry), `whisper.cpp` | jobs with the same backend set | `GGML_USE_CUDA` and `GGML_USE_VULKAN` are public defines |
| everything on macOS | `cli (macos, metal)` and `desktop (macos, metal)`; `ggml-base` and `ggml-cpu` also `cli (macos, cpu)` | `GGML_METAL` |

`--profile r` on the Windows desktop job and the `dx` target directory only change `OUT_DIR`, which the base directory removes. Whether `dx` passes anything else to cmake is unproven, so check the hit rate of the desktop jobs in the first run.

### Single writer

- Exactly one job per OS saves: `cli (linux, all)`, `cli (macos, metal)`, `cli (windows, cuda-vulkan)`. Every other job restores only. One writer cannot overwrite another's objects, and the entry does not depend on which job finishes first.
- Key: `ccache-<os>-v1-<run id>-<run attempt>`. The restore key is the prefix `ccache-<os>-v1-`, so a lockfile, Rust or configuration change never misses. Change `v1` in `setup-compiler-cache` to start the cache over.
- The writer saves only on `develop`, when the job succeeded and the statistics show a miss, meaning something was added. A run that only hits keeps the existing entry and its age.
- After the save, `scripts/prune-rust-caches.sh compiler-cache` deletes the other entries of the family, only if it can list the new one. `prune caches` (`superseded`) clears any leftover.
- A run cold in ccache but warm in the Rust cache compiles no ggml at all, so it never fills the compiler cache. To seed it, delete the writer's Rust cache entry (`gh cache delete`) and re-run: that cold run compiles ggml and saves the first entry.

### Switches

- `PURR_COMPILER_CACHE` at the top of `build.yml`: `on` or `off`. With `off` no job installs, restores, saves or configures anything and the old behaviour is untouched. The literal after `||` in the expression is the default. A manual run can override it with the `compiler_cache` input (`default`, `on`, `off`).
- `PURR_SHRINK_RUST_CACHE` and the `shrink_rust_cache` input, see [Shrinking the Rust caches](#shrinking-the-rust-caches). Default `off`.

### Reading the statistics

- Every job prints `ccache --show-stats --verbose` and a summary table (hits, misses, hit rate, cache size, files) in its step summary, with the key it restored and whether it saved. They are printed on failure too.
- Expected: a cold first run is 0 % hits everywhere. When ggml is recompiled on a restored cache with the same configuration, nearly every file hits; the misses name what differs.
- `ccache --show-stats` reports `Uncacheable` calls by reason. A nonzero count for `Unsupported compiler option` names the flag.

### Budget

| Entry | Expected | Cap |
|---|---|---|
| ccache linux | not measured, a few hundred MiB | 2 GiB |
| ccache macos | not measured, under 100 MiB | 1 GiB |
| ccache windows | not measured, a few hundred MiB | 2 GiB |

With the Rust families at 6.6 GiB and about 1 GiB for the three entries (a guess), the steady state is about 7.5 GiB, under the 10 GiB limit. All three at their caps would be 11.6 GiB, so read `cache size` in the first summaries and lower a cap if it fills. While a writer saves, the old and the new entry of its OS exist together for a moment. Shrinking the Rust caches frees space; it is the only change that shrinks existing entries.

### Shrinking the Rust caches

Off by default. The Rust caches hold the ggml build tree a second time. With `PURR_SHRINK_RUST_CACHE: "on"`, the jobs marked `shrink` (the Linux `all` CLI and desktop, the macOS Metal CLI and desktop, the Windows `cuda-vulkan` CLI and desktop) delete the whole `target/**/build/whisper-rs-sys-*` and `target/**/.fingerprint/whisper-rs-sys-*` directories with `scripts/drop-ggml-build.sh` just before rust-cache saves, on `develop` after a successful job.

- Both directories have to go. Cargo reads `.fingerprint/<package>-<hash>` to decide whether a build script is fresh, and a fresh one is not run: cargo replays `build/<package>-<hash>/output`. Removing only the build tree would link against missing libraries. With the fingerprint gone, cargo runs the build script again on the next run, whose cmake commands then hit ccache.
- The cost is a cmake configure, bindgen, the link and a rebuild of `whisper-rs-sys` on every run, a few minutes per job, instead of a cold ggml build only when a Rust cache is lost.
- It is only safe while the ccache entry holds those objects. Jobs whose objects the writer does not produce (the Linux `cuda` and `vulkan` CLIs) are not marked. If the ccache entry is lost, the next run of a shrunk job compiles ggml cold, as it did before any cache.
- The job summary lists the removed directories and their uncompressed size. The saved cache space is what `gh cache list` shows for the families before and after.

## Running one variant by hand

- Actions → build → Run workflow builds every job on the chosen ref.
- `compiler_cache` and `shrink_rust_cache` override the two switches for that run.
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
15. `setup-compiler-cache`: prints `ccache version 4.14.2` and the configuration with `base_dir`, `hash_dir = false` and `compiler_check`. A `::warning::` that ccache could not be set up means the job built without it.
16. In the build log of a job that compiles ggml (`whisper-rs-sys` builds verbosely), the compile commands start with `ccache` for `.c`, `.cpp` and `.cu` files. `whisper-rs-sys` is `Fresh` on a warm Rust cache and compiles nothing, so the first meaningful statistics come from a run that rebuilds ggml (see Single writer).
17. First run that compiles ggml: every compiler cache summary shows 0 % hits and the writer reports `saving a new entry`. `gh cache list` gains one `ccache-<os>-v1-...` entry per OS and no older one.
18. Run after that, with the Rust cache lost or `PURR_SHRINK_RUST_CACHE` on: the writer's table shows hits for nearly all of its files. Compare the desktop jobs' hit rates with the writer's (Linux `all`, Windows, macOS): a low rate means the jobs do not compile ggml identically. The Linux `cuda` job is expected to hit only the CPU and Vulkan objects. A Windows table with no hits at all means ccache did not cache `cl` or nvcc there.
19. A run with the `compiler_cache` input set to `off` shows no ccache step output and no `ccache-*` cache activity.
