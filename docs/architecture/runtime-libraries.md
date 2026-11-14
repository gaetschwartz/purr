# GPU runtime libraries

Purr starts on a machine with no NVIDIA driver, no CUDA libraries and no Vulkan loader, and uses
whichever GPU runtime it finds. This document says what the binaries need, where they look for it,
and how a package should provide it.

## What the Windows binaries import

What `dumpbin /dependents` reports for the release `purr.exe` (`cuda,vulkan`), as enforced by
`scripts/check-windows-imports.ps1`:

- Hard imports: Windows system DLLs only, including the Universal CRT api-set DLLs of Windows 10
  and 11. The C and C++ runtime is linked statically (`crt-static` in `.cargo/config.toml`, the
  `x64-windows-static` vcpkg triplet for FFmpeg, `CMAKE_MSVC_RUNTIME_LIBRARY` for ggml), so the
  Visual C++ Redistributable (`VCRUNTIME140*.dll`, `MSVCP140*.dll`) is not needed. The cuBLAS
  libraries only import `KERNEL32.dll`.
- Delay-loaded: `cublas64_13.dll` and `vulkan-1.dll`. `cublasLt64_13.dll` is pulled in by cuBLAS.
- Not imported: `nvcuda.dll` (the static CUDA runtime opens the driver itself on first use) and
  `cudart64_13.dll` (linked statically).

`scripts/check-windows-imports.ps1` keeps the hard-import list free of the NVIDIA and Vulkan
libraries and of the Visual C++ runtime, and `scripts/check-windows-sys.ps1` runs `purr sys`
without them.

## Where the names come from

The binaries' build scripts (`purr-cli`, `purr-app`) emit the delay-load link arguments through
`purr-delayload` from their own `cuda` and `vulkan` features. `purr-core`'s build script emits the
library names as `PURR_CUDA_RUNTIME_LIBS` and `PURR_VULKAN_RUNTIME_LIBS` for its own `cuda` and
`vulkan` features, which the engine crates turn on with theirs; the delay-load failure hook and the
Linux Vulkan forwarders live in `purr-core` too. The names belong to the binary, not to an engine:
every engine whose ggml uses CUDA or Vulkan shares them. `Inventory::detect` opens the libraries
before an engine lists its devices (see [engines.md](engines.md)).

## Search order

The command line and the desktop app share one order (`purr_core::accel::runtime_dir`):

1. the directories of `PURR_RUNTIME_DIR` (a path list), so an installer or a developer can force a place;
2. next to the executable: the executable's own folder on Windows, `lib` on Linux, and a `runtime`
   folder on both. This keeps the all-in-one archives working and prefers the libraries that
   shipped with the build;
3. the per-user folder: `%LOCALAPPDATA%\purr\runtime` on Windows, `~/.local/share/purr/runtime` on
   Linux, shared by every copy of Purr the user runs;
4. on Windows, `%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin`;
5. the system's own search (`PATH`, the system folders, `ld.so` paths).

For each backend the libraries are opened from the first folder that holds all of them, dependencies
first, so one backend never mixes versions from two places. They stay mapped, so the delay-load
helper and `dlopen` of a soname bind to the module that was opened. `purr runtime status` lists the
folders and what each holds, `purr runtime path` prints where to unpack a pack, and `purr sys` and
the desktop Settings page say where a missing library was looked for.

## Linux

`libcudart` and `libcublas` are `DT_NEEDED` entries of the binary and are resolved by `ld.so`
before `main`, through the `$ORIGIN/lib` rpath, `LD_LIBRARY_PATH` or `ld.so.conf`. No preload can
supply them, so the CUDA runtime stays bundled in `lib` next to the binary. The search order applies
to what Purr opens itself: `libcuda.so.1` and `libvulkan.so.1`.

The Vulkan loader found this way is also what `purr sys` uses to read each device's driver name, driver
version string and Vulkan API version (`purr_core::accel::vulkan_probe`, which loads it at run time
without linking it). It creates one Vulkan instance, only when ggml has a Vulkan device, and
is never run for a transcription.

## transcribe.cpp (cargo feature `transcribe`)

The transcribe.cpp engine (`purr-engine-transcribe`, see [engines.md](engines.md)) is an engine, not a GPU
backend, so it is not in `AccelBackend` or `detect_missing`. It has its own ggml, built by
`transcribe-cpp-sys` with `dynamic-backends`, so the two ggml versions stay apart.

- Windows: the binaries delay-load `transcribe.dll`. `purr-delayload` adds `/DELAYLOAD:transcribe.dll` for
  a binary built with `transcribe` and hands the name to the engine crate as
  `PURR_TRANSCRIBE_RUNTIME_LIBS`. `purr.exe` imports nothing from `ggml.dll`: whisper.cpp's ggml is
  inside it, and a DLL import names its DLL, so the two never bind to each other. `check-windows-imports.ps1`
  forbids `transcribe.dll` and `ggml*` among the hard imports and, with `-ExpectDelayLoaded transcribe.dll`,
  requires the delay-load.
- `/DELAYLOAD` only covers imports of the executable, which is all there is: `transcribe.dll` is the one
  library the executable imports. Its `ggml.dll`, `ggml-base.dll` and the backend modules are found by
  `transcribe.dll` itself, from its own folder, so they must sit next to it.
- Before any call into transcribe.cpp, `purr_engine_transcribe::runtime::get` opens `transcribe.dll` from
  the folders of [Search order](#search-order) (`LoadLibraryExW` with the DLL's folder as the load
  directory) and then has transcribe.cpp register its backend modules from that folder. It never falls back to
  the system's search: `ggml.dll` is a name other software puts on `PATH`. The delay-load helper then binds
  to the module that was opened. The failure hook of `purr-core` aborts for any delay-loaded library other
  than the Vulkan loader, so no code path may reach transcribe.cpp unguarded.
- `purr sys` prints the outcome under "Engines": loaded (with the folder and the version), `transcribe.dll`
  not found (with the folders searched), found but not loadable (a library it needs, such as the Visual C++
  runtime, is missing), or no compute backend. Its devices appear in the one device list, next to
  whisper.cpp's, with the engines that see each. `scripts/check-windows-sys.ps1 -ExpectTranscribe` checks
  the first two on a copy of the executable alone and on one with the libraries.
- The libraries are built with the dynamic C and C++ runtime (`/MD`, `transcribe-cpp-sys` does not follow
  `crt-static`), so they import `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll` and `MSVCP140.dll`. The
  executable still does not. Without the Visual C++ Redistributable the engine reports itself as not loadable
  and everything else works; a package can put app-local copies of those DLLs next to `transcribe.dll`.
- Linux and macOS have no delay-load: the binary lists `libtranscribe.so` (`libtranscribe.dylib`) as needed,
  so it must sit in `lib` next to the binary (the `$ORIGIN/lib` rpath that the Linux jobs already set) or on
  the library path, together with the `libggml*` libraries and backend modules of the same folder. A binary
  built with `transcribe` does not start without them.
- On Linux the static ggml of whisper.cpp would be exported by the executable, and the dynamic linker would then bind
  `libtranscribe.so` and `libggml-base.so` to it instead of to their own: measured with `LD_DEBUG=bindings`. The link
  argument `-Wl,--exclude-libs,ALL`, emitted by `purr-delayload` for a Linux binary built with `transcribe`, keeps archive
  symbols out of the dynamic table, and `test (linux)` fails if a ggml or gguf symbol shows up in `nm -D` of the
  executable or a transcribe.cpp library binds to it. See [ci.md](ci.md#linux). The folder of the libraries is the
  install tree (`lib` under `OUT_DIR` of `transcribe-cpp-sys`), which also holds the backend modules, not the build tree.

## Runtime pack (Windows)

`purr-runtime-cuda-windows-x64.zip` holds `cublas64_*.dll` and `cublasLt64_*.dll` at its root and is
produced by `scripts/package-release-assets.sh` from the CLI artifact. With
`PURR_SPLIT_CUDA_RUNTIME: "true"` in the workflow, the Windows CLI zip and the desktop portable zip
leave those DLLs out and the pack is the only place they ship.

## What packages should do

- NSIS installer: unpack the pack into `$LOCALAPPDATA\purr\runtime` (an optional component or a
  download step).
- Scoop: a `purr-cuda-runtime` package that extracts the pack into `$persist_dir\runtime`, and a
  `env_set` of `PURR_RUNTIME_DIR` to that folder.
- Homebrew and macOS: nothing, Metal is part of the system.
- A package built with `transcribe` ships the set the `transcribe-cpp-sys` build script stages next to the
  executable in the target directory: on Windows `transcribe.dll`, `ggml.dll`, `ggml-base.dll` and
  `ggml-cpu-<tier>.dll` for each instruction set tier (`x64`, `sse42`, `sandybridge`, `haswell`, `skylakex`,
  `cannonlake`, `cascadelake`, `icelake`, `alderlake`), twelve files and about 12 MB, which is about 4.5 MB in a zip.
  All sit in one folder next to `purr.exe` (or in a `runtime` folder, or the per-user one).
  `scripts/stage-transcribe-runtime.ps1` is the one place that lists them; it copies by name from the `out\bin` of the
  `transcribe-cpp-sys` build (not from the profile folder, where the build script only copies them when it runs),
  fails on a missing file or on another `transcribe*.dll` or `ggml*.dll` in the source, and checks a folder or zip with `-Check`. A target
  directory that has seen other feature sets may hold stale modules, so nothing globs.
- Windows CLI zip: the build job stages the libraries into the artifact folder, and the release scripts zip that
  folder whole (the CUDA split only leaves out the cuBLAS and cudart libraries).
- NSIS installer with `transcribe`: `dx bundle` does not list DLLs. `[bundle] resources` in `Dioxus.toml` names the
  twelve files under `dist-runtime/`, which the desktop job fills with the staging script between two `dx bundle`
  runs; the NSIS template installs each resource next to `purr-app.exe`. Off by default (`PURR_DESKTOP_TRANSCRIBE`)
  because it has not been run: see [ci.md](ci.md#windows-packaging). The portable archive is staged the same way.
  On Debian the Linux modules (fourteen on x86-64) would go under `/usr/lib/purr-desktop` with the CUDA runtime,
  which is on the binary's rpath; not done.
- Debian package: keeps shipping the CUDA runtime under `/usr/lib/purr-desktop`.
