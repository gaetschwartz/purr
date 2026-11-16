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
3. in a build with debug assertions only, the folder the transcribe.cpp build left its libraries in
   (`Origin::BuildOutput`, see [transcribe.cpp](#transcribecpp-cargo-feature-transcribe)): a
   `cargo run` or `dx serve` finds them without any setting, wherever the executable was copied to
   or `dx` rebuilt it;
4. the per-user folder: `%LOCALAPPDATA%\purr\runtime` on Windows, `~/.local/share/purr/runtime` on
   Linux, shared by every copy of Purr the user runs;
5. on Windows, `%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin`;
6. the system's own search (`PATH`, the system folders, `ld.so` paths).

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
- The test executables of `purr-engine-transcribe` delay-load `transcribe.dll` the same way: its build script asks
  `purr-delayload` for the arguments with `rustc-link-arg-tests`, so a test binary starts without the library next to it
  and finds it through the build-output folder below. Linux keeps the rpath that build script sets for every target, and macOS is unchanged.
- `/DELAYLOAD` only covers imports of the executable, which is all there is: `transcribe.dll` is the one
  library the executable imports. Its `ggml.dll`, `ggml-base.dll` and the backend modules are found by
  `transcribe.dll` itself, from its own folder, so they must sit next to it.
- Before any call into transcribe.cpp, `purr_engine_transcribe::runtime::get` opens `transcribe.dll` from
  the folders of [Search order](#search-order) (`LoadLibraryExW` with the DLL's folder as the load
  directory) and then has transcribe.cpp register its backend modules from that folder. It never falls back to
  the system's search: `ggml.dll` is a name other software puts on `PATH`. The delay-load helper then binds
  to the module that was opened. The failure hook of `purr-core` aborts for any delay-loaded library other
  than the Vulkan loader, so no code path may reach transcribe.cpp unguarded.
- Only a success is kept for the life of the process. A library that is missing or cannot be opened is looked for
  again on the next call (the desktop app's "Try again" relies on it): a missing library costs a few file checks and
  opens nothing, and a library that was opened stays mapped and is never unloaded. A backend registration that found
  no compute backend is kept too, because transcribe.cpp scans a folder once and answers the same afterwards.
  `get` loads one call at a time, so the backends are registered once and the log callback is set once.
- A build with debug assertions also searches the folder where `transcribe-cpp-sys` installed its libraries (`bin`
  under its `OUT_DIR` on Windows). The build script of `purr-engine-transcribe` reads it from the
  `DEP_TRANSCRIBE_CPP_RUNTIME_DIR` that cargo hands to the crate depending on `transcribe-cpp`, and `purr-delayload`
  turns it into `PURR_TRANSCRIBE_BUILD_DIR` only when `CARGO_CFG_DEBUG_ASSERTIONS` is set; the engine reads the variable
  only under `cfg!(debug_assertions)`. `dx serve` runs the executable from a folder it empties at every build, so
  without this the libraries in the profile folder are out of reach. A release build has neither the variable nor the
  folder in its search, and the folder appears in the searched list of the error and of `purr sys` like the others.
  Consequence: a debug executable copied alone to an empty folder still finds its libraries, so
  `check-windows-sys.ps1 -ExpectTranscribe` is meant for release builds.
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

## GPU backend modules of transcribe.cpp

The cargo features `vulkan`, `cuda` and `metal` of `purr-cli` and `purr-app` reach `purr-engine-transcribe` when `transcribe` is
on, which asks `transcribe-cpp-sys` for the matching backend. Each backend is one more loadable module next to `transcribe.dll`:

| Feature | Windows module | Imports of the module (`dumpbin /dependents`) | Size |
|---|---|---|---:|
| `vulkan` | `ggml-vulkan.dll` | `ggml-base.dll`, `vulkan-1.dll`, the Visual C++ runtime, system DLLs | 45,311,488 |
| `cuda` | `ggml-cuda.dll` | `ggml-base.dll`, `cublas64_13.dll` (which pulls in `cublasLt64_13.dll`), the Visual C++ runtime, system DLLs | 148,412,416 |
| `metal` | (macOS) `libggml-metal` module, metallib embedded | not measured | not measured |

On Linux the modules are `libggml-vulkan.so` and `libggml-cuda.so` in the same `lib` folder as `libtranscribe.so`; not measured.
The CUDA module links the CUDA runtime statically, so `cudart64_*.dll` is not one of its imports.

- Nothing links a module. transcribe.cpp opens each `ggml-*` module of its own folder through the system loader when
  `runtime::get` has it scan that folder, so the delay-load hook is not involved and no code path can abort for a module. A module
  whose import cannot be loaded (no Vulkan loader, a `vulkan-1.dll` that is not a library, no cuBLAS) is skipped by ggml, which logs
  it, and its devices are not listed. The engine is then loaded with the devices of the modules that did load, always the CPU.
- The runtime libraries of the binary's own `cuda` and `vulkan` features are opened from the search folders by `runtime::get` itself
  before it has transcribe.cpp scan its modules (`purr_core::accel::runtime_library::detect_missing`, the call `Inventory::detect` makes
  for the other engine). transcribe.cpp scans once per process, `open_on_best_device` asks for `availability` before it detects, and a
  module scanned before cuBLAS was opened from a folder outside `PATH` (`PURR_RUNTIME_DIR`, the per-user folder, `%CUDA_PATH%`) would be left
  out for good. A library that cannot be opened is reported as `RuntimeLibraryMissing` for the backend whichever engine runs the model.
- Modules are only found in the folder `transcribe.dll` was opened from, which `purr sys` prints. The per-user and the `runtime` folders
  hold a module only if they hold `transcribe.dll` too.
- Binary size: the Vulkan module adds 45.3 MB (14.2 MB deflated, 18.7 MB for the thirteen libraries together) and the CUDA module
  148.4 MB (133.9 MB deflated), on top of twelve files and 12.6 MB (4.6 MB deflated) for the CPU build, see
  [verification.md](verification.md#packaging-evidence).

### Packaging the modules

`scripts/stage-transcribe-runtime.ps1` lists the modules per backend (`-Backends vulkan`, `-Backends cuda`, both, or none) and fails
when the source holds a `ggml*.dll` for a backend that was not named, or lacks one that was. `-OptionalDestination <dir>` puts the CUDA
module there instead of next to the rest. `-Check` takes the same `-Backends`. `check-windows-imports.ps1 -Module` fails when a module
imports anything but the GPU runtime, ggml's own libraries, the Visual C++ runtime and Windows libraries, or delay-loads anything.

- The CUDA set is separable from the default packages: `ggml-cuda.dll`, `cublas64_*.dll` and `cublasLt64_*.dll`. `package-release-assets.sh`
  puts `ggml-cuda.dll` in `purr-runtime-cuda-windows-x64.zip` next to the cuBLAS libraries when the CLI folder holds it, and with
  `PURR_SPLIT_CUDA_RUNTIME: "true"` leaves it out of the CLI zip and the portable archive. A pack unpacked next to `purr.exe` (or next
  to `transcribe.dll`) enables CUDA for both engines; unpacked into the per-user folder it only carries cuBLAS for Whisper, because the
  module is not looked for there. Teaching `runtime::get` to scan the other runtime folders for modules (one `init_backends` per folder)
  is not done.
- The installer resources of `Dioxus.toml` name `ggml-vulkan.dll` and no CUDA module: the installer is "Vulkan and CPU".
- Linux and macOS packages: the CLI jobs copy the whole `lib` folder of the `transcribe-cpp-sys` install tree into `lib` next to the
  binary, so the modules go along; see [ci.md](ci.md#transcribecpp-the-second-ggml).

## Runtime pack (Windows)

`purr-runtime-cuda-windows-x64.zip` holds `cublas64_*.dll` and `cublasLt64_*.dll` at its root, and `ggml-cuda.dll` when the
build has the CUDA backend of the transcribe.cpp engine, and is produced by `scripts/package-release-assets.sh` from the CLI
artifact. With
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
  `cannonlake`, `cascadelake`, `icelake`, `alderlake`), twelve files and about 12 MB, which is about 4.5 MB in a zip, and one module per GPU backend of the build, see
  [GPU backend modules](#gpu-backend-modules-of-transcribecpp).
  All sit in one folder next to `purr.exe` (or in a `runtime` folder, or the per-user one).
  `scripts/stage-transcribe-runtime.ps1` is the one place that lists them; it copies by name from the `out\bin` of the
  `transcribe-cpp-sys` build (not from the profile folder, where the build script only copies them when it runs),
  fails on a missing file or on another `transcribe*.dll` or `ggml*.dll` in the source (a backend module counts as unlisted unless `-Backends`
  names its backend), and checks a folder or zip with `-Check`. A target
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
