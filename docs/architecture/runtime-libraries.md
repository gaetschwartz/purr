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
- Debian package: keeps shipping the CUDA runtime under `/usr/lib/purr-desktop`.
