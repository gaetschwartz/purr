# Accelerators

## Backends

Purr runs Whisper through whisper.cpp and ggml. A backend is in a build only if it was compiled in; `purr sys` lists every backend with its state.

| Backend | Hardware | Notes |
|---|---|---|
| CUDA | NVIDIA GPUs | needs the CUDA runtime libraries at run time |
| Vulkan | Intel, AMD and NVIDIA GPUs, integrated or discrete | the path for Intel iGPUs and Arc; needs the system's Vulkan loader and driver |
| Metal, CoreML | Apple silicon | needs a build with the `metal` or `coreml` feature |
| ROCm, SYCL | AMD, Intel | recognised when ggml lists them; Purr has no build feature for them |
| CPU | any | always works; last resort |

There is no OpenVINO and no Intel NPU support.

Automatic selection tries the backends in priority order (CUDA, ROCm, CoreML, Metal, SYCL, Vulkan, CPU) and moves on when one is unavailable or fails to load the model. `--accel BACKEND` forces one, without fallback unless `--accel-fallback` is also given.

## Choosing a device

Within a backend, the device is picked in this order: a discrete GPU before an integrated one, then more memory, then the lower ggml index. There is no option to name a device. To restrict what ggml sees, set `GGML_VK_VISIBLE_DEVICES` (Vulkan) or `CUDA_VISIBLE_DEVICES` (CUDA) before starting Purr; the ranking then applies to what remains.

## Reading `purr sys`

- **Backends**: `Not built`, `Built, but no usable device`, `Built, but the runtime library ... cannot be loaded` (the library comes from the GPU driver or a system package), or `Available`.
- **Devices**: one block per ggml device, as ggml reports it: name, description, type (discrete or integrated GPU), backend, PCI bus id when ggml has one, memory, and ggml's capability flags. Integrated GPUs share system RAM, so their memory is system memory. `Picked by automatic selection for` marks the device a backend would use. ggml does not report the driver or the Vulkan version, so for Vulkan devices Purr asks the Vulkan driver itself and prints `Driver` (the name and version string the driver gives, unmodified) and `Vulkan API`; when it cannot tie the driver's device to ggml's by PCI bus id or unique name, it says so instead of guessing.
- **CPU**: for each instruction set, `Compiled` is what this binary was built with (ggml's compile-time flags) and `This CPU` is what the running processor reports (x86 only; `n/a` where it cannot be asked or ggml does not say). AMX is reported for the build only. `Compiled in but not supported by this CPU` means the build can fail with an illegal instruction; `Supported by this CPU but not compiled in` means the build leaves speed unused.

## Reporting

Open an issue with the "Accelerator report" form. It asks for your hardware, driver versions, the whole `purr sys` output, and two timings of the same clip: `purr -m base --accel vulkan -t 4 a13.mp3` against `purr -m base --accel cpu -t 4 a13.mp3`, with `a13.mp3` from the `samples` folder.
