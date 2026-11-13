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

## Devices and their order

A device is one backend driving one piece of hardware, as `purr sys` lists them: the same GPU seen by CUDA and by Vulkan is two devices, because they differ in speed and in what can go wrong. Purr tries the devices one after the other and moves on to the next when one cannot be used or fails to load the model. The CPU is always the last resort, and it is never switched off.

In the desktop app, Settings > Hardware is that list. Each device has a switch (the CPU has none) and buttons to move it up or down. Without any change the order is: backend priority (CUDA, ROCm, CoreML, Metal, SYCL, Vulkan, CPU), then within a backend discrete GPUs before integrated ones, then more memory, then the lower ggml index. To run on the CPU alone, move the CPU to the top or switch every GPU off. A device that was in the list but is no longer found stays there, greyed as "Not found", and is skipped; a device found for the first time is added after the existing GPUs. A device is identified by its backend, its reported name and, when two share a name, its position among them.

The Transcribe screen has an Advanced row with a Device choice. It makes one device the first choice for the runs of that session; the rest of the list still follows if it fails. It is not saved and resets when the app restarts.

Settings saved by earlier versions (an acceleration switch and one backend) are converted when they are loaded: acceleration off switches every GPU off, a forced backend puts its devices first, automatic keeps the usual order. The "fall back if unavailable" option is gone: falling through the list always happens.

## Choosing devices on the command line

- `--accel BACKEND` puts that backend's devices first and keeps the other devices and the CPU as fallbacks. `--accel cpu` and `--no-gpu` use the CPU alone. `--accel-priority a,b` lists backends to try first.
- `--device` starts on one device. Either the index `purr sys` prints for it (`--device 2`), or tokens joined by `/` in any order, each naming its own field: a backend (`cuda`, `vulkan`, `rocm`, ...), a type (`gpu` for discrete, `igpu` for integrated, `cpu`, `accel`) and a device id (`0000:01:00.0`, as `purr sys` prints it; integrated GPUs and the CPU usually report none). So `--device vulkan/igpu` is the integrated Vulkan GPU and `--device cuda/0000:01:00.0` the CUDA device at that PCI address. The explicit `backend=vulkan/type=igpu` form (also `id=` and `index=`) means the same. `cpu` matches the CPU. The selector must match exactly one device: if none or several match, Purr stops and lists the devices with their indexes and the fields that would tell them apart. Giving a field twice or an unknown token is an error.
- To restrict what ggml sees, set `GGML_VK_VISIBLE_DEVICES` (Vulkan) or `CUDA_VISIBLE_DEVICES` (CUDA) before starting Purr.

## Reading `purr sys`

- **Backends**: `Not built`, `Built, but no usable device`, `Built, but the runtime library ... cannot be loaded` (the library comes from the GPU driver or a system package), or `Available`.
- **Devices**: one block per ggml device, as ggml reports it: name, description, type (discrete or integrated GPU), backend, PCI bus id when ggml has one, memory, and ggml's capability flags. Integrated GPUs share system RAM, so their memory is system memory. `Device N` is the index `--device` takes; `Picked by automatic selection for` marks the device a backend would use first. `purr sys --json` adds `default_order`, the indexes in the order they are tried when nothing was arranged; `devices[].id` is the index, `devices[].tpe` the type and `devices[].device_id` the device id. ggml does not report the driver or the Vulkan version, so for Vulkan devices Purr asks the Vulkan driver itself and prints `Driver` (the name and version string the driver gives, unmodified) and `Vulkan API`; when it cannot tie the driver's device to ggml's by PCI bus id or unique name, it says so instead of guessing.
- **CPU**: for each instruction set, `Compiled` is what this binary was built with (ggml's compile-time flags) and `This CPU` is what the running processor reports (x86 only; `n/a` where it cannot be asked or ggml does not say). AMX is reported for the build only. `Compiled in but not supported by this CPU` means the build can fail with an illegal instruction; `Supported by this CPU but not compiled in` means the build leaves speed unused.

## Reporting

Open an issue with the "Accelerator report" form. It asks for your hardware, driver versions, the whole `purr sys` output, and two timings of the same clip: `purr -m base --accel vulkan -t 4 a13.mp3` against `purr -m base --accel cpu -t 4 a13.mp3`, with `a13.mp3` from the `samples` folder.
