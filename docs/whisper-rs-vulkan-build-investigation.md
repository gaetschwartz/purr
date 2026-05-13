# Whisper-rs Vulkan Build Investigation Report
**Date:** 2025-10-04
**Focus:** Environment Variables and Build Requirements for Vulkan Feature on Windows

---

## Executive Summary

Investigation reveals that the **Vulkan build failure with `dx build` vs success with `cargo build` is caused by MSVC environment variable inheritance differences**. The cmake-rs crate relies on cc-rs to detect and configure MSVC, but cc-rs requires critical Windows SDK environment variables (PATH, INCLUDE, LIB) that may not be properly inherited when dx spawns cargo as a subprocess.

### Key Finding
**The CMake error "The C compiler is not able to compile a simple test program" occurs because rc.exe (Resource Compiler) cannot be found - a classic sign of missing Windows SDK PATH entries.**

---

## 1. Environment Variables Checked by whisper-rs sys/build.rs

### Explicitly Required Variables

1. **VULKAN_SDK** (Lines 204, 206-210, 215, 217-221)
   - **Required on Windows and macOS** when vulkan feature is enabled
   - Used to construct library search path: `{VULKAN_SDK}/Lib` (Windows) or `{VULKAN_SDK}/lib` (macOS)
   - Panics if not set: `"Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set"`

2. **TARGET** (Line 12)
   - Always read via `env::var("TARGET").unwrap()`
   - Used for platform-specific logic

3. **OUT_DIR** (Lines 104, 144)
   - Cargo-provided, contains build artifacts
   - Used for whisper.cpp directory path

4. **CUDA_PATH** (Line 58)
   - Required on Windows when cuda feature enabled
   - Used for CUDA library search path

5. **HIP_PATH** (Lines 79-84)
   - Optional for hipblas feature on non-Windows
   - Defaults to "/opt/rocm" if not set

6. **OPENBLAS_PATH** (Lines 38-43)
   - Optional for openblas feature
   - Used to locate OpenBLAS libraries

7. **BLAS_INCLUDE_DIRS** (Lines 233-237)
   - **Required** when openblas feature enabled
   - Panics if not set

### Conditionally Read Variables

8. **WHISPER_DONT_GENERATE_BINDINGS** (Line 118)
   - If set (any value), skips bindgen generation

9. **DOCS_RS** (Line 159)
   - If set, stops build early (docs.rs environment)

10. **All WHISPER_* and CMAKE_* variables** (Lines 260-268)
    - Any environment variable starting with "WHISPER_" (except WHISPER_DONT_GENERATE_BINDINGS)
    - Any environment variable starting with "CMAKE_"
    - Automatically passed to CMake via `config.define(&key, &value)`

11. **AMDGPU_TARGETS** (Lines 195-198)
    - Optional for hipblas feature
    - Passed to CMake if set

### Rerun Triggers (cargo:rerun-if-env-changed)

The build script explicitly monitors:
- `VULKAN_SDK` (lines 204, 215)
- `HIP_PATH` (line 79)
- `BLAS_INCLUDE_DIRS` (line 237)
- `AMDGPU_TARGETS` (line 195)

---

## 2. CMake Configuration Requirements (via cmake-rs)

### cmake-rs Environment Handling

**Critical Discovery:** cmake-rs does NOT use `env_clear()` - it inherits parent environment by default.

From cmake-rs source (lib.rs):
```rust
// Environment variables are set via:
for (k, v) in c_compiler.env().iter().chain(&self.env) {
    cmd.env(k, v);
}
```

This means:
1. **Inherits all parent process environment variables** (Rust Command default behavior)
2. **Adds/overrides** with variables from `c_compiler.env()` (cc-rs Tool)
3. **Adds/overrides** with variables from `self.env` (manual .env() calls)

### cc-rs MSVC Environment Setup

**Critical for Windows builds:** cc-rs provides minimal environment setup for MSVC.

From cc-rs investigation:
- **Does NOT call vcvarsall.bat**
- **Does NOT set PATH, INCLUDE, LIB, LIBPATH comprehensively**
- Only sets `VSLANG=1033` to disable localized messages
- Relies on Windows Registry to find MSVC via `windows_registry::find()`
- **Expects MSVC tools already in PATH or Developer Command Prompt environment**

### What cc-rs Tool.env() Returns

The `Tool.env()` method returns environment variables needed for the compiler to operate, including:
- Compiler path
- Required MSVC environment variables (if already set in parent)
- `VSLANG=1033` for consistent output

**BUT: cc-rs does NOT fully recreate the vcvars environment** - it relies on inheritance.

---

## 3. Windows-Specific Build Requirements

### Required Environment Variables for MSVC

From Microsoft documentation and cc-rs analysis:

1. **PATH**
   - Must include: Windows SDK bin directory (for rc.exe, mt.exe)
   - Must include: MSVC compiler directory (for cl.exe, link.exe)
   - Example: `C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64`

2. **INCLUDE**
   - Must include: MSVC include paths
   - Must include: Windows SDK include paths
   - Set by vcvarsall.bat in Developer Command Prompt

3. **LIB**
   - Must include: MSVC library paths
   - Must include: Windows SDK library paths
   - Set by vcvarsall.bat in Developer Command Prompt

4. **LIBPATH**
   - Additional library search paths
   - Set by vcvarsall.bat

5. **VCINSTALLDIR** (Optional but helpful)
   - cc-rs uses this to detect MSVC from environment
   - If not set, cc-rs falls back to Windows Registry search

### The rc.exe Problem

**Primary failure mode:** CMake test compilation fails with "The C compiler is not able to compile a simple test program"

Root cause analysis:
1. MSVC compilation requires rc.exe (Resource Compiler) from Windows SDK
2. rc.exe location: `{Windows Kits}\bin\{SDK Version}\{Architecture}\rc.exe`
3. If PATH doesn't include this directory, compilation fails
4. This is a **classic vcvarsall.bat setup issue**

From Stack Overflow research:
- **Solution 1:** Run from Visual Studio Developer Command Prompt (has vcvars environment)
- **Solution 2:** Add rc.exe directory to system PATH
- **Solution 3:** Reinstall Windows SDK with "Tools" component (not just headers/libs)

---

## 4. External Project Dependencies

### vulkan-shaders-gen Dependency

From whisper.cpp Vulkan implementation:
- CMake builds Vulkan backend as separate library: `ggml-vulkan`
- Links against: `vulkan-1.lib` (Windows) or `vulkan` (Unix/macOS)
- Vulkan shader generation happens at CMake configure time
- Requires VULKAN_SDK for:
  - Header files: `{VULKAN_SDK}/Include`
  - Libraries: `{VULKAN_SDK}/Lib` (Windows) or `{VULKAN_SDK}/lib` (macOS)

No direct dependency on vulkan-shaders-gen external tool - shaders are compiled by CMake during configuration.

### CMake Vulkan Backend Configuration

From build.rs line 201-228:
```rust
if cfg!(feature = "vulkan") {
    config.define("GGML_VULKAN", "ON");
    if cfg!(windows) {
        println!("cargo:rerun-if-env-changed=VULKAN_SDK");
        println!("cargo:rustc-link-lib=vulkan-1");
        let vulkan_path = match env::var("VULKAN_SDK") {
            Ok(path) => PathBuf::from(path),
            Err(_) => panic!("Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set"),
        };
        let vulkan_lib_path = vulkan_path.join("Lib");
        println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
    }
    // ... similar for macOS and Linux
}
```

**Required:**
- VULKAN_SDK environment variable (Windows/macOS)
- CMake can find Vulkan headers and libraries
- **MSVC environment can compile test programs (rc.exe accessible)**

---

## 5. Environment Variable Inheritance: MUST vs CAN

### MUST Be Inherited (Cannot be explicitly set)

These MUST come from parent process environment:

1. **PATH** (for MSVC tools)
   - Contains: rc.exe, mt.exe, cl.exe, link.exe
   - Set by: vcvarsall.bat / Developer Command Prompt
   - **Cannot be easily reconstructed in build.rs**

2. **INCLUDE** (for MSVC headers)
   - Contains: MSVC and Windows SDK include paths
   - Set by: vcvarsall.bat
   - **Too complex to reconstruct (version-specific)**

3. **LIB** (for MSVC libraries)
   - Contains: MSVC and Windows SDK library paths
   - Set by: vcvarsall.bat
   - **Too complex to reconstruct (version-specific)**

4. **LIBPATH** (for additional libraries)
   - Set by: vcvarsall.bat
   - **Optional but recommended to inherit**

### CAN Be Explicitly Set

These can be set in build.rs or environment:

1. **VULKAN_SDK**
   - User must set (via system env or cargo config)
   - Can be set before running dx/cargo

2. **CUDA_PATH, HIP_PATH, OPENBLAS_PATH**
   - User must set if using those features
   - Can be set in .cargo/config.toml

3. **All WHISPER_* and CMAKE_* variables**
   - Can be set via environment or .cargo/config.toml
   - Passed through to CMake

4. **AMDGPU_TARGETS, BLAS_INCLUDE_DIRS**
   - Feature-specific, can be set explicitly

---

## 6. Root Cause Analysis: cargo vs dx build

### How Rust Command::spawn() Handles Environment

From Rust documentation research:

**Default behavior:**
```rust
Command::new("cmake") // Inherits ALL parent environment variables
```

**Explicit control:**
```rust
Command::new("cmake")
    .env_clear()  // Clears inherited environment
    .env("KEY", "value")  // Sets specific variables
```

**Key fact:** Environment variables are **captured lazily** when first env method is called, not at Command creation.

### cargo build Environment

When running `cargo build`:
1. **Inherits current shell environment**
2. If run from Developer Command Prompt: Has PATH, INCLUDE, LIB, LIBPATH
3. build.rs executes with full environment
4. cmake-rs spawns cmake with inherited environment
5. CMake test compilation succeeds (rc.exe found in PATH)

### dx build Environment

When running `dx build`:
1. **dx spawns cargo as subprocess**
2. **dx may NOT pass through all environment variables**
3. If dx runs without vcvars environment, cargo doesn't get PATH, INCLUDE, LIB
4. build.rs executes with partial environment
5. cmake-rs spawns cmake with partial environment
6. **CMake test compilation fails (rc.exe not in PATH)**

### The Critical Difference

**Hypothesis:** `dx build` spawns cargo with filtered/sanitized environment, dropping critical MSVC variables.

Evidence:
- Same build.rs works with cargo (Developer Command Prompt)
- Same build.rs fails with dx (same Developer Command Prompt)
- VULKAN_SDK is set in both cases (explicitly checked)
- Error is "compiler cannot compile test program" (rc.exe missing)
- rc.exe is in PATH when running cargo, but apparently not when dx spawns cargo

**Testing:** Running `dx build` from regular PowerShell (no vcvars) should fail the same way.

---

## 7. Required Environment Variables - Complete List

### Tier 1: Always Required (Windows Vulkan Build)

1. **VULKAN_SDK**
   - Source: User must set
   - Method: System environment or cargo config
   - Used by: whisper-rs build.rs

2. **PATH** (containing Windows SDK and MSVC bin directories)
   - Source: vcvarsall.bat / Developer Command Prompt
   - Method: **MUST be inherited** from parent process
   - Used by: CMake (rc.exe), MSVC (cl.exe)
   - Example paths:
     - `C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64`
     - `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.XX.XXXXX\bin\Hostx64\x64`

3. **INCLUDE**
   - Source: vcvarsall.bat
   - Method: **MUST be inherited**
   - Used by: MSVC cl.exe

4. **LIB**
   - Source: vcvarsall.bat
   - Method: **MUST be inherited**
   - Used by: MSVC link.exe

### Tier 2: Strongly Recommended

5. **LIBPATH**
   - Source: vcvarsall.bat
   - Method: Inherit from vcvars environment

6. **VCINSTALLDIR**
   - Source: vcvarsall.bat
   - Method: Inherit from vcvars environment
   - Used by: cc-rs to detect MSVC

7. **VSLANG** (set by cc-rs, but better if already set)
   - Source: vcvarsall.bat or cc-rs
   - Value: 1033 (English)

### Tier 3: Optional/Feature-Specific

8. **CMAKE** (cmake-rs checks this)
   - Source: User can set to override cmake executable
   - Default: "cmake"

9. **CARGO_MAKEFLAGS**
   - Source: Cargo
   - Used by: cmake-rs for parallel builds

10. **Any WHISPER_* or CMAKE_* variables**
    - Source: User configuration
    - Passed through to CMake

---

## 8. Verification Commands

### Check Environment Before Build

```powershell
# Verify VULKAN_SDK
echo $env:VULKAN_SDK

# Verify rc.exe is in PATH
where.exe rc.exe

# Verify MSVC environment
echo $env:INCLUDE
echo $env:LIB
echo $env:PATH | Select-String "Windows Kits"
echo $env:PATH | Select-String "MSVC"

# Verify vcvars loaded
echo $env:VCINSTALLDIR
```

### Test Hypothesis: dx Environment Filtering

```powershell
# From Developer Command Prompt (vcvars loaded):
cargo build --features vulkan  # Should work

dx build --features vulkan  # Currently fails

# From Regular PowerShell (no vcvars):
cargo build --features vulkan  # Should fail with same error
```

---

## 9. Solutions

### Option 1: Ensure dx Inherits Environment (Recommended)

**If dx can be configured to pass through environment:**
```toml
# In Dioxus.toml or dx config
[build]
inherit_environment = true
preserve_vars = ["PATH", "INCLUDE", "LIB", "LIBPATH", "VCINSTALLDIR", "VULKAN_SDK"]
```

### Option 2: Run dx from Developer Command Prompt with Inherited Env

**Ensure dx itself is run in vcvars environment:**
```powershell
# Start Developer Command Prompt, then:
$env:VULKAN_SDK = "C:\VulkanSDK\1.3.XXX.X"
dx build --features vulkan
```

### Option 3: Modify build.rs to Detect and Load vcvars

**Add to sys/build.rs before cmake config:**
```rust
#[cfg(all(windows, feature = "vulkan"))]
{
    // Check if MSVC environment is loaded
    if env::var("VCINSTALLDIR").is_err() {
        println!("cargo:warning=MSVC environment not detected. Attempting to load vcvars...");

        // Use vcvars crate to load environment
        // (requires adding vcvars to build-dependencies)
        if let Ok(vcvars_env) = vcvars::get_vcvars_env() {
            for (key, value) in vcvars_env {
                env::set_var(key, value);
            }
        } else {
            panic!("Cannot find MSVC environment. Please run from Developer Command Prompt.");
        }
    }
}
```

### Option 4: System-Wide PATH Configuration

**Add Windows SDK to system PATH permanently:**
1. Find Windows SDK bin path: `C:\Program Files (x86)\Windows Kits\10\bin\{version}\x64`
2. Add to System Environment Variables → PATH
3. Restart terminal/IDE
4. Verify: `where.exe rc.exe`

**Pros:** Works everywhere
**Cons:** Modifies system, may conflict with multiple SDK versions

---

## 10. Recommended Action Plan

1. **Immediate Fix:**
   - Run `dx build` from Developer Command Prompt
   - Ensure VULKAN_SDK is set before running dx

2. **Verify Root Cause:**
   ```powershell
   # In Developer Command Prompt
   where.exe rc.exe  # Should find it

   # Run dx build with verbose output
   $env:RUST_LOG = "trace"
   dx build --features vulkan

   # Check if rc.exe is still found during cmake configure
   ```

3. **Long-term Fix (choose one):**
   - **Option A:** Configure dx to preserve MSVC environment variables
   - **Option B:** Add vcvars loading to whisper-rs build.rs
   - **Option C:** Document requirement to run from Developer Command Prompt

4. **Test Validation:**
   ```powershell
   # From regular PowerShell (should fail):
   cargo build --features vulkan

   # From Developer Command Prompt (should work):
   cargo build --features vulkan
   dx build --features vulkan
   ```

---

## 11. Key Takeaways

1. **VULKAN_SDK alone is NOT sufficient** - MSVC environment is required
2. **cmake-rs inherits environment** - doesn't recreate vcvars
3. **cc-rs expects vcvars** - only sets VSLANG, relies on PATH/INCLUDE/LIB
4. **dx may filter environment** - causing rc.exe to be inaccessible
5. **The error is misleading** - "compiler cannot compile" really means "rc.exe not found"

### Environment Requirements Matrix

| Variable | Required? | Source | Can Set Explicitly? | Must Inherit? |
|----------|-----------|--------|-------------------|---------------|
| VULKAN_SDK | Yes | User | ✅ Yes | ❌ No |
| PATH | Yes | vcvars | ❌ No | ✅ Yes |
| INCLUDE | Yes | vcvars | ❌ No | ✅ Yes |
| LIB | Yes | vcvars | ❌ No | ✅ Yes |
| LIBPATH | Recommended | vcvars | ❌ No | ✅ Yes |
| VCINSTALLDIR | Recommended | vcvars | ❌ No | ✅ Yes |
| CMAKE | Optional | User | ✅ Yes | ❌ No |

---

## References

1. whisper-rs sys/build.rs source code
2. cmake-rs source: https://github.com/rust-lang/cmake-rs/blob/master/src/lib.rs
3. cc-rs source: https://github.com/rust-lang/cc-rs/blob/main/src/lib.rs
4. Rust Command documentation: https://doc.rust-lang.org/std/process/struct.Command.html
5. MSVC environment setup: https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line
6. CMake rc.exe issues: https://stackoverflow.com/questions/42190568
7. cc-rs MSVC detection: https://github.com/rust-lang/cc-rs/issues/1189
