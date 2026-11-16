param(
    [Parameter(Mandatory)][string]$Exe,
    [switch]$ExpectCuda,
    [switch]$ExpectVulkan,
    [switch]$ExpectTranscribe,
    [ValidateSet('vulkan', 'cuda')][string[]]$ExpectTranscribeBackends = @()
)

$ErrorActionPreference = 'Stop'

function New-Copy([string]$prefix, [string[]]$excludeDlls = @()) {
    $dir = Join-Path ([IO.Path]::GetTempPath()) ("$prefix-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $dir | Out-Null
    Copy-Item $Exe (Join-Path $dir 'purr.exe')
    Copy-Item (Join-Path (Split-Path -Parent (Resolve-Path $Exe)) '*.dll') $dir -Exclude $excludeDlls
    return $dir
}

$bare = Join-Path ([IO.Path]::GetTempPath()) ("purr-bare-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $bare | Out-Null
Copy-Item $Exe (Join-Path $bare 'purr.exe')
$withLibraries = $null
$brokenLoader = $null
$withoutCudaRuntime = $null
if ($ExpectTranscribe) {
    $withLibraries = New-Copy 'purr-libs'
    if ($ExpectTranscribeBackends -contains 'vulkan') {
        $brokenLoader = New-Copy 'purr-broken-loader'
        Set-Content -LiteralPath (Join-Path $brokenLoader 'vulkan-1.dll') -Value 'not a library'
    }
    if ($ExpectTranscribeBackends -contains 'cuda') {
        $withoutCudaRuntime = New-Copy 'purr-no-cuda' @('cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll')
    }
}

$savedPath = $env:PATH
$savedVariables = @{}
foreach ($name in 'CUDA_PATH', 'PURR_RUNTIME_DIR') {
    $savedVariables[$name] = [Environment]::GetEnvironmentVariable($name)
    [Environment]::SetEnvironmentVariable($name, $null)
}
foreach ($variable in Get-ChildItem Env: | Where-Object { $_.Name -like 'CUDA_PATH_*' }) {
    $savedVariables[$variable.Name] = $variable.Value
    [Environment]::SetEnvironmentVariable($variable.Name, $null)
}

function Invoke-Sys([string]$dir) {
    $output = & (Join-Path $dir 'purr.exe') sys --json | Out-String
    $exitCode = $LASTEXITCODE
    $engines = & (Join-Path $dir 'purr.exe') engines --json | Out-String
    return @{ ExitCode = $exitCode; Json = $output; Engines = $engines }
}

try {
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    $human = & (Join-Path $bare 'purr.exe') sys | Out-String
    $humanExitCode = $LASTEXITCODE
    $json = & (Join-Path $bare 'purr.exe') sys --json | Out-String
    $exitCode = $LASTEXITCODE
    if ($ExpectTranscribe) {
        $loaded = Invoke-Sys $withLibraries
        if ($brokenLoader) { $broken = Invoke-Sys $brokenLoader }
        if ($withoutCudaRuntime) { $noCuda = Invoke-Sys $withoutCudaRuntime }
    }
}
finally {
    $env:PATH = $savedPath
    foreach ($entry in $savedVariables.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value)
    }
    Remove-Item -Recurse -Force $bare
    foreach ($dir in $withLibraries, $brokenLoader, $withoutCudaRuntime) {
        if ($dir) { Remove-Item -Recurse -Force $dir }
    }
}

Write-Host $human

$failed = $false
function Require([bool]$condition, [string]$what) {
    if (-not $condition) {
        Write-Host "::error::$what"
        $script:failed = $true
    }
}

function Status($backend) {
    if ($backend -is [string]) { return $backend }
    return @($backend.PSObject.Properties.Name)[0]
}

function Get-TranscribeEngine($run) {
    return @(($run.Json | ConvertFrom-Json).engines) | Where-Object { $_.engine -eq 'transcribe-cpp' }
}

function Get-DeviceBackends($engine) {
    return @($engine.status.ready.devices | ForEach-Object { $_.backend })
}

Require ($humanExitCode -eq 0 -and $exitCode -eq 0) "purr sys exited with $humanExitCode (text) and $exitCode (--json) without GPU runtimes"
$backends = ($json | ConvertFrom-Json).backends
Require ((Status $backends.cpu) -eq 'Available') 'the CPU backend is not available'
if ($ExpectCuda) {
    Require ((Status $backends.cuda) -eq 'RuntimeLibraryMissing') 'CUDA is not reported as missing a runtime library'
    Require ($backends.cuda.RuntimeLibraryMissing.library -match '^cublasLt64_\d+\.dll$') 'CUDA is not reported as missing cublasLt'
}
if ($ExpectVulkan) {
    Require ((Status $backends.vulkan) -ne 'Disabled') 'Vulkan is not reported as built'
}
if ($ExpectTranscribe) {
    $engine = @(($json | ConvertFrom-Json).engines) | Where-Object { $_.engine -eq 'transcribe-cpp' }
    Require ($null -ne $engine) 'purr sys --json does not list the transcribe.cpp engine'
    Require ($engine.status.library_missing.library -eq 'transcribe.dll') 'the transcribe.cpp engine is not reported as missing transcribe.dll'
    Require ($human -match 'transcribe\.cpp engine - Not loadable') 'purr sys does not say the transcribe.cpp engine is not loadable'
    Require (@(($json | ConvertFrom-Json).hardware | Where-Object { $_.seen_by.engine -contains 'transcribe-cpp' }).Count -eq 0) 'purr sys --json lists a device for the transcribe.cpp engine that cannot be loaded'
    Require ($loaded.ExitCode -eq 0) "purr sys exited with $($loaded.ExitCode) next to the transcribe.cpp libraries"
    $ready = Get-TranscribeEngine $loaded
    Require ($ready.status.ready.version -match '^\d+\.\d+\.\d+') 'the transcribe.cpp engine is not loaded next to its libraries'
    Require (@($ready.status.ready.devices).Count -ge 1) 'the transcribe.cpp engine lists no device next to its libraries'
    $shared = @(($loaded.Json | ConvertFrom-Json).hardware | Where-Object { $_.seen_by.engine -contains 'whisper-cpp' -and $_.seen_by.engine -contains 'transcribe-cpp' })
    Require ($shared.Count -ge 1) 'no device is seen by both the whisper.cpp and the transcribe.cpp engine next to the libraries'
    foreach ($backend in $ExpectTranscribeBackends) {
        $built = @((Get-TranscribeEngine @{ Json = $loaded.Engines }).build.backends | Where-Object { $_.backend -eq $backend -and $_.status -eq 'built' })
        Require ($built.Count -eq 1) "the transcribe.cpp engine is not reported as built with $backend"
    }
    if ($broken) {
        Require ($broken.ExitCode -eq 0) "purr sys exited with $($broken.ExitCode) with a Vulkan loader that cannot be loaded next to the transcribe.cpp libraries"
        $engine = Get-TranscribeEngine $broken
        Require ($null -ne $engine.status.ready) 'the transcribe.cpp engine is not ready with a Vulkan loader that cannot be loaded'
        Require (-not ((Get-DeviceBackends $engine) -contains 'vulkan')) 'the transcribe.cpp engine lists a Vulkan device although the loader cannot be loaded'
        Require ((Get-DeviceBackends $engine) -contains 'cpu') 'the transcribe.cpp engine lists no CPU without the Vulkan loader'
    }
    if ($noCuda) {
        Require ($noCuda.ExitCode -eq 0) "purr sys exited with $($noCuda.ExitCode) without the CUDA runtime next to the transcribe.cpp libraries"
        $engine = Get-TranscribeEngine $noCuda
        Require ($null -ne $engine.status.ready) 'the transcribe.cpp engine is not ready without the CUDA runtime'
        Require (-not ((Get-DeviceBackends $engine) -contains 'cuda')) 'the transcribe.cpp engine lists a CUDA device without the CUDA runtime'
        Require ((Get-DeviceBackends $engine) -contains 'cpu') 'the transcribe.cpp engine lists no CPU without the CUDA runtime'
    }
}
if ($failed) { exit 1 }
