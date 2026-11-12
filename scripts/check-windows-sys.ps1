param(
    [Parameter(Mandatory)][string]$Exe,
    [switch]$ExpectCuda,
    [switch]$ExpectVulkan
)

$ErrorActionPreference = 'Stop'

$bare = Join-Path ([IO.Path]::GetTempPath()) ("purr-bare-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $bare | Out-Null
Copy-Item $Exe (Join-Path $bare 'purr.exe')

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

try {
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    $output = & (Join-Path $bare 'purr.exe') sys | Out-String
    $exitCode = $LASTEXITCODE
}
finally {
    $env:PATH = $savedPath
    foreach ($entry in $savedVariables.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value)
    }
    Remove-Item -Recurse -Force $bare
}

$lines = $output -replace '\x1b\[[0-9;]*m', '' -split "`r?`n" |
    ForEach-Object { ($_ -replace '\s+', ' ').Trim() }
$lines | ForEach-Object { Write-Host $_ }

$failed = $false
function Require([string]$pattern, [string]$what) {
    if (-not ($lines | Where-Object { $_ -match $pattern })) {
        Write-Host "::error::$what"
        $script:failed = $true
    }
}

if ($exitCode -ne 0) {
    Write-Host "::error::purr sys exited with $exitCode without GPU runtimes"
    $failed = $true
}
Require '^\* CPU - Available$' 'the CPU backend is not available'
if ($ExpectCuda) {
    Require '^\* CUDA - Built, but the runtime library cublasLt64_\d+\.dll cannot be loaded(;.*)?$' 'CUDA is not reported as missing cublasLt'
}
if ($ExpectVulkan) {
    Require '^\* Vulkan - (Built|Available)' 'Vulkan is not reported as built'
}
if ($failed) { exit 1 }
