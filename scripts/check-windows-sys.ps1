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
    $human = & (Join-Path $bare 'purr.exe') sys | Out-String
    $humanExitCode = $LASTEXITCODE
    $json = & (Join-Path $bare 'purr.exe') sys --json | Out-String
    $exitCode = $LASTEXITCODE
}
finally {
    $env:PATH = $savedPath
    foreach ($entry in $savedVariables.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value)
    }
    Remove-Item -Recurse -Force $bare
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
if ($failed) { exit 1 }
