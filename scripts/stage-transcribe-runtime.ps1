param(
    [Parameter(Mandatory, ParameterSetName = 'Directory')][string]$Source,
    [Parameter(Mandatory, ParameterSetName = 'Messages')][string]$CargoMessages,
    [Parameter(Mandatory, ParameterSetName = 'BuildDirectory')][string]$BuildDirectory,
    [Parameter(Mandatory, ParameterSetName = 'Directory')]
    [Parameter(Mandatory, ParameterSetName = 'Messages')]
    [Parameter(Mandatory, ParameterSetName = 'BuildDirectory')][string]$Destination,
    [Parameter(Mandatory, ParameterSetName = 'Check')][string]$Check,
    [ValidateSet('vulkan', 'cuda')][string[]]$Backends = @(),
    [Parameter(ParameterSetName = 'Directory')]
    [Parameter(ParameterSetName = 'Messages')]
    [Parameter(ParameterSetName = 'BuildDirectory')][string]$OptionalDestination
)

$ErrorActionPreference = 'Stop'

$Core = @(
    'transcribe.dll',
    'ggml.dll',
    'ggml-base.dll',
    'ggml-cpu-x64.dll',
    'ggml-cpu-sse42.dll',
    'ggml-cpu-sandybridge.dll',
    'ggml-cpu-haswell.dll',
    'ggml-cpu-skylakex.dll',
    'ggml-cpu-cannonlake.dll',
    'ggml-cpu-cascadelake.dll',
    'ggml-cpu-icelake.dll',
    'ggml-cpu-alderlake.dll'
)

$BackendModules = @{
    vulkan = @('ggml-vulkan.dll')
    cuda   = @('ggml-cuda.dll')
}

# A backend whose runtime libraries are large and not on every machine; with -OptionalDestination its
# modules go there instead of next to the rest, ready to be packed as a runtime pack.
$OptionalBackends = @('cuda')

$Libraries = @($Core) + @($Backends | Sort-Object -Unique | ForEach-Object { $BackendModules[$_] })

function Test-Owned([string]$name) {
    return $name -like 'transcribe*.dll' -or $name -like 'ggml*.dll'
}

function Fail([string]$message) {
    Write-Host "::error::$message"
    exit 1
}

function Test-Optional([string]$name) {
    foreach ($backend in $OptionalBackends) {
        if ($BackendModules[$backend] -contains $name) { return $true }
    }
    return $false
}

if ($PSCmdlet.ParameterSetName -ne 'Check') {
    if ($PSCmdlet.ParameterSetName -eq 'Messages') {
        $outDirs = @(Get-Content -LiteralPath $CargoMessages | Where-Object { $_ -match '"build-script-executed"' } |
            ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.package_id -match 'transcribe-cpp-sys' } |
            ForEach-Object { $_.out_dir } | Select-Object -Unique)
        if ($outDirs.Count -ne 1) { Fail "$CargoMessages names $($outDirs.Count) build script outputs of transcribe-cpp-sys, expected 1" }
        $Source = Join-Path $outDirs[0] 'bin'
    }
    elseif ($PSCmdlet.ParameterSetName -eq 'BuildDirectory') {
        $bins = @(Get-ChildItem -Path (Join-Path $BuildDirectory 'build\transcribe-cpp-sys-*\out\bin') -Directory -ErrorAction SilentlyContinue |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'transcribe.dll') })
        if ($bins.Count -ne 1) { Fail "$BuildDirectory holds $($bins.Count) transcribe-cpp-sys builds with transcribe.dll, expected 1" }
        $Source = $bins[0].FullName
    }
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) { Fail "$Source is not a directory" }
    $missing = @($Libraries | Where-Object { -not (Test-Path -LiteralPath (Join-Path $Source $_) -PathType Leaf) })
    if ($missing.Count -gt 0) { Fail "$Source lacks $($missing -join ', ')" }

    $unlisted = @(Get-ChildItem -LiteralPath $Source -File -Filter *.dll | Where-Object { (Test-Owned $_.Name) -and ($Libraries -notcontains $_.Name) })
    if ($unlisted.Count -gt 0) {
        Fail "$Source holds transcribe.cpp libraries that are not in the list of this script for the backends '$($Backends -join ',')': $($unlisted.Name -join ', ')"
    }

    $destinations = @($Destination)
    if ($OptionalDestination) { $destinations += $OptionalDestination }
    foreach ($directory in $destinations) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
        foreach ($stale in @(Get-ChildItem -LiteralPath $directory -File -Filter *.dll | Where-Object { (Test-Owned $_.Name) })) {
            Remove-Item -LiteralPath $stale.FullName
        }
    }

    $total = 0
    foreach ($name in $Libraries) {
        $target = if ($OptionalDestination -and (Test-Optional $name)) { $OptionalDestination } else { $Destination }
        $item = Copy-Item -LiteralPath (Join-Path $Source $name) -Destination $target -PassThru
        $total += $item.Length
        Write-Host ('{0,-28} {1,12:N0} bytes -> {2}' -f $name, $item.Length, $target)
    }
    Write-Host ('{0,-28} {1,12:N0} bytes in {2} files' -f 'total', $total, $Libraries.Count)
    exit 0
}

if (Test-Path -LiteralPath $Check -PathType Container) {
    $present = @(Get-ChildItem -LiteralPath $Check -File -Filter *.dll | ForEach-Object { $_.Name })
}
elseif (Test-Path -LiteralPath $Check -PathType Leaf) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath $Check).Path)
    try {
        $present = @($zip.Entries | Where-Object { $_.FullName -notmatch '[\\/]' -and $_.Name -like '*.dll' } | ForEach-Object { $_.Name })
    }
    finally {
        $zip.Dispose()
    }
}
else {
    Fail "$Check does not exist"
}

$missing = @($Libraries | Where-Object { $present -notcontains $_ })
$unlisted = @($present | Where-Object { (Test-Owned $_) -and ($Libraries -notcontains $_) })
if ($missing.Count -gt 0) { Fail "$Check lacks $($missing -join ', ')" }
if ($unlisted.Count -gt 0) { Fail "$Check holds transcribe.cpp libraries that are not in the list for the backends '$($Backends -join ',')': $($unlisted -join ', ')" }
Write-Host "$Check holds the $($Libraries.Count) transcribe.cpp libraries of the backends '$($Backends -join ',')' and no other"
