param(
    [Parameter(Mandatory, ParameterSetName = 'Stage')][string]$Source,
    [Parameter(Mandatory, ParameterSetName = 'Stage')][string]$Destination,
    [Parameter(Mandatory, ParameterSetName = 'Check')][string]$Check
)

$ErrorActionPreference = 'Stop'

$Libraries = @(
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

function Test-Owned([string]$name) {
    return $name -like 'transcribe*.dll' -or $name -like 'ggml*.dll'
}

function Fail([string]$message) {
    Write-Host "::error::$message"
    exit 1
}

if ($PSCmdlet.ParameterSetName -eq 'Stage') {
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) { Fail "$Source is not a directory" }
    $missing = @($Libraries | Where-Object { -not (Test-Path -LiteralPath (Join-Path $Source $_) -PathType Leaf) })
    if ($missing.Count -gt 0) { Fail "$Source lacks $($missing -join ', ')" }

    $unlisted = @(Get-ChildItem -LiteralPath $Source -File -Filter *.dll | Where-Object { (Test-Owned $_.Name) -and ($Libraries -notcontains $_.Name) })
    if ($unlisted.Count -gt 0) {
        Fail "$Source holds transcribe.cpp libraries that are not in the list of this script: $($unlisted.Name -join ', ')"
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    foreach ($stale in @(Get-ChildItem -LiteralPath $Destination -File -Filter *.dll | Where-Object { (Test-Owned $_.Name) -and ($Libraries -notcontains $_.Name) })) {
        Remove-Item -LiteralPath $stale.FullName
    }

    $total = 0
    foreach ($name in $Libraries) {
        $item = Copy-Item -LiteralPath (Join-Path $Source $name) -Destination $Destination -PassThru
        $total += $item.Length
        Write-Host ('{0,-28} {1,12:N0} bytes' -f $name, $item.Length)
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
if ($unlisted.Count -gt 0) { Fail "$Check holds transcribe.cpp libraries that are not in the list: $($unlisted -join ', ')" }
Write-Host "$Check holds the $($Libraries.Count) transcribe.cpp libraries and no other"
