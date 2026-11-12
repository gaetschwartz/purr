param(
    [Parameter(Mandatory)][string]$Exe,
    [Parameter(Mandatory)][string[]]$Forbidden
)

$ErrorActionPreference = 'Stop'

$Forbidden += '^vcruntime140', '^msvcp140', '^concrt140\.dll$', '^vcomp140\.dll$'

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$dumpbin = & $vswhere -latest -find 'VC\Tools\MSVC\**\bin\Hostx64\x64\dumpbin.exe' | Select-Object -First 1
if (-not $dumpbin) { throw 'dumpbin.exe not found' }

$report = & $dumpbin /nologo /dependents $Exe
if ($LASTEXITCODE -ne 0) { throw "dumpbin failed on $Exe" }

$hard = @()
$delayed = @()
$section = ''
foreach ($line in $report) {
    if ($line -match '^\s*Image has the following dependencies:') { $section = 'hard' }
    elseif ($line -match '^\s*Image has the following delay load dependencies:') { $section = 'delayed' }
    elseif ($line -match '^\s*Summary') { $section = '' }
    elseif ($line -match '^\s+(\S+\.dll)\s*$') {
        if ($section -eq 'hard') { $hard += $Matches[1] }
        elseif ($section -eq 'delayed') { $delayed += $Matches[1] }
    }
}

Write-Host "$Exe imports: $($hard -join ' ')"
Write-Host "$Exe delay-loads: $($delayed -join ' ')"

$failed = $false
foreach ($dll in $hard) {
    foreach ($pattern in $Forbidden) {
        if ($dll -match $pattern) {
            Write-Host "::error::$Exe has a hard dependency on $dll"
            $failed = $true
        }
    }
}
if ($hard.Count -eq 0) {
    Write-Host "::error::no imports found in $Exe"
    $failed = $true
}
if ($failed) { exit 1 }
