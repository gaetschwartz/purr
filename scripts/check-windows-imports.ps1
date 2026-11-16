param(
    [Parameter(Mandatory)][string]$Exe,
    [string[]]$Forbidden = @(),
    [string[]]$ExpectDelayLoaded = @(),
    [switch]$Module
)

$ErrorActionPreference = 'Stop'

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

if ($Module) {
    # ggml skips a module whose import cannot be loaded, so a module may import the GPU runtime, ggml and Windows only
    $allowed = '^ggml-base\.dll$', '^vulkan-1\.dll$', '^nvcuda\.dll$', '^cudart64_\d+\.dll$', '^cublas64_\d+\.dll$', '^cublasLt64_\d+\.dll$',
        '^vcruntime140', '^msvcp140', '^api-ms-win-'
    foreach ($dll in $hard) {
        $system = Test-Path -LiteralPath (Join-Path $env:SystemRoot "System32\$dll")
        $listed = @($allowed | Where-Object { $dll -match $_ }).Count -gt 0
        if (-not $system -and -not $listed) {
            Write-Host "::error::$Exe imports $dll, which is neither a Windows library nor one a ggml backend module may need"
            $failed = $true
        }
    }
    if ($delayed.Count -gt 0) {
        Write-Host "::error::$Exe delay-loads $($delayed -join ', '), a module has no delay-load hook to catch a missing library"
        $failed = $true
    }
}
else {
    $Forbidden += '^vcruntime140', '^msvcp140', '^concrt140\.dll$', '^vcomp140\.dll$', '^transcribe\.dll$', '^ggml'
    foreach ($dll in $hard) {
        foreach ($pattern in $Forbidden) {
            if ($dll -match $pattern) {
                Write-Host "::error::$Exe has a hard dependency on $dll"
                $failed = $true
            }
        }
    }
}
foreach ($dll in $ExpectDelayLoaded) {
    if ($delayed -notcontains $dll) {
        Write-Host "::error::$Exe does not delay-load $dll"
        $failed = $true
    }
}
if ($hard.Count -eq 0) {
    Write-Host "::error::no imports found in $Exe"
    $failed = $true
}
if ($failed) { exit 1 }
