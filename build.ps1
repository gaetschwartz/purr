# Whisper UI PowerShell Build Script
# Usage: .\build.ps1 [command]

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

function Show-Help {
    Write-Host "Whisper UI Development Commands:" -ForegroundColor Green
    Write-Host ""
    Write-Host "Building:" -ForegroundColor Yellow
    Write-Host "  build         - Build debug version"
    Write-Host "  build-release - Build optimized release version"
    Write-Host ""
    Write-Host "Testing:" -ForegroundColor Yellow
    Write-Host "  test          - Run all tests"
    Write-Host "  test-verbose  - Run tests with verbose output"
    Write-Host "  test-core     - Run only core library tests"
    Write-Host "  test-cli      - Run only CLI tests"
    Write-Host ""
    Write-Host "Development:" -ForegroundColor Yellow
    Write-Host "  lint          - Run clippy linter"
    Write-Host "  format        - Format code with rustfmt"
    Write-Host "  check         - Check code without building"
    Write-Host ""
    Write-Host "Setup:" -ForegroundColor Yellow
    Write-Host "  install       - Install binary to cargo bin directory"
    Write-Host "  setup-models  - Download Whisper models"
    Write-Host "  clean         - Clean build artifacts"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\build.ps1 build-release"
    Write-Host "  .\build.ps1 test"
    Write-Host "  .\build.ps1 setup-models"
}

function Invoke-Build {
    Write-Host "Building debug version..." -ForegroundColor Blue
    cargo build
}

function Invoke-BuildRelease {
    Write-Host "Building release version..." -ForegroundColor Blue
    cargo build --release
}

function Invoke-Test {
    Write-Host "Running tests..." -ForegroundColor Blue
    cargo test
}

function Invoke-TestVerbose {
    Write-Host "Running tests with verbose output..." -ForegroundColor Blue
    cargo test -- --nocapture
}

function Invoke-TestCore {
    Write-Host "Running core library tests..." -ForegroundColor Blue
    cargo test -p whisper-ui-core
}

function Invoke-TestCli {
    Write-Host "Running CLI tests..." -ForegroundColor Blue
    cargo test -p whisper-ui-cli
}

function Invoke-Lint {
    Write-Host "Running clippy linter..." -ForegroundColor Blue
    cargo clippy -- -D warnings
}

function Invoke-Format {
    Write-Host "Formatting code..." -ForegroundColor Blue
    cargo fmt
}

function Invoke-Check {
    Write-Host "Checking code..." -ForegroundColor Blue
    cargo check
}

function Invoke-Install {
    Write-Host "Installing binary..." -ForegroundColor Blue
    cargo install --path whisper-ui-cli
}

function Invoke-SetupModels {
    Write-Host "Setting up Whisper models..." -ForegroundColor Blue
    
    if (!(Test-Path "models")) {
        New-Item -ItemType Directory -Path "models"
        Write-Host "Created models directory" -ForegroundColor Green
    }
    
    $models = @(
        @{
            name = "ggml-tiny.en.bin"
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
            size = "39 MB"
        },
        @{
            name = "ggml-base.en.bin" 
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
            size = "142 MB"
        }
    )
    
    foreach ($model in $models) {
        $path = "models\$($model.name)"
        if (!(Test-Path $path)) {
            Write-Host "Downloading $($model.name) ($($model.size))..." -ForegroundColor Yellow
            try {
                Invoke-WebRequest -Uri $model.url -OutFile $path -UseBasicParsing
                Write-Host "Downloaded $($model.name)" -ForegroundColor Green
            }
            catch {
                Write-Host "Failed to download $($model.name): $($_.Exception.Message)" -ForegroundColor Red
            }
        }
        else {
            Write-Host "$($model.name) already exists" -ForegroundColor Green
        }
    }
    
    Write-Host "Model setup completed!" -ForegroundColor Green
}

function Invoke-Clean {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Blue
    cargo clean
}

function Invoke-Example {
    Write-Host "Example usage:" -ForegroundColor Cyan
    Write-Host "1. Build the project: .\build.ps1 build-release"
    Write-Host "2. Download models: .\build.ps1 setup-models" 
    Write-Host "3. Run transcription: .\target\release\whisper-ui-cli.exe audio.wav"
}

function Invoke-DevSetup {
    Write-Host "Setting up development environment..." -ForegroundColor Blue
    Invoke-SetupModels
    Invoke-Build
    Invoke-Test
    Write-Host "Development environment ready!" -ForegroundColor Green
}

function Invoke-Release {
    Write-Host "Preparing release build..." -ForegroundColor Blue
    Invoke-Clean
    Invoke-Lint
    Invoke-Test
    Invoke-BuildRelease
    Write-Host "Release build completed successfully!" -ForegroundColor Green
}

# Main command dispatcher
switch ($Command.ToLower()) {
    "help" { Show-Help }
    "build" { Invoke-Build }
    "build-release" { Invoke-BuildRelease }
    "test" { Invoke-Test }
    "test-verbose" { Invoke-TestVerbose }
    "test-core" { Invoke-TestCore }
    "test-cli" { Invoke-TestCli }
    "lint" { Invoke-Lint }
    "format" { Invoke-Format }
    "check" { Invoke-Check }
    "install" { Invoke-Install }
    "setup-models" { Invoke-SetupModels }
    "clean" { Invoke-Clean }
    "example" { Invoke-Example }
    "dev-setup" { Invoke-DevSetup }
    "release" { Invoke-Release }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host ""
        Show-Help
        exit 1
    }
}
