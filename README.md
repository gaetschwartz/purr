# Whisper UI

A high-performance audio transcription tool built with Rust, using OpenAI's Whisper model via whisper.cpp bindings and FFmpeg for audio processing.

## Features

- 🚀 **High Performance**: Built with Rust for maximum performance
- 🎯 **GPU Acceleration**: CUDA/OpenCL support for faster transcription
- 🎵 **Multiple Audio Formats**: Supports MP3, WAV, FLAC, M4A, and more via FFmpeg
- 🌍 **Multi-language**: Support for 99+ languages with auto-detection
- 📝 **Multiple Output Formats**: Text, JSON, and SRT subtitle formats
- ⚡ **Async Processing**: Non-blocking audio processing and transcription
- 🔧 **Configurable**: Extensive configuration options for fine-tuning

## Architecture

The project consists of two main crates:

- **`whisper-ui-core`**: Core library with transcription functionality
- **`whisper-ui-cli`**: Command-line interface using the core library

## Prerequisites

### System Dependencies

#### Windows
```powershell
# Install FFmpeg (using Chocolatey)
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg libavformat-dev libavcodec-dev libavutil-dev libswresample-dev pkg-config
```

#### macOS
```bash
brew install ffmpeg pkg-config
```

### Whisper Model

You'll need a Whisper model file. Download from the official repository:

```bash
# Create models directory
mkdir models

# Download base English model (39 MB)
curl -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin" -o models/ggml-base.en.bin

# Or download base multilingual model (142 MB)
curl -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin" -o models/ggml-base.bin
```

Available models:
- `tiny.en` / `tiny` - 39 MB / 39 MB
- `base.en` / `base` - 142 MB / 142 MB  
- `small.en` / `small` - 466 MB / 466 MB
- `medium.en` / `medium` - 1.5 GB / 1.5 GB
- `large-v1` / `large-v2` / `large-v3` - 2.9 GB

## Installation

### Building from Source

```bash
# Clone the repository
git clone <repository-url>
cd whisper-ui

# Build the project
cargo build --release

# The binary will be at target/release/whisper-ui-cli
```

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with verbose output
cargo test -- --nocapture

# Run only core library tests
cargo test -p whisper-ui-core

# Run only CLI tests  
cargo test -p whisper-ui-cli
```

## Usage

### Basic Usage

```bash
# Transcribe an audio file
./target/release/whisper-ui-cli audio.wav

# Specify a model
./target/release/whisper-ui-cli audio.mp3 --model models/ggml-base.en.bin

# Specify language
./target/release/whisper-ui-cli audio.wav --language en
```

### Advanced Usage

```bash
# JSON output with timestamps
./target/release/whisper-ui-cli audio.wav --output json --timestamps

# SRT subtitle format
./target/release/whisper-ui-cli audio.wav --output srt

# Disable GPU acceleration
./target/release/whisper-ui-cli audio.wav --no-gpu

# Set number of threads
./target/release/whisper-ui-cli audio.wav --threads 4

# Verbose output
./target/release/whisper-ui-cli audio.wav --verbose

# All options combined
./target/release/whisper-ui-cli audio.wav \\
  --model models/ggml-base.en.bin \\
  --language en \\
  --output json \\
  --timestamps \\
  --threads 8 \\
  --temperature 0.2 \\
  --verbose
```

### CLI Options

```
Usage: whisper-ui-cli [OPTIONS] <AUDIO_FILE>

Arguments:
  <AUDIO_FILE>  Path to the audio file to transcribe

Options:
  -m, --model <MODEL>              Path to the Whisper model file
  -l, --language <LANGUAGE>        Language code (e.g., en, es, fr)
      --no-gpu                     Disable GPU acceleration
  -t, --threads <THREADS>          Number of threads to use
  -o, --output <OUTPUT>            Output format [default: text] [possible values: text, json, srt]
      --timestamps                 Include timestamps in output (text format only)
      --word-timestamps            Include word-level timestamps (if supported)
      --temperature <TEMPERATURE>  Temperature for sampling [default: 0.0]
  -v, --verbose                    Verbose output
  -h, --help                       Print help
  -V, --version                    Print version
```

## Library Usage

You can also use the core library in your own Rust projects:

```toml
[dependencies]
whisper-ui-core = { path = "path/to/whisper-ui-core" }
tokio = { version = "1.0", features = ["full"] }
```

```rust
use whisper_ui_core::{transcribe_audio_file, TranscriptionConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = TranscriptionConfig::new()
        .with_language("en")
        .with_gpu(true)
        .with_threads(4);

    let result = transcribe_audio_file("audio.wav", Some(config)).await?;
    
    println!("Transcription: {}", result.text);
    println!("Duration: {:.2}s", result.audio_duration);
    println!("Processing time: {:.2}s", result.processing_time);
    
    Ok(())
}
```

## Performance Tips

1. **GPU Acceleration**: Enable GPU acceleration for significantly faster processing
2. **Model Selection**: Use smaller models (tiny, base) for faster processing, larger models (large) for better accuracy
3. **Threading**: Set `--threads` to match your CPU cores for optimal performance
4. **Audio Format**: WAV files typically process faster than compressed formats

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Ensure FFmpeg is installed and in your PATH
2. **Model not found**: Download a Whisper model or specify the correct path
3. **GPU errors**: Try using `--no-gpu` flag to disable GPU acceleration
4. **Out of memory**: Use a smaller model or disable GPU acceleration

### Performance Monitoring

Use the `--verbose` flag to see detailed performance information:

```bash
./target/release/whisper-ui-cli audio.wav --verbose
```

This shows:
- Audio duration
- Processing time  
- Real-time factor (processing_time / audio_duration)
- Number of segments
- GPU acceleration status

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run `cargo test`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - The original Whisper model
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - C++ implementation  
- [whisper-rs](https://github.com/tazz4843/whisper-rs) - Rust bindings
- [FFmpeg](https://ffmpeg.org/) - Audio processing
