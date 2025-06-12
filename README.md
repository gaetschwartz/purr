# Purr - Audio Transcription Suite

A comprehensive audio transcription suite built with Rust, featuring CLI tools and cross-platform GUI applications powered by OpenAI's Whisper AI.

## Projects Overview

This workspace contains multiple components for audio transcription:

### 🖥️ **Desktop App** (`desktop/`)
Native desktop application with drag-and-drop transcription interface

### 🌐 **Web App** (`web/`)  
Web-based transcription interface (future implementation)

### 📱 **Mobile App** (`mobile/`)
Mobile transcription app (future implementation)

### 🎨 **UI Components** (`ui/`)
Shared cross-platform UI components built with Dioxus

### 🔧 **CLI Tool** (`purr/`)
Command-line interface for batch transcription and model management

### ⚙️ **Core Engine** (`purr-core/`)
Rust library providing Whisper AI transcription functionality

### 🌍 **API** (`api/`)
Shared backend logic and server functions

## 🚀 Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) (latest stable)
- [Dioxus CLI](https://github.com/DioxusLabs/dioxus): `cargo install dioxus-cli`

### Running the Desktop App

```bash
cd desktop
dx serve --platform desktop
```

### Using the CLI Tool

```bash
# Install CLI tool
cargo install --path purr

# Download a model (first time setup)
purr models download base

# Transcribe an audio file
purr path/to/audio.mp3

# Streaming transcription
purr --no-stream path/to/audio.mp3
```

### Development

```bash
# Build all workspace members
cargo build

# Run tests
cargo test

# Run with GPU acceleration
cargo run --features vulkan  # or cuda, metal, coreml
```

## 🎵 Features

### Core Transcription
- **High-quality transcription** using OpenAI's Whisper AI
- **Multiple model sizes** (tiny, base, small, medium, large-v3)
- **GPU acceleration** support (CUDA, Vulkan, Metal, CoreML)
- **Multiple output formats** (text, JSON, SRT subtitles)
- **Language detection** and manual language specification

### Desktop GUI
- **🖱️ Drag & Drop**: Drop audio files directly onto the interface
- **📁 File Browser**: Manual file selection with format filtering
- **📋 Clipboard Integration**: Copy transcription results instantly
- **🔄 Real-time Status**: Visual feedback during processing
- **❌ Error Handling**: Clear error messages and recovery

### CLI Interface
- **Batch processing** of multiple files
- **Streaming output** for real-time results
- **Model management** (download, list, delete)
- **GPU device listing** and status checks
- **Flexible output options** (stdout, file, different formats)

### Cross-Platform Support
- **Desktop**: Windows, macOS, Linux
- **Web**: Browser-based interface (planned)
- **Mobile**: iOS and Android apps (planned)

## 📁 Architecture

```
purr/
├── desktop/           # Desktop app (Dioxus native)
├── web/              # Web app (Dioxus + WASM)
├── mobile/           # Mobile app (Dioxus + Mobile)
├── ui/               # Shared UI components
│   └── transcription.rs  # Cross-platform transcription component
├── api/              # Shared backend logic
├── purr/             # CLI application
├── purr-core/        # Core transcription engine
└── samples/          # Example audio files
```

### Cross-Platform Component Design

The transcription functionality is implemented as a shared component in `ui/src/transcription.rs` with platform-specific adaptations:

- **Feature Gates**: Desktop-specific dependencies (file dialogs, clipboard) are optional
- **Platform Detection**: Runtime adaptation for different platforms
- **Shared Logic**: Core transcription flow works across all platforms
- **UI Consistency**: Same interface and behavior across desktop, web, and mobile

## 🔧 Supported Audio Formats

- **MP3** - Most common format
- **WAV** - Uncompressed audio
- **M4A** - Apple audio format
- **FLAC** - Lossless compression
- **OGG** - Open source format

## ⚡ Performance

### GPU Acceleration
- **CUDA** - NVIDIA graphics cards
- **Vulkan** - Cross-platform GPU API
- **Metal** - Apple Silicon and Intel Macs
- **CoreML** - Apple's machine learning framework

### Model Sizes
- **tiny** - 39MB, fastest (good for real-time)
- **base** - 142MB, recommended balance
- **small** - 466MB, better accuracy
- **medium** - 1.5GB, high accuracy
- **large-v3** - 3.0GB, best accuracy

## 🛠️ Development Guide

### Building Components

```bash
# Desktop app
cd desktop && dx serve

# CLI tool
cargo run --bin purr -- --help

# Core library tests
cd purr-core && cargo test

# Cross-platform UI development
cd ui && cargo check --features desktop
```

### Adding New Features

1. **Core Logic**: Add to `purr-core/` for shared functionality
2. **UI Components**: Add to `ui/` for cross-platform interface elements
3. **Platform-Specific**: Implement in respective platform directories
4. **CLI Features**: Add to `purr/` for command-line functionality

### Feature Flags

```toml
[features]
default = []
desktop = ["rfd", "copypasta"]  # Desktop-specific dependencies
vulkan = ["purr-core/vulkan"]   # Vulkan GPU acceleration
cuda = ["purr-core/cuda"]       # CUDA GPU acceleration
metal = ["purr-core/metal"]     # Metal GPU acceleration
coreml = ["purr-core/coreml"]   # CoreML acceleration
```

## 📖 Documentation

- **CLI Usage**: See `purr --help` and `purr models --help`
- **Core API**: Check `purr-core/src/lib.rs` for library documentation
- **UI Components**: Review `ui/src/` for component APIs
- **Examples**: Sample code in `examples/` directories

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test across platforms (`cargo test`, `dx check`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style
- Follow Rust conventions (`cargo fmt`, `cargo clippy`)
- Add tests for new functionality
- Update documentation for API changes
- Ensure cross-platform compatibility

## 🐛 Troubleshooting

### Model Issues
```bash
# Download a model manually
purr models download base

# Check available models
purr models list

# Check downloaded models
purr models downloaded
```

### GPU Issues
```bash
# Check GPU status
purr gpu status

# List available devices
purr gpu list

# Disable GPU if needed
purr --no-gpu audio.mp3
```

### File Format Issues
- Ensure your audio file is in a supported format
- Try converting with tools like `ffmpeg` if needed
- Check file permissions and accessibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://openai.com/research/whisper) - The amazing speech recognition model
- [Dioxus](https://dioxuslabs.com/) - Cross-platform Rust GUI framework
- [whisper-rs](https://github.com/tazz4843/whisper-rs) - Rust bindings for Whisper
- [Rust Community](https://www.rust-lang.org/community) - For amazing tools and libraries

---

**Built with ❤️ in Rust** | **Powered by Whisper AI** | **Cross-platform with Dioxus**