# NexaStory Desktop

**Windows ONLY** - AI-powered creative writing assistant that runs 100% locally with GGUF models.

## Features

- **100% Local** - Your data never leaves your computer, no internet required
- **Local AI Models** - Use any GGUF model
- **Duo Model (Speculative Decoding)** - Use two models for 2-4x faster generation
- **Native llama.cpp** - No external server required (llama-cpp-2 v0.1.143)
- **GPU Acceleration** - CUDA support for NVIDIA GPUs
- **CPU Optimized** - AVX, AVX2, AVX-512 auto-detection
- **Memory Efficient** - Sliding context window and adaptive batching

## System Requirements

| Platform | Requirements |
|----------|--------------|
| **Windows 10/11 x64** | Minimum 8GB RAM, 4GB free disk space |

### GPU Support (Optional)
- **NVIDIA GPU** - CUDA support for faster inference
- **CPU** - AVX/AVX2 optimizations for all modern CPUs

## Recommended Models

### Single Model
- Llama 3.2 3B Q4_K_M
- Qwen 2.5 7B Q4_K_M
- Mistral 7B Q4_K_M

### Duo Model Pairs (Speculative Decoding)
| Main Model | Draft Model | Speed Boost |
|------------|-------------|-------------|
| Llama 3.2 3B | Llama 3.2 1B | 2-3x |
| Qwen 2.5 7B | Qwen 2.5 1.5B | 2-4x |
| Mistral 7B | Mistral 0.5B | 2-3x |

## Installation

### Download Release
Download the latest release from the [Releases](../../releases) page.

### Build from Source

```powershell
# Clone the repository
git clone https://github.com/yourusername/nexastory-desktop.git
cd nexastory-desktop

# Install dependencies
bun install

# Build for Windows (CPU)
bun run tauri:build:native

# Build for Windows (CUDA - NVIDIA GPU)
bun run tauri:build:cuda
```

## Development

```powershell
# Start development server
bun run tauri:dev

# Run lint
bun run lint
cargo clippy --all-features
```

## Project Structure

```
nexastory-desktop/
├── src/                    # Next.js frontend
│   ├── components/         # React components
│   ├── lib/               # Utilities and API
│   └── app/               # Next.js app router
├── src-tauri/             # Rust backend
│   ├── src/
│   │   ├── llm.rs        # llama.cpp integration
│   │   ├── models.rs     # Data models
│   │   ├── database.rs   # SQLite operations
│   │   ├── enrichment.rs # Prompt enrichment
│   │   └── memory.rs     # Memory optimization
│   └── Cargo.toml
└── scripts/              # Build scripts
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF inference engine
- [Tauri](https://tauri.app) - Desktop application framework
- [shadcn/ui](https://ui.shadcn.com) - UI components
