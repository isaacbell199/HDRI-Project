# Changelog

All notable changes to NexaStory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2024-01-XX

### Added
- **Duo Model (Speculative Decoding)** - Use two models for faster generation
  - Draft model predicts tokens, main model verifies
  - Configurable draft tokens (1-8)
  - Configurable acceptance threshold (0.5-1.0)
  - Toggle on/off in Settings
- **Native llama.cpp Integration** - No external server required
  - Direct GGUF model loading
  - Streaming token generation
  - Full hardware acceleration support
  - **Updated to llama-cpp-2 v0.1.143** for latest llama.cpp features
- **CPU Optimizations**
  - Auto-detection of AVX, AVX2, AVX-512, FMA
  - Automatic thread optimization
  - Adaptive batch processing
- **GPU Support**
  - CUDA (NVIDIA) support for Windows
- **Memory Management**
  - Sliding context window
  - Context compression
  - Memory pressure monitoring
  - Adaptive batch sizing
- **GitHub Actions CI/CD**
  - Windows builds
  - Automated releases
  - CUDA build workflow

### Changed
- Upgraded to Tauri 2.0
- **Upgraded llama-cpp-2 from v0.1.140 to v0.1.143**
  - New sampler API: `LlamaSampler::chain_simple()` replaces builder pattern
  - `LlamaSampler::penalties()` now requires `penalty_last_n` parameter
  - Added `LlamaSampler::dist()` as final sampler for probabilistic token selection
- Improved prompt enrichment pipeline
- Better error handling and logging
- Optimized release builds (LTO, strip)
- **Windows ONLY** - Removed macOS and Linux support for cleaner codebase

### Fixed
- Model loading validation
- Memory leak prevention
- Thread safety improvements
- Sampler chain configuration for proper token selection

## [0.2.0] - 2024-01-XX

### Added
- Project management (create, edit, delete)
- Chapter management with word count
- Character management with detailed profiles
- Location management
- Lore notes system
- Generation presets
- Project settings (genre, style, tone, etc.)
- Hardware detection
- Model scanning and loading
- Text generation with streaming
- Enrichment pipeline for literary quality

### Technical
- SQLite database for local storage
- Tauri 2 + Next.js 16 architecture
- Zustand for state management
- shadcn/ui components

## [0.1.0] - Initial Release

### Added
- Basic project structure
- Tauri + Next.js setup
- Basic UI components
