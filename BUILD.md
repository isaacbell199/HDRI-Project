# NexaStory Build Workflow

## Version
- **Application**: 0.3.0
- **llama-cpp-2**: 0.1.140 (NEVER CHANGE)

## Build Requirements

### Windows (Target Platform)
- Windows 10/11 (64-bit)
- Node.js 18+ or Bun
- Rust 1.70+ (via rustup)
- Visual Studio Build Tools 2022
- Windows SDK

### Installation
```powershell
# Install Rust
winget install Rustlang.Rustup

# Install Bun
powershell -c "irm bun.sh/install.ps1 | iex"

# Install project dependencies
bun install
```

## Build Commands

### Development
```bash
# Start Next.js dev server
bun run dev

# Start Tauri in dev mode
bun run tauri:dev
```

### Production Build

#### CPU (Default - AVX/AVX2 Optimized)
```bash
# Full build with NSIS + MSI installers
bun run tauri:build
```

#### GPU Accelerated
```bash
# NVIDIA CUDA
bun run tauri:build:cuda

# Vulkan (AMD/Intel)
bun run tauri:build:vulkan
```

### Release Distribution
```bash
# Creates ZIP distribution with all files
bun run build:release
```

## Output Structure

```
dist/nexastory-0.3.0-windows-x64/
├── NexaStory.exe              # Standalone executable
├── NexaStory_0.3.0_x64.msi    # MSI installer
├── NexaStory_0.3.0_x64-setup.exe # NSIS installer
├── data/
│   ├── models/                # GGUF model files
│   ├── cache/                 # Generation cache
│   ├── logs/                  # Application logs
│   ├── errors/                # Error reports
│   ├── exports/               # Exported projects
│   ├── backups/               # Database backups
│   ├── settings/              # App settings
│   └── db/                    # SQLite database
├── README.md
└── VERSION
```

## Architecture

### Frontend (Next.js 16)
- React 19 with TypeScript
- Tailwind CSS 4 + shadcn/ui
- Zustand for state management
- Framer Motion for animations

### Backend (Tauri v2 + Rust)
- SQLite database via sqlx
- llama.cpp integration for GGUF inference
- CPU optimizations: AVX, AVX2, AVX-512, FMA
- Speculative decoding (Duo Model)

## Features

### ✅ 100% Desktop
- Native Windows application
- No browser required
- Offline-first design

### ✅ 100% Offline
- Local GGUF model inference
- No internet required after setup
- Privacy-focused

### ✅ 100% Local
- All data stored locally
- Portable data folder
- No cloud dependencies

### ✅ Windows Only
- Target: x86_64-pc-windows-msvc
- NSIS + MSI installers
- Native Windows integration

## Duo Model (Speculative Decoding)

Use a smaller "draft" model to accelerate generation:

```typescript
// Enable duo model mode
await setDuoModelEnabled(true)

// Load draft model
await loadDraftModel('path/to/draft-model.gguf')

// Main model verifies draft predictions
// ~2x speedup on compatible models
```

### Recommended Pairs
| Main Model | Draft Model | Speedup |
|------------|-------------|---------|
| Llama 3.2 3B | Llama 3.2 1B | ~2x |
| Qwen 2.5 7B | Qwen 2.5 1.5B | ~2.5x |
| Mistral 7B | Mistral 0.5B | ~1.8x |

## Troubleshooting

### Build Errors
```bash
# Clean Rust build
cd src-tauri && cargo clean

# Reinstall dependencies
rm -rf node_modules && bun install
```

### Model Loading Issues
- Ensure GGUF file is valid
- Check available RAM (model size × 1.5)
- Verify AVX support: Task Manager > Performance > CPU

### Database Issues
- Database stored in `data/nexastory.db`
- Backups in `data/backups/`
- Use restore function in Settings

## License
MIT License
