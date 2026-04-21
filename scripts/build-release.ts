#!/usr/bin/env bun

/**
 * NexaStory Release Build Script
 * Creates a portable ZIP distribution for Windows
 * 
 * Usage: bun run build:release
 * 
 * This script:
 * 1. Builds the frontend (Next.js)
 * 2. Builds Tauri app with MSI and NSIS installers
 * 3. Creates a distribution folder with all files
 * 4. Packages everything into a ZIP file
 */

import { spawn, execSync } from 'child_process'
import { join, basename } from 'path'
import { existsSync, mkdirSync, cpSync, writeFileSync, rmSync, readdirSync } from 'fs'

const VERSION = '0.3.0'
const DIST_DIR = 'dist'
const RELEASE_DIR = join(DIST_DIR, `nexastory-${VERSION}-windows-x64`)

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  red: '\x1b[31m',
  cyan: '\x1b[36m'
}

function log(message: string, color: string = colors.reset) {
  console.log(`${color}${message}${colors.reset}`)
}

function logStep(step: string) {
  console.log('\n' + colors.cyan + '▶ ' + step + colors.reset)
}

function logSuccess(message: string) {
  console.log(colors.green + '  ✓ ' + message + colors.reset)
}

function logError(message: string) {
  console.log(colors.red + '  ✗ ' + message + colors.reset)
}

async function runCommand(command: string, args: string[] = [], cwd?: string): Promise<boolean> {
  return new Promise((resolve) => {
    const proc = spawn(command, args, {
      cwd,
      stdio: 'inherit',
      shell: true
    })

    proc.on('close', (code) => {
      resolve(code === 0)
    })

    proc.on('error', (err) => {
      logError(`Command failed: ${err.message}`)
      resolve(false)
    })
  })
}

async function main() {
  console.log(colors.bright + '\n' + '='.repeat(60) + colors.reset)
  console.log(colors.bright + `  NexaStory v${VERSION} - Windows Build Script` + colors.reset)
  console.log(colors.bright + '='.repeat(60) + colors.reset + '\n')

  // Step 1: Build frontend
  logStep('Building frontend (Next.js)...')
  const frontendSuccess = await runCommand('bun', ['run', 'build'])
  if (!frontendSuccess) {
    logError('Frontend build failed!')
    process.exit(1)
  }
  logSuccess('Frontend built successfully')

  // Step 2: Build Tauri application
  logStep('Building Tauri application (EXE + MSI + NSIS)...')
  const tauriSuccess = await runCommand('bun', [
    'tauri', 'build',
    '--target', 'x86_64-pc-windows-msvc',
    '--bundles', 'nsis,msi'
  ])
  if (!tauriSuccess) {
    logError('Tauri build failed!')
    process.exit(1)
  }
  logSuccess('Tauri application built successfully')

  // Step 3: Create distribution directory
  logStep('Creating distribution directory...')
  
  // Clean up existing directory
  if (existsSync(RELEASE_DIR)) {
    rmSync(RELEASE_DIR, { recursive: true, force: true })
  }
  mkdirSync(RELEASE_DIR, { recursive: true })
  logSuccess('Distribution directory created')

  // Step 4: Copy executables
  logStep('Copying executables...')
  const targetDir = 'src-tauri/target/x86_64-pc-windows-msvc/release'

  // Copy main EXE
  const exePath = join(targetDir, 'NexaStory.exe')
  if (existsSync(exePath)) {
    cpSync(exePath, join(RELEASE_DIR, 'NexaStory.exe'))
    logSuccess('NexaStory.exe')
  } else {
    logError('NexaStory.exe not found!')
  }

  // Copy NSIS installer
  const nsisDir = join(targetDir, 'bundle/nsis')
  if (existsSync(nsisDir)) {
    const nsisFiles = readdirSync(nsisDir).filter(f => f.endsWith('.exe'))
    for (const file of nsisFiles) {
      cpSync(join(nsisDir, file), join(RELEASE_DIR, file))
      logSuccess(file)
    }
  }

  // Copy MSI installer
  const msiDir = join(targetDir, 'bundle/msi')
  if (existsSync(msiDir)) {
    const msiFiles = readdirSync(msiDir).filter(f => f.endsWith('.msi'))
    for (const file of msiFiles) {
      cpSync(join(msiDir, file), join(RELEASE_DIR, file))
      logSuccess(file)
    }
  }

  // Step 5: Create data directory structure
  logStep('Creating data directory structure...')
  const dataDirs = [
    'data/models',
    'data/cache',
    'data/logs',
    'data/errors',
    'data/exports',
    'data/backups',
    'data/settings',
    'data/db'
  ]

  for (const dir of dataDirs) {
    const fullPath = join(RELEASE_DIR, dir)
    mkdirSync(fullPath, { recursive: true })
    // Create .gitkeep file
    writeFileSync(join(fullPath, '.gitkeep'), '# This directory will be populated automatically by NexaStory\n')
  }
  logSuccess('Data directories created')

  // Step 6: Create README
  logStep('Creating README...')
  const readmeContent = `# NexaStory v${VERSION}

AI-powered creative writing assistant with local GGUF model support.

## Quick Start

1. Run \`NexaStory.exe\` to start the application
2. Place your GGUF model files in the \`data/models/\` folder
3. Go to **Models > Configuration** to load your model
4. Start writing with AI assistance!

## Features

- **Local AI Models**: Run GGUF models locally for privacy
- **Duo Model (Speculative Decoding)**: Use a smaller draft model for faster generation
- **CPU Optimizations**: AVX, AVX2, AVX-512, FMA support
- **Memory Optimization**: Sliding context window, batch processing
- **Portable**: All data stored in the \`data/\` folder

## System Requirements

- Windows 10/11 (64-bit)
- 8GB RAM minimum (16GB recommended)
- CPU with AVX support (AVX2 recommended for best performance)

## Duo Model Setup

Speculative decoding uses two models:
1. **Target Model** (large, accurate): e.g., Llama 3.2 3B
2. **Draft Model** (small, fast): e.g., Llama 3.2 1B

Recommended pairs:
- Llama 3.2 3B + 1B (~2x speedup)
- Qwen 2.5 7B + 1.5B (~2.5x speedup)
- Mistral 7B + 0.5B (~1.8x speedup)

## Folder Structure

\`\`\`
nexastory/
├── NexaStory.exe           # Main application
├── NexaStory_0.3.0_x64.msi # MSI installer
├── NexaStory_0.3.0_x64-setup.exe # NSIS installer
├── data/
│   ├── models/             # Place your GGUF files here
│   ├── cache/              # Model cache
│   ├── logs/               # Application logs
│   ├── errors/             # Error reports
│   ├── exports/            # Exported projects
│   ├── backups/            # Project backups
│   ├── settings/           # App settings
│   └── db/                 # Database
└── README.md               # This file
\`\`\`

## Installation Options

### Option 1: Portable (Recommended)
Simply run \`NexaStory.exe\` directly. No installation required!

### Option 2: MSI Installer
Run \`NexaStory_0.3.0_x64.msi\` to install system-wide.

### Option 3: NSIS Installer
Run \`NexaStory_0.3.0_x64-setup.exe\` for a traditional Windows installer.

## GPU Support

The default build uses CPU with AVX optimizations. For GPU acceleration:

- **NVIDIA**: Build with \`bun run tauri:build:cuda\`
- **AMD/Intel**: Not currently supported (Vulkan feature removed — CUDA only)

## License

MIT License - See LICENSE file for details.

---
Built with Tauri, Next.js, and llama.cpp
`

  writeFileSync(join(RELEASE_DIR, 'README.md'), readmeContent)
  logSuccess('README.md created')

  // Step 7: Create VERSION file
  const versionContent = `NexaStory v${VERSION}
Build Date: ${new Date().toISOString()}
Platform: Windows x64
`
  writeFileSync(join(RELEASE_DIR, 'VERSION'), versionContent)
  logSuccess('VERSION file created')

  // Step 8: Create ZIP archive
  logStep('Creating ZIP archive...')
  const zipPath = join(DIST_DIR, `nexastory-${VERSION}-windows-x64.zip`)

  if (existsSync(zipPath)) {
    rmSync(zipPath, { force: true })
  }

  try {
    // Use PowerShell to create ZIP
    execSync(`powershell -Command "Compress-Archive -Path '${RELEASE_DIR}' -DestinationPath '${zipPath}'"`, {
      stdio: 'inherit'
    })
    logSuccess('ZIP archive created')
  } catch {
    logError('Failed to create ZIP archive')
    log('Please manually create ZIP from: ' + RELEASE_DIR, colors.yellow)
  }

  // Final summary
  console.log('\n' + colors.bright + '='.repeat(60) + colors.reset)
  console.log(colors.green + '  ✅ BUILD COMPLETE!' + colors.reset)
  console.log(colors.bright + '='.repeat(60) + colors.reset + '\n')

  console.log(colors.cyan + 'Distribution folder:' + colors.reset + ' ' + RELEASE_DIR)
  console.log(colors.cyan + 'ZIP archive:' + colors.reset + ' ' + zipPath)
  console.log('\n' + colors.bright + 'Contents:' + colors.reset)
  console.log('  • NexaStory.exe (standalone executable)')
  console.log('  • MSI installer')
  console.log('  • NSIS installer')
  console.log('  • data/ folder structure')
  console.log('  • README.md')
  console.log('  • VERSION')

  console.log('\n' + colors.green + '🎉 Ready for distribution!' + colors.reset + '\n')
}

main().catch((err) => {
  logError('Build script failed: ' + err.message)
  process.exit(1)
})
