# NexaStory Windows Build Script
# PowerShell script for building NexaStory on Windows

param(
    [Parameter(Position=0)]
    [ValidateSet("install", "build", "dev", "cuda", "avx2", "avx512")]
    [string]$Command = "build"
)

Write-Host "🚀 NexaStory Build Script (Windows)" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

function Install-Dependencies {
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    
    # Check for bun
    $bunExists = Get-Command bun -ErrorAction SilentlyContinue
    if (-not $bunExists) {
        Write-Host "Installing bun..." -ForegroundColor Yellow
        irm bun.sh/install.ps1 | iex
    }
    
    # Install npm dependencies
    bun install
    
    Write-Host "✅ Dependencies installed!" -ForegroundColor Green
}

function Build-App {
    param([string]$Features = "llama-native")
    
    Write-Host "🔨 Building with features: $Features" -ForegroundColor Yellow
    
    $env:RUSTFLAGS = "-C target-cpu=native"
    
    if ($Features -eq "avx2") {
        $env:RUSTFLAGS = "-C target-cpu=native -C target-feature=+avx2,+fma"
        $Features = "llama-native"
    }
    elseif ($Features -eq "avx512") {
        $env:RUSTFLAGS = "-C target-cpu=native -C target-feature=+avx512f,+avx512dq,+avx512vl"
        $Features = "llama-native"
    }
    
    bun run tauri build --release --features $Features
    
    Write-Host "✅ Build complete!" -ForegroundColor Green
}

function Start-Dev {
    Write-Host "🚀 Starting development server..." -ForegroundColor Yellow
    bun run tauri dev
}

# Main
switch ($Command) {
    "install" {
        Install-Dependencies
    }
    "build" {
        Build-App "llama-native"
    }
    "dev" {
        Start-Dev
    }
    "cuda" {
        Build-App "llama-cuda"
    }
    "avx2" {
        Build-App "avx2"
    }
    "avx512" {
        Build-App "avx512"
    }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Usage: .\build.ps1 {install|build|dev|cuda|avx2|avx512}"
    }
}
