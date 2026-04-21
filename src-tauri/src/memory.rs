//! Memory and CPU Optimization Module for LLM Generation
//!
//! This module provides:
//! - Token estimation and counting
//! - Sliding context window management
//! - Batch processing with CPU throttling
//! - Memory pressure monitoring
//! - Context trimming strategies

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ============================================================================
// Constants for Memory Optimization
// ============================================================================

/// Average characters per token (approximation for most models)
pub const CHARS_PER_TOKEN: f32 = 4.0;

/// Default maximum context tokens for sliding window
pub const DEFAULT_MAX_CONTEXT_TOKENS: usize = 2048;

/// Minimum context tokens to always keep (system prompt + recent context)
pub const MIN_CONTEXT_TOKENS: usize = 256;

/// Default batch size for generation (tokens per batch)
pub const DEFAULT_BATCH_SIZE_TOKENS: usize = 50;

/// Default delay between batches (milliseconds)
pub const DEFAULT_BATCH_DELAY_MS: u64 = 50;

/// Maximum tokens before triggering context compression
pub const COMPRESSION_THRESHOLD_TOKENS: usize = 1500;

/// Memory warning threshold (percentage of available RAM)
pub const MEMORY_WARNING_THRESHOLD: f64 = 80.0;

// ============================================================================
// Token Estimation
// ============================================================================

/// Estimate token count from text
/// Uses a simple heuristic: tokens ≈ characters / 4
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }

    // Simple estimation: ~4 chars per token average
    let char_count = text.chars().count();
    let base_estimate = (char_count as f32 / CHARS_PER_TOKEN) as usize;

    // Add overhead for special tokens, whitespace, etc.
    let overhead = text.lines().count() / 10; // ~10% overhead for structure

    base_estimate + overhead + 1
}

/// Estimate tokens for a prompt with context
pub fn estimate_prompt_tokens(
    system_prompt: &str,
    user_prompt: &str,
    context: Option<&str>,
) -> usize {
    let mut total = estimate_tokens(system_prompt);
    total += estimate_tokens(user_prompt);

    if let Some(ctx) = context {
        total += estimate_tokens(ctx);
    }

    // Add special tokens overhead
    total += 10;

    total
}

// ============================================================================
// Sliding Window Context Management
// ============================================================================

/// Configuration for sliding window
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Maximum tokens to keep in context
    pub max_tokens: usize,
    /// Minimum tokens to always preserve from start (system prompt)
    pub preserve_start_tokens: usize,
    /// Minimum tokens to always preserve from end (recent context)
    pub preserve_end_tokens: usize,
    /// Whether to use smart truncation (sentence boundaries)
    pub smart_truncation: bool,
}

impl Default for SlidingWindowConfig {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_CONTEXT_TOKENS,
            preserve_start_tokens: 100,
            preserve_end_tokens: 500,
            smart_truncation: true,
        }
    }
}

/// Sliding window context manager
pub struct SlidingContextWindow {
    /// Configuration
    config: SlidingWindowConfig,
    /// Current context chunks with their token estimates
    context_chunks: VecDeque<ContextChunk>,
    /// Total estimated tokens in window
    total_tokens: usize,
}

/// A chunk of context with metadata
#[derive(Debug, Clone)]
pub struct ContextChunk {
    /// The text content
    pub text: String,
    /// Estimated token count
    pub token_count: usize,
    /// Priority (higher = less likely to be trimmed)
    pub priority: ChunkPriority,
    /// Source identifier
    pub source: String,
}

/// Priority levels for context chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChunkPriority {
    /// Can be trimmed first
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// Should be preserved if possible
    High = 2,
    /// Must always be preserved (system prompt)
    Critical = 3,
}

impl SlidingContextWindow {
    /// Create a new sliding window with configuration
    pub fn new(config: SlidingWindowConfig) -> Self {
        Self {
            config,
            context_chunks: VecDeque::new(),
            total_tokens: 0,
        }
    }

    /// Add a chunk to the context window
    pub fn add_chunk(&mut self, chunk: ContextChunk) {
        self.total_tokens += chunk.token_count;
        self.context_chunks.push_back(chunk);

        // Trim if over limit
        self.trim_if_needed();
    }

    /// Add text with automatic token estimation
    pub fn add_text(&mut self, text: &str, priority: ChunkPriority, source: &str) {
        let chunk = ContextChunk {
            text: text.to_string(),
            token_count: estimate_tokens(text),
            priority,
            source: source.to_string(),
        };
        self.add_chunk(chunk);
    }

    /// Get the current context as a single string
    pub fn get_context(&self) -> String {
        self.context_chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Get current token count
    pub fn token_count(&self) -> usize {
        self.total_tokens
    }

    /// Check if window is over capacity
    pub fn is_over_capacity(&self) -> bool {
        self.total_tokens > self.config.max_tokens
    }

    /// Trim context to fit within limits
    fn trim_if_needed(&mut self) {
        while self.total_tokens > self.config.max_tokens && self.context_chunks.len() > 1 {
            // Find the lowest priority chunk that's not critical
            let mut lowest_priority_idx = None;
            let mut lowest_priority = ChunkPriority::Critical;

            for (i, chunk) in self.context_chunks.iter().enumerate() {
                // Skip first and last chunks (preserve_start and preserve_end)
                if i == 0 || i >= self.context_chunks.len().saturating_sub(1) {
                    continue;
                }

                if chunk.priority < lowest_priority {
                    lowest_priority = chunk.priority;
                    lowest_priority_idx = Some(i);
                }
            }

            // Remove the lowest priority chunk
            if let Some(idx) = lowest_priority_idx {
                let removed = self.context_chunks.remove(idx).unwrap();
                self.total_tokens = self.total_tokens.saturating_sub(removed.token_count);
                log::debug!(
                    "Trimmed context chunk: {} ({} tokens, priority {:?})",
                    removed.source,
                    removed.token_count,
                    removed.priority
                );
            } else {
                // All chunks are critical or we can't trim more
                break;
            }
        }
    }

    /// Clear all context
    pub fn clear(&mut self) {
        self.context_chunks.clear();
        self.total_tokens = 0;
    }

    /// Get the number of chunks
    pub fn chunk_count(&self) -> usize {
        self.context_chunks.len()
    }
}

// ============================================================================
// Batch Processing Configuration
// ============================================================================

/// Configuration for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Number of tokens to generate per batch
    pub tokens_per_batch: usize,
    /// Delay between batches in milliseconds
    pub batch_delay_ms: u64,
    /// Whether to adapt batch size based on system load
    pub adaptive_batch_size: bool,
    /// Minimum batch delay (for adaptive mode)
    pub min_batch_delay_ms: u64,
    /// Maximum batch delay (for adaptive mode)
    pub max_batch_delay_ms: u64,
    /// Target CPU usage percentage (for adaptive mode)
    pub target_cpu_percent: f32,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            tokens_per_batch: DEFAULT_BATCH_SIZE_TOKENS,
            batch_delay_ms: DEFAULT_BATCH_DELAY_MS,
            adaptive_batch_size: true,
            min_batch_delay_ms: 20,
            max_batch_delay_ms: 200,
            target_cpu_percent: 70.0,
        }
    }
}

impl BatchConfig {
    /// Create a fast config (lower latency, higher CPU)
    pub fn fast() -> Self {
        Self {
            tokens_per_batch: 100,
            batch_delay_ms: 20,
            adaptive_batch_size: false,
            min_batch_delay_ms: 10,
            max_batch_delay_ms: 50,
            target_cpu_percent: 85.0,
        }
    }

    /// Create a balanced config
    pub fn balanced() -> Self {
        Self {
            tokens_per_batch: 50,
            batch_delay_ms: 50,
            adaptive_batch_size: true,
            min_batch_delay_ms: 30,
            max_batch_delay_ms: 100,
            target_cpu_percent: 70.0,
        }
    }

    /// Create a conservative config (lower CPU, higher latency)
    pub fn conservative() -> Self {
        Self {
            tokens_per_batch: 30,
            batch_delay_ms: 100,
            adaptive_batch_size: true,
            min_batch_delay_ms: 50,
            max_batch_delay_ms: 200,
            target_cpu_percent: 50.0,
        }
    }

    /// Create config for low-memory systems
    pub fn low_memory() -> Self {
        Self {
            tokens_per_batch: 25,
            batch_delay_ms: 75,
            adaptive_batch_size: true,
            min_batch_delay_ms: 50,
            max_batch_delay_ms: 150,
            target_cpu_percent: 60.0,
        }
    }
}

// ============================================================================
// Memory Pressure Monitoring
// ============================================================================

/// Memory usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryInfo {
    /// Total system memory in GB
    pub total_gb: f64,
    /// Available memory in GB
    pub available_gb: f64,
    /// Used memory in GB
    pub used_gb: f64,
    /// Memory usage percentage
    #[serde(rename = "usedPercent")]
    pub usage_percent: f64,
    /// Whether memory is under pressure
    pub is_under_pressure: bool,
    /// Recommended context tokens based on available memory
    pub recommended_context_tokens: usize,
    /// Recommended batch size based on available memory
    pub recommended_batch_size: usize,
}

impl MemoryInfo {
    /// Get current memory information
    pub fn current() -> Self {
        let sys = sysinfo::System::new_all();

        let total = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let available = sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let used = total - available;
        let usage_percent = if total > 0.0 { (used / total) * 100.0 } else { 0.0 };
        let is_under_pressure = usage_percent > MEMORY_WARNING_THRESHOLD;

        // Calculate recommendations
        let recommended_context_tokens = if available < 2.0 {
            1024
        } else if available < 4.0 {
            2048
        } else if available < 8.0 {
            4096
        } else if available < 16.0 {
            8192
        } else {
            16384
        };

        let recommended_batch_size = if available < 2.0 {
            25
        } else if available < 4.0 {
            50
        } else if available < 8.0 {
            100
        } else {
            200
        };

        Self {
            total_gb: total,
            available_gb: available,
            used_gb: used,
            usage_percent,
            is_under_pressure,
            recommended_context_tokens,
            recommended_batch_size,
        }
    }

    /// Get recommended settings based on available memory
    pub fn recommended_batch_config(&self) -> BatchConfig {
        if self.available_gb < 2.0 {
            BatchConfig::low_memory()
        } else if self.available_gb < 4.0 {
            BatchConfig::conservative()
        } else if self.available_gb < 8.0 {
            BatchConfig::balanced()
        } else {
            BatchConfig::fast()
        }
    }

    /// Get recommended context window size
    pub fn recommended_context_tokens(&self) -> usize {
        self.recommended_context_tokens
    }
}

// ============================================================================
// Context Compression
// ============================================================================

/// Strategies for compressing context
#[derive(Debug, Clone, Copy)]
pub enum CompressionStrategy {
    /// Remove oldest content first
    TrimOldest,
    /// Remove based on priority
    TrimLowPriority,
    /// Summarize content (requires LLM)
    Summarize,
    /// Keep only essential context
    KeepEssential,
}

/// Compress context using the specified strategy
pub fn compress_context(
    context: &str,
    max_tokens: usize,
    strategy: CompressionStrategy,
) -> String {
    let current_tokens = estimate_tokens(context);

    if current_tokens <= max_tokens {
        return context.to_string();
    }

    match strategy {
        CompressionStrategy::TrimOldest => {
            // Split into paragraphs and keep the most recent
            let paragraphs: Vec<&str> = context.split("\n\n").collect();
            let mut result = String::new();
            let mut token_count = 0;

            for para in paragraphs.iter().rev() {
                let para_tokens = estimate_tokens(para);
                if token_count + para_tokens <= max_tokens {
                    result.insert_str(0, para);
                    result.insert_str(0, "\n\n");
                    token_count += para_tokens;
                } else {
                    break;
                }
            }

            result.trim().to_string()
        }

        CompressionStrategy::KeepEssential => {
            // Keep only the first and last portions
            let chars: Vec<char> = context.chars().collect();
            let target_chars = (max_tokens as f32 * CHARS_PER_TOKEN) as usize;

            if chars.len() <= target_chars {
                return context.to_string();
            }

            let first_part = target_chars / 3;
            let last_part = target_chars - first_part;

            let mut result = String::new();
            result.extend(chars.iter().take(first_part));
            result.push_str("\n\n[...content truncated...]\n\n");
            result.extend(chars.iter().skip(chars.len() - last_part));

            result
        }

        CompressionStrategy::TrimLowPriority => {
            // For now, same as TrimOldest
            // In a full implementation, this would use NLP to identify important sections
            compress_context(context, max_tokens, CompressionStrategy::TrimOldest)
        }

        CompressionStrategy::Summarize => {
            // Summarization would require an LLM call
            // For now, fall back to keeping essential
            compress_context(context, max_tokens, CompressionStrategy::KeepEssential)
        }
    }
}

// ============================================================================
// Generation Optimization
// ============================================================================

/// Optimize a prompt for memory efficiency
pub fn optimize_prompt(
    system_prompt: &str,
    user_prompt: &str,
    context: Option<&str>,
    max_tokens: usize,
) -> String {
    let mut total = String::new();

    // System prompt is critical - always include
    total.push_str(system_prompt);
    total.push_str("\n\n");

    // Calculate remaining budget
    let system_tokens = estimate_tokens(system_prompt);
    let user_tokens = estimate_tokens(user_prompt);
    let remaining = max_tokens.saturating_sub(system_tokens + user_tokens + 50); // 50 for overhead

    // Add context if we have room
    if let Some(ctx) = context {
        let context_tokens = estimate_tokens(ctx);

        if context_tokens <= remaining {
            total.push_str(ctx);
            total.push_str("\n\n");
        } else {
            // Compress context to fit
            let compressed = compress_context(
                ctx,
                remaining,
                CompressionStrategy::KeepEssential,
            );
            total.push_str(&compressed);
            total.push_str("\n\n");
        }
    }

    // User prompt is also critical
    total.push_str(user_prompt);

    total
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        // Empty string
        assert_eq!(estimate_tokens(""), 0);

        // Simple text
        let text = "Hello, this is a test.";
        let tokens = estimate_tokens(text);
        assert!(tokens > 0);
        assert!(tokens < 20); // Should be roughly 5-6 tokens

        // Longer text
        let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(10);
        let long_tokens = estimate_tokens(&long_text);
        assert!(long_tokens > 50);
    }

    #[test]
    fn test_sliding_window() {
        let config = SlidingWindowConfig {
            max_tokens: 100,
            ..Default::default()
        };
        let mut window = SlidingContextWindow::new(config);

        // Add some chunks
        window.add_text("This is the first chunk.", ChunkPriority::High, "intro");
        window.add_text("This is the second chunk with more content.", ChunkPriority::Normal, "body");
        window.add_text("This is the third chunk.", ChunkPriority::Low, "extra");

        // Window should have trimmed to fit
        assert!(window.token_count() <= 100);
    }

    #[test]
    fn test_compress_context() {
        let context = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n\nFourth paragraph.";
        let compressed = compress_context(context, 20, CompressionStrategy::KeepEssential);

        // Should be smaller than original
        assert!(estimate_tokens(&compressed) <= 25);
    }

    #[test]
    fn test_batch_config_presets() {
        let fast = BatchConfig::fast();
        assert!(fast.batch_delay_ms < 50);

        let conservative = BatchConfig::conservative();
        assert!(conservative.batch_delay_ms > 50);

        let low_mem = BatchConfig::low_memory();
        assert!(low_mem.tokens_per_batch <= 30);
    }

    #[test]
    fn test_memory_info() {
        let info = MemoryInfo::current();

        // Should have valid memory info
        assert!(info.total_gb > 0.0);
        assert!(info.available_gb > 0.0);
        assert!(info.usage_percent >= 0.0 && info.usage_percent <= 100.0);

        // Should give a valid recommendation
        let config = info.recommended_batch_config();
        assert!(config.batch_delay_ms >= config.min_batch_delay_ms);
    }
}
