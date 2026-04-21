//! Cache system for NexaStory
//!
//! Provides caching capabilities for:
//! - Generated content (LLM responses)
//! - Database query results
//! - Session state

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Constants
// ============================================================================

/// Maximum cache size in bytes (500 MB)
pub const MAX_CACHE_SIZE_BYTES: u64 = 500 * 1024 * 1024;

/// Maximum entries per cache type
pub const MAX_ENTRIES_PER_TYPE: usize = 1000;

// ============================================================================
// Cache Types
// ============================================================================

/// Types of cache entries
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CacheType {
    /// LLM generated content
    Generation,
    /// Database query results
    DbQuery,
    /// Embeddings or vectors
    Embedding,
    /// Application session data
    Session,
}

/// A single cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Unique identifier for this entry
    pub id: String,
    /// Type of cache
    pub cache_type: CacheType,
    /// The cached content (JSON string)
    pub content: String,
    /// Hash of the input that produced this result
    pub input_hash: String,
    /// Creation timestamp (Unix epoch seconds)
    pub created_at: u64,
    /// Last access timestamp
    pub last_accessed: u64,
    /// Access count
    pub access_count: u64,
    /// Size in bytes
    pub size_bytes: u64,
    /// Time-to-live in seconds (0 = no expiry)
    pub ttl_seconds: u64,
    /// Associated project ID (if any)
    pub project_id: Option<String>,
    /// Metadata tags
    pub tags: Vec<String>,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total entries count
    pub total_entries: u64,
    /// Total size in bytes
    pub total_size_bytes: u64,
    /// Entries by type
    pub entries_by_type: HashMap<String, u64>,
    /// Size by type in bytes
    pub size_by_type: HashMap<String, u64>,
    /// Hit count
    pub hit_count: u64,
    /// Miss count
    pub miss_count: u64,
    /// Cache directory path
    pub cache_directory: String,
}

/// Information about a single cache entry for listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntryInfo {
    pub id: String,
    pub cache_type: String,
    pub created_at: String,
    pub last_accessed: String,
    pub access_count: u64,
    pub size_bytes: u64,
    pub project_id: Option<String>,
    pub tags: Vec<String>,
}

// ============================================================================
// Cache Directory Management
// ============================================================================

/// Get the cache directory path (M1 fix: uses OnceLock via settings instead of env var)
fn get_cache_directory() -> PathBuf {
    crate::settings::get_cache_dir()
}

/// Ensure cache subdirectory exists
fn ensure_cache_subdir(cache_type: &CacheType) -> PathBuf {
    let cache_dir = get_cache_directory();
    let subdir = match cache_type {
        CacheType::Generation => "generations",
        CacheType::DbQuery => "db_queries",
        CacheType::Embedding => "embeddings",
        CacheType::Session => "sessions",
    };
    let path = cache_dir.join(subdir);
    if let Err(e) = fs::create_dir_all(&path) {
        log::error!("Failed to create cache directory {:?}: {}", path, e);
    }
    path
}

/// Get current Unix timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Hash function for cache keys using double FNV-1a for reduced collision risk (H5)
fn simple_hash(input: &str) -> String {
    // Primary FNV-1a hash
    let mut hash1: u64 = 0xcbf29ce484222325;
    for byte in input.bytes() {
        hash1 ^= byte as u64;
        hash1 = hash1.wrapping_mul(0x100000001b3);
    }
    // Secondary FNV-1a hash with different seed
    let mut hash2: u64 = 0x9dc5e8a1b7d3c4f6;
    for byte in input.bytes() {
        hash2 ^= byte as u64;
        hash2 = hash2.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}{:016x}", hash1, hash2)
}

// ============================================================================
// Cache Statistics Management
// ============================================================================

/// Get the stats file path
fn get_stats_file_path() -> PathBuf {
    get_cache_directory().join("stats.json")
}

/// Load stats from file and return (hit_count, miss_count)
fn load_stats_from_file() -> (u64, u64) {
    let stats_file = get_stats_file_path();
    if stats_file.exists() {
        if let Ok(json) = fs::read_to_string(&stats_file) {
            if let Ok(saved) = serde_json::from_str::<HashMap<String, u64>>(&json) {
                let hits = saved.get("hit_count").copied().unwrap_or(0);
                let misses = saved.get("miss_count").copied().unwrap_or(0);
                return (hits, misses);
            }
        }
    }
    (0, 0)
}

/// Initialize atomic counters from saved stats (call once at startup)
pub fn init_stats() {
    let (hits, misses) = load_stats_from_file();
    CACHE_HIT_COUNT.store(hits, Ordering::Relaxed);
    CACHE_MISS_COUNT.store(misses, Ordering::Relaxed);
    LAST_STATS_PERSIST.store(current_timestamp(), Ordering::Relaxed);
    log::info!("Cache stats initialized: {} hits, {} misses", hits, misses);
}

/// Save stats to file
fn save_stats(hit_count: u64, miss_count: u64) {
    let stats_file = get_stats_file_path();
    let stats = serde_json::json!({
        "hit_count": hit_count,
        "miss_count": miss_count,
    });

    // Ensure parent directory exists
    if let Some(parent) = stats_file.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            log::warn!("Failed to create cache stats directory: {}", e); // (L2)
        }
    }

    if let Ok(json) = serde_json::to_string_pretty(&stats) {
        if let Err(e) = fs::write(&stats_file, json) {
            log::warn!("Failed to write cache stats to {:?}: {}", stats_file, e); // (L2)
        }
    }
}

/// In-memory atomic counters for cache hits/misses (H4, N1)
/// These are much faster than file I/O on every cache access.
/// Stats are persisted to disk periodically and on demand.
static CACHE_HIT_COUNT: AtomicU64 = AtomicU64::new(0);
static CACHE_MISS_COUNT: AtomicU64 = AtomicU64::new(0);

/// Timestamp of last stats persistence (N1 debounce)
static LAST_STATS_PERSIST: AtomicU64 = AtomicU64::new(0);

/// Minimum interval between stats persist operations (60 seconds)
const STATS_PERSIST_INTERVAL_SECS: u64 = 60;

/// Increment hit count atomically (H4, N1)
fn increment_hit_count() {
    CACHE_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
    log::debug!("Cache hit count incremented to {}", CACHE_HIT_COUNT.load(Ordering::Relaxed));
    maybe_persist_stats();
}

/// Increment miss count atomically (H4, N1)
fn increment_miss_count() {
    CACHE_MISS_COUNT.fetch_add(1, Ordering::Relaxed);
    log::debug!("Cache miss count incremented to {}", CACHE_MISS_COUNT.load(Ordering::Relaxed));
    maybe_persist_stats();
}

/// Persist stats to disk if enough time has elapsed since last persist (N1 debounce)
fn maybe_persist_stats() {
    let now = current_timestamp();
    let last = LAST_STATS_PERSIST.load(Ordering::Relaxed);
    if now.saturating_sub(last) >= STATS_PERSIST_INTERVAL_SECS {
        // Try to claim the persist operation (compare_exchange to avoid thundering herd)
        if LAST_STATS_PERSIST.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            let hit_count = CACHE_HIT_COUNT.load(Ordering::Relaxed);
            let miss_count = CACHE_MISS_COUNT.load(Ordering::Relaxed);
            save_stats(hit_count, miss_count);
        }
    }
}

/// Force-persist stats to disk (called on shutdown or explicit stats request)
fn flush_stats() {
    let hit_count = CACHE_HIT_COUNT.load(Ordering::Relaxed);
    let miss_count = CACHE_MISS_COUNT.load(Ordering::Relaxed);
    save_stats(hit_count, miss_count);
    LAST_STATS_PERSIST.store(current_timestamp(), Ordering::Relaxed);
}

// ============================================================================
// Cache Operations
// ============================================================================

/// Store an entry in the cache
pub fn cache_store(
    cache_type: CacheType,
    id: &str,
    content: &str,
    input_hash: &str,
    ttl_seconds: u64,
    project_id: Option<String>,
    tags: Vec<String>,
) -> Result<CacheEntry> {
    let dir = ensure_cache_subdir(&cache_type);
    let now = current_timestamp();

    let entry = CacheEntry {
        id: id.to_string(),
        cache_type: cache_type.clone(),
        content: content.to_string(),
        input_hash: input_hash.to_string(),
        created_at: now,
        last_accessed: now,
        access_count: 1,
        size_bytes: content.len() as u64,
        ttl_seconds,
        project_id,
        tags,
    };

    // Save to file
    let filename = format!("{}.json", simple_hash(id));
    let filepath = dir.join(&filename);

    // (H5) Hash collision detection: if file exists, verify it's for the same key
    if filepath.exists() {
        if let Ok(existing_json) = fs::read_to_string(&filepath) {
            if let Ok(existing_entry) = serde_json::from_str::<CacheEntry>(&existing_json) {
                if existing_entry.id != id {
                    log::warn!(
                        "Cache hash collision detected! Key '{}' collides with existing key '{}'. \
                         Using suffixed filename to preserve both entries.",
                        id, existing_entry.id
                    );
                    // Use a suffixed filename to avoid overwriting
                    let collision_hash = simple_hash(&format!("{}:collision:{}", id, current_timestamp()));
                    let collision_filename = format!("{}.json", collision_hash);
                    let collision_filepath = dir.join(&collision_filename);
                    let json = serde_json::to_string_pretty(&entry)?;
                    fs::write(&collision_filepath, json)?;
                    log::info!("Cache stored with collision-safe key: {} ({:?})", id, cache_type);
                    return Ok(entry);
                }
                // Same key - overwrite is intentional (update)
            }
        }
    }

    let json = serde_json::to_string_pretty(&entry)?;
    fs::write(&filepath, json)?;

    log::info!("Cache stored: {} ({:?})", id, cache_type);
    Ok(entry)
}

/// Retrieve an entry from the cache
pub fn cache_get(cache_type: CacheType, id: &str) -> Result<Option<CacheEntry>> {
    let dir = ensure_cache_subdir(&cache_type);
    let filename = format!("{}.json", simple_hash(id));
    let filepath = dir.join(&filename);

    if !filepath.exists() {
        // Cache miss - increment miss count
        increment_miss_count();
        return Ok(None);
    }

    let json = fs::read_to_string(&filepath)?;
    let mut entry: CacheEntry = serde_json::from_str(&json)?;

    // Check TTL
    if entry.ttl_seconds > 0 {
        let now = current_timestamp();
        if now > entry.created_at + entry.ttl_seconds {
            // Expired, remove it
            fs::remove_file(&filepath).ok();
            log::info!("Cache expired and removed: {}", id);
            // Cache miss due to expiry
            increment_miss_count();
            return Ok(None);
        }
    }

    // Cache hit - increment hit count
    increment_hit_count();

    // Update access stats
    entry.last_accessed = current_timestamp();
    entry.access_count += 1;

    // Save updated entry
    let updated_json = serde_json::to_string(&entry)?;
    fs::write(&filepath, updated_json)?;

    Ok(Some(entry))
}

/// Check if entry exists and is valid
pub fn cache_exists(cache_type: CacheType, id: &str) -> bool {
    let dir = ensure_cache_subdir(&cache_type);
    let filename = format!("{}.json", simple_hash(id));
    let filepath = dir.join(&filename);

    if !filepath.exists() {
        return false;
    }

    // Verify it's not expired
    if let Ok(json) = fs::read_to_string(&filepath) {
        if let Ok(entry) = serde_json::from_str::<CacheEntry>(&json) {
            if entry.ttl_seconds > 0 {
                let now = current_timestamp();
                return now <= entry.created_at + entry.ttl_seconds;
            }
            return true;
        }
    }

    false
}

/// Remove an entry from the cache
pub fn cache_remove(cache_type: CacheType, id: &str) -> Result<bool> {
    let dir = ensure_cache_subdir(&cache_type);
    let filename = format!("{}.json", simple_hash(id));
    let filepath = dir.join(&filename);

    if filepath.exists() {
        fs::remove_file(&filepath)?;
        log::info!("Cache removed: {}", id);
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Clear all entries of a specific type
pub fn cache_clear_type(cache_type: CacheType) -> Result<u64> {
    let dir = ensure_cache_subdir(&cache_type);
    let mut count = 0;

    if dir.exists() {
        for entry in fs::read_dir(&dir)? {
            if let Ok(entry) = entry {
                if entry.path().extension().map(|e| e == "json").unwrap_or(false) {
                    fs::remove_file(entry.path())?;
                    count += 1;
                }
            }
        }
    }

    log::info!("Cache cleared for type {:?}: {} entries", cache_type, count);
    Ok(count)
}

/// Clear all cache entries
pub fn cache_clear_all() -> Result<u64> {
    let mut total = 0;
    total += cache_clear_type(CacheType::Generation)?;
    total += cache_clear_type(CacheType::DbQuery)?;
    total += cache_clear_type(CacheType::Embedding)?;
    total += cache_clear_type(CacheType::Session)?;
    Ok(total)
}

/// Clean up expired entries across all cache types
pub fn cache_cleanup_expired() -> Result<u64> {
    let mut count = 0;
    let now = current_timestamp();

    for cache_type in [CacheType::Generation, CacheType::DbQuery, CacheType::Embedding, CacheType::Session] {
        let dir = ensure_cache_subdir(&cache_type);

        if dir.exists() {
            for entry in fs::read_dir(&dir)? {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.extension().map(|e| e == "json").unwrap_or(false) {
                        if let Ok(json) = fs::read_to_string(&path) {
                            if let Ok(cache_entry) = serde_json::from_str::<CacheEntry>(&json) {
                                if cache_entry.ttl_seconds > 0 && now > cache_entry.created_at + cache_entry.ttl_seconds {
                                    fs::remove_file(&path)?;
                                    count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    log::info!("Cache cleanup: {} expired entries removed", count);
    Ok(count)
}

/// Get cache statistics
pub fn cache_get_stats() -> Result<CacheStats> {
    let mut stats = CacheStats {
        total_entries: 0,
        total_size_bytes: 0,
        entries_by_type: HashMap::new(),
        size_by_type: HashMap::new(),
        hit_count: 0,
        miss_count: 0,
        cache_directory: get_cache_directory().to_string_lossy().to_string(),
    };

    for cache_type in [CacheType::Generation, CacheType::DbQuery, CacheType::Embedding, CacheType::Session] {
        let dir = ensure_cache_subdir(&cache_type);
        let type_name = format!("{:?}", cache_type).to_lowercase();
        let mut type_count = 0u64;
        let mut type_size = 0u64;

        if dir.exists() {
            for entry in fs::read_dir(&dir)? {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.extension().map(|e| e == "json").unwrap_or(false) {
                        if let Ok(metadata) = entry.metadata() {
                            type_count += 1;
                            type_size += metadata.len();
                        }
                    }
                }
            }
        }

        stats.total_entries += type_count;
        stats.total_size_bytes += type_size;
        stats.entries_by_type.insert(type_name.clone(), type_count);
        stats.size_by_type.insert(type_name, type_size);
    }

    // Load hit/miss stats from atomic counters (H4, N1)
    flush_stats();
    stats.hit_count = CACHE_HIT_COUNT.load(Ordering::Relaxed);
    stats.miss_count = CACHE_MISS_COUNT.load(Ordering::Relaxed);

    Ok(stats)
}

/// List entries of a specific type
pub fn cache_list(cache_type: CacheType) -> Result<Vec<CacheEntryInfo>> {
    let dir = ensure_cache_subdir(&cache_type);
    let mut entries = Vec::new();

    if dir.exists() {
        for entry in fs::read_dir(&dir)? {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.extension().map(|e| e == "json").unwrap_or(false) {
                    if let Ok(json) = fs::read_to_string(&path) {
                        if let Ok(cache_entry) = serde_json::from_str::<CacheEntry>(&json) {
                            entries.push(CacheEntryInfo {
                                id: cache_entry.id,
                                cache_type: format!("{:?}", cache_entry.cache_type).to_lowercase(),
                                created_at: chrono::DateTime::from_timestamp(cache_entry.created_at as i64, 0)
                                    .map(|d| d.format("%Y-%m-%d %H:%M:%S").to_string())
                                    .unwrap_or_default(),
                                last_accessed: chrono::DateTime::from_timestamp(cache_entry.last_accessed as i64, 0)
                                    .map(|d| d.format("%Y-%m-%d %H:%M:%S").to_string())
                                    .unwrap_or_default(),
                                access_count: cache_entry.access_count,
                                size_bytes: cache_entry.size_bytes,
                                project_id: cache_entry.project_id,
                                tags: cache_entry.tags,
                            });
                        }
                    }
                }
            }
        }
    }

    // Sort by last accessed (most recent first)
    entries.sort_by(|a, b| b.last_accessed.cmp(&a.last_accessed));

    Ok(entries)
}

/// Store a generated content (convenience function)
pub fn cache_generation(
    prompt_hash: &str,
    generated_text: &str,
    project_id: Option<String>,
    model_name: Option<String>,
) -> Result<CacheEntry> {
    let mut tags = vec!["llm".to_string()];
    if let Some(model) = model_name {
        tags.push(model);
    }

    cache_store(
        CacheType::Generation,
        &format!("gen_{}", prompt_hash),
        generated_text,
        prompt_hash,
        86400 * 7, // 7 days TTL for generations
        project_id,
        tags,
    )
}

/// Find cached generation by input hash
pub fn find_cached_generation(input_hash: &str) -> Result<Option<CacheEntry>> {
    let id = format!("gen_{}", input_hash);
    cache_get(CacheType::Generation, &id)
}

/// Store a DB query result
pub fn cache_db_query(
    query_hash: &str,
    result_json: &str,
    ttl_seconds: u64,
) -> Result<CacheEntry> {
    cache_store(
        CacheType::DbQuery,
        &format!("dbq_{}", query_hash),
        result_json,
        query_hash,
        ttl_seconds,
        None,
        vec!["db_query".to_string()],
    )
}

/// Find cached DB query
pub fn find_cached_db_query(query_hash: &str) -> Result<Option<CacheEntry>> {
    let id = format!("dbq_{}", query_hash);
    cache_get(CacheType::DbQuery, &id)
}

/// Store an embedding
pub fn cache_embedding(
    text_hash: &str,
    embedding_json: &str,
    model_name: &str,
) -> Result<CacheEntry> {
    cache_store(
        CacheType::Embedding,
        &format!("emb_{}", text_hash),
        embedding_json,
        text_hash,
        86400 * 30, // 30 days TTL for embeddings
        None,
        vec!["embedding".to_string(), model_name.to_string()],
    )
}

/// Find cached embedding
pub fn find_cached_embedding(text_hash: &str) -> Result<Option<CacheEntry>> {
    let id = format!("emb_{}", text_hash);
    cache_get(CacheType::Embedding, &id)
}

/// Get the cache directory path (for frontend)
pub fn get_cache_directory_path() -> String {
    get_cache_directory().to_string_lossy().to_string()
}

/// Get cache size summary
pub fn get_cache_size() -> Result<(u64, u64)> {
    let stats = cache_get_stats()?;
    Ok((stats.total_entries, stats.total_size_bytes))
}

/// Enforce cache size limits by removing oldest entries
pub fn enforce_cache_limits() -> Result<u64> {
    let stats = cache_get_stats()?;
    let mut removed = 0u64;
    
    // Check total size
    if stats.total_size_bytes > MAX_CACHE_SIZE_BYTES {
        log::warn!(
            "Cache size ({:.2} MB) exceeds limit ({:.2} MB), cleaning up...",
            stats.total_size_bytes as f64 / (1024.0 * 1024.0),
            MAX_CACHE_SIZE_BYTES as f64 / (1024.0 * 1024.0)
        );
        
        // Remove expired entries first
        removed += cache_cleanup_expired()?;
        
        // If still over limit, remove oldest entries
        let new_stats = cache_get_stats()?;
        if new_stats.total_size_bytes > MAX_CACHE_SIZE_BYTES {
            removed += remove_oldest_entries_until_under_limit()?;
        }
    }
    
    // Check entries per type
    for (cache_type_name, count) in &stats.entries_by_type {
        if *count > MAX_ENTRIES_PER_TYPE as u64 {
            let ct = match cache_type_name.as_str() {
                "generation" => CacheType::Generation,
                "dbquery" => CacheType::DbQuery,
                "embedding" => CacheType::Embedding,
                "session" => CacheType::Session,
                _ => continue,
            };
            let to_remove = count - MAX_ENTRIES_PER_TYPE as u64;
            log::info!("Cache type {} has {} entries (max {}), removing oldest {}", 
                cache_type_name, count, MAX_ENTRIES_PER_TYPE, to_remove);
            removed += remove_oldest_entries(ct, to_remove as usize)?;
        }
    }
    
    if removed > 0 {
        log::info!("Cache limit enforcement removed {} entries", removed);
    }
    
    Ok(removed)
}

/// Remove oldest entries from a specific cache type
fn remove_oldest_entries(cache_type: CacheType, count: usize) -> Result<u64> {
    let dir = ensure_cache_subdir(&cache_type);
    let mut entries: Vec<(PathBuf, u64)> = Vec::new();
    
    if !dir.exists() {
        return Ok(0);
    }
    
    // Collect all entries with their modification times
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map(|e| e == "json").unwrap_or(false) {
            if let Ok(metadata) = entry.metadata() {
                if let Ok(modified) = metadata.modified() {
                    if let Ok(duration) = modified.duration_since(UNIX_EPOCH) {
                        entries.push((path, duration.as_secs()));
                    }
                }
            }
        }
    }
    
    // Sort by modification time (oldest first)
    entries.sort_by_key(|(_, time)| *time);
    
    // Remove oldest entries
    let mut removed = 0u64;
    for (path, _) in entries.into_iter().take(count) {
        if fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    
    Ok(removed)
}

/// Remove oldest entries across all cache types until under size limit
fn remove_oldest_entries_until_under_limit() -> Result<u64> {
    let mut removed = 0u64;
    let mut all_entries: Vec<(PathBuf, u64)> = Vec::new();
    
    // Collect all entries with their modification times
    for cache_type in [CacheType::Generation, CacheType::DbQuery, CacheType::Embedding, CacheType::Session] {
        let dir = ensure_cache_subdir(&cache_type);
        if dir.exists() {
            for entry in fs::read_dir(&dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map(|e| e == "json").unwrap_or(false) {
                    if let Ok(metadata) = entry.metadata() {
                        if let Ok(modified) = metadata.modified() {
                            if let Ok(duration) = modified.duration_since(UNIX_EPOCH) {
                                all_entries.push((path, duration.as_secs()));
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Sort by modification time (oldest first)
    all_entries.sort_by_key(|(_, time)| *time);
    
    // Remove oldest entries until under limit
    let stats = cache_get_stats()?;
    let mut current_size = stats.total_size_bytes;
    
    for (path, _) in all_entries {
        if current_size <= MAX_CACHE_SIZE_BYTES {
            break;
        }
        if let Ok(metadata) = fs::metadata(&path) {
            let size = metadata.len();
            if fs::remove_file(&path).is_ok() {
                current_size -= size;
                removed += 1;
            }
        }
    }
    
    Ok(removed)
}
