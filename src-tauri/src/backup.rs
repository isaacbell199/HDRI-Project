//! Backup management for NexaStory
//! Automatic backups are stored in data/backups/

use anyhow::Result;
use chrono::Local;
use std::fs;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Backup metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupInfo {
    pub id: String,
    pub filename: String,
    pub created_at: String,
    pub size_bytes: u64,
    pub project_count: usize,
    pub chapter_count: usize,
    pub is_auto: bool,
}

/// Get counts from the database (helper function)
async fn get_database_counts() -> Result<(usize, usize)> {
    // Use the global database pool instead of creating a separate one (M2)
    let pool = crate::database::get_pool()?;
    
    let project_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM projects")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);
    
    let chapter_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chapters")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);
    
    Ok((project_count as usize, chapter_count as usize))
}

/// Get the backups directory path (M1 fix: uses OnceLock via settings instead of env var)
pub fn get_backups_dir() -> PathBuf {
    crate::settings::get_backups_dir()
}

/// Ensure backups directory exists
pub fn ensure_backups_dir() -> Result<PathBuf> {
    let dir = get_backups_dir();
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Create a full database backup
pub async fn create_backup(db_url: &str, is_auto: bool) -> Result<BackupInfo> {
    let backups_dir = ensure_backups_dir()?;

    // Generate backup filename
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
    let prefix = if is_auto { "auto" } else { "manual" };
    let filename = format!("{}_backup_{}.db", prefix, timestamp);
    let backup_path = backups_dir.join(&filename);

    // Copy the database file
    // Parse the SQLite URL to get the file path
    let db_path = db_url
        .strip_prefix("sqlite:")
        .map(|s| s.split('?').next().unwrap_or(s))
        .ok_or_else(|| anyhow::anyhow!("Invalid database URL"))?;

    // NEW-10 fix: Use VACUUM INTO for consistent backup instead of fs::copy
    // fs::copy on an active SQLite database can produce a corrupt backup on Windows
    // VACUUM INTO creates a consistent snapshot without requiring exclusive access
    let pool = crate::database::get_pool()?;
    let backup_path_str = backup_path.to_string_lossy().to_string();
    let vacuum_result = sqlx::query(&format!("VACUUM INTO '{}'", backup_path_str.replace('\'', "''")))
        .execute(&pool)
        .await;
    
    match vacuum_result {
        Ok(_) => {
            log::info!("Created consistent backup via VACUUM INTO: {}", filename);
        }
        Err(e) => {
            log::warn!("VACUUM INTO failed (falling back to file copy): {}", e);
            // Fallback to file copy if VACUUM INTO is not supported
            fs::copy(db_path, &backup_path)?;
        }
    }

    // Get file size
    let metadata = fs::metadata(&backup_path)?;
    let size_bytes = metadata.len();

    // Query the database for actual project and chapter counts
    let (project_count, chapter_count) = get_database_counts().await
        .unwrap_or((0, 0));

    let backup_info = BackupInfo {
        id: timestamp.to_string(),
        filename,
        created_at: Local::now().to_rfc3339(),
        size_bytes,
        project_count,
        chapter_count,
        is_auto,
    };

    log::info!(
        "Created {} backup: {} ({} bytes, {} projects, {} chapters)",
        if is_auto { "automatic" } else { "manual" },
        backup_info.filename,
        size_bytes,
        project_count,
        chapter_count
    );

    Ok(backup_info)
}

/// List all available backups
pub fn list_backups() -> Result<Vec<BackupInfo>> {
    let backups_dir = get_backups_dir();

    if !backups_dir.exists() {
        return Ok(vec![]);
    }

    let mut backups = Vec::new();

    for entry in fs::read_dir(&backups_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|e| e == "db").unwrap_or(false) {
            let filename = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            let metadata = fs::metadata(&path)?;
            let created_at = metadata.modified()
                .map(|t| {
                    let datetime: chrono::DateTime<Local> = t.into();
                    datetime.to_rfc3339()
                })
                .unwrap_or_else(|_| "Unknown".to_string());

            let is_auto = filename.starts_with("auto_");

            backups.push(BackupInfo {
                id: filename.clone(),
                filename,
                created_at,
                size_bytes: metadata.len(),
                project_count: 0,
                chapter_count: 0,
                is_auto,
            });
        }
    }

    // Sort by filename (which contains sortable timestamp: YYYY-MM-DD_HH-MM-SS) (L3)
    // This is more reliable than sorting by the string-based created_at field
    backups.sort_by(|a, b| b.filename.cmp(&a.filename));

    Ok(backups)
}

/// Restore database from a backup
pub async fn restore_backup(db_url: &str, backup_filename: &str) -> Result<()> {
    let backups_dir = get_backups_dir();
    let backup_path = backups_dir.join(backup_filename);

    if !backup_path.exists() {
        return Err(anyhow::anyhow!("Backup file not found: {}", backup_filename));
    }

    // NEW-2 fix: Validate path is within backups directory (prevent path traversal)
    let canonical_backup = backup_path.canonicalize()
        .map_err(|_| anyhow::anyhow!("Invalid backup path"))?;
    let canonical_backups_dir = backups_dir.canonicalize()
        .unwrap_or_else(|_| backups_dir.clone());
    if !canonical_backup.starts_with(&canonical_backups_dir) {
        return Err(anyhow::anyhow!("Backup file must be within the backups directory"));
    }

    // Validate backup file extension
    if backup_path.extension().map(|e| e != "db").unwrap_or(true) {
        return Err(anyhow::anyhow!("Invalid backup file: must be a .db file"));
    }

    // Parse the SQLite URL to get the file path
    let db_path = db_url
        .strip_prefix("sqlite:")
        .map(|s| s.split('?').next().unwrap_or(s))
        .ok_or_else(|| anyhow::anyhow!("Invalid database URL"))?;

    // Create a backup of current database before restore
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
    let pre_restore_backup = format!("pre_restore_{}.db", timestamp);
    let pre_restore_path = backups_dir.join(&pre_restore_backup);

    if fs::metadata(db_path).is_ok() {
        fs::copy(db_path, &pre_restore_path)?;
        log::info!("Created pre-restore backup: {}", pre_restore_backup);
    }

    // CRITICAL: Close all database connections before replacing the file (C5)
    // This prevents corruption from overwriting an active SQLite database
    crate::database::close_pool();

    // Copy the backup to the database location
    fs::copy(&backup_path, db_path)?;

    log::info!("Restored database from backup: {}", backup_filename);
    // NEW-27 fix: Re-initialize the database pool after restoring
    // This allows the app to continue working without requiring a restart
    if let Err(e) = crate::database::init_database(db_url).await {
        log::warn!("Failed to re-initialize database after restore (app may need restart): {}", e);
    } else {
        log::info!("Database re-initialized after restore");
    }

    Ok(())
}

/// Delete a backup file
pub fn delete_backup(backup_filename: &str) -> Result<()> {
    let backups_dir = get_backups_dir();
    let backup_path = backups_dir.join(backup_filename);

    // NEW-4 fix: Validate path is within backups directory (prevent path traversal)
    if backup_path.exists() {
        let canonical_backup = backup_path.canonicalize()
            .map_err(|_| anyhow::anyhow!("Invalid backup path"))?;
        let canonical_backups_dir = backups_dir.canonicalize()
            .unwrap_or_else(|_| backups_dir.clone());
        if !canonical_backup.starts_with(&canonical_backups_dir) {
            return Err(anyhow::anyhow!("Backup file must be within the backups directory"));
        }
        // Validate extension
        if backup_path.extension().map(|e| e != "db").unwrap_or(true) {
            return Err(anyhow::anyhow!("Only .db backup files can be deleted"));
        }
        fs::remove_file(&backup_path)?;
        log::info!("Deleted backup: {}", backup_filename);
    }

    Ok(())
}

/// Clean up old automatic backups, keeping only the most recent N
pub fn cleanup_old_backups(keep_count: usize) -> Result<usize> {
    let backups = list_backups()?;
    let auto_backups: Vec<_> = backups.iter()
        .filter(|b| b.is_auto)
        .collect();

    let mut deleted = 0;

    // Delete old auto backups
    if auto_backups.len() > keep_count {
        for backup in auto_backups.iter().skip(keep_count) {
            if delete_backup(&backup.filename).is_ok() {
                deleted += 1;
            }
        }
    }

    Ok(deleted)
}

/// Get the exports directory path (M1 fix: uses OnceLock via settings instead of env var)
pub fn get_exports_dir() -> PathBuf {
    crate::settings::get_exports_dir()
}

/// Ensure exports directory exists
pub fn ensure_exports_dir() -> Result<PathBuf> {
    let dir = get_exports_dir();
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Save export to file
pub fn save_export(filename: &str, content: &str) -> Result<String> {
    let exports_dir = ensure_exports_dir()?;
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
    
    // NEW-3 fix: Sanitize filename to prevent path traversal
    // Extract only the file name component (strips any directory separators)
    let safe_filename = std::path::Path::new(filename)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "export".to_string());
    // Remove any remaining dangerous characters
    let safe_filename: String = safe_filename
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    
    let final_filename = if safe_filename.is_empty() {
        format!("export_{}.json", timestamp)
    } else {
        format!("{}_{}.json", safe_filename, timestamp)
    };

    let export_path = exports_dir.join(&final_filename);
    
    // Verify the path is within exports directory
    let canonical_export = export_path.canonicalize()
        .unwrap_or_else(|_| export_path.clone());
    let canonical_exports = exports_dir.canonicalize()
        .unwrap_or_else(|_| exports_dir.clone());
    if !canonical_export.starts_with(&canonical_exports) {
        return Err(anyhow::anyhow!("Export path must be within the exports directory"));
    }
    fs::write(&export_path, content)?;

    log::info!("Saved export to: {}", export_path.display());

    Ok(export_path.to_string_lossy().to_string())
}
