//! Settings management for NexaStory
//! All settings are stored in the data folder next to the executable

use parking_lot::RwLock;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::models::{AppSettings, LLMSettings};

/// Global application state
pub struct AppState {
    pub db_url: String,
    pub app_settings: RwLock<AppSettings>,
    pub llm_settings: RwLock<LLMSettings>,
    pub models_directory: RwLock<Option<String>>,
}

impl AppState {
    pub fn new(db_url: String, app_settings: AppSettings, llm_settings: LLMSettings) -> Self {
        Self {
            db_url,
            app_settings: RwLock::new(app_settings),
            llm_settings: RwLock::new(llm_settings),
            models_directory: RwLock::new(None),
        }
    }
}

/// Global data directory path, set once during startup (M1 fix)
static DATA_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Initialize the global data directory path. Must be called once during startup. (M1)
pub fn set_data_dir(path: PathBuf) {
    if DATA_DIR.set(path).is_err() {
        log::warn!("Data directory already initialized, ignoring duplicate set");
    }
}

/// Get the data directory path (next to executable)
pub fn get_data_dir() -> PathBuf {
    if let Some(dir) = DATA_DIR.get() {
        return dir.clone();
    }
    // Fallback: use executable directory (should not happen in production)
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("data")
}

/// Get the settings directory path
fn get_settings_dir() -> PathBuf {
    get_data_dir().join("settings")
}

/// Load app settings from a specific path
pub fn load_app_settings_from(path: &Path) -> AppSettings {
    if path.exists() {
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(settings) = serde_json::from_str(&content) {
                return settings;
            }
        }
    }

    AppSettings::default()
}

/// Load LLM settings from a specific path
pub fn load_llm_settings_from(path: &Path) -> LLMSettings {
    if path.exists() {
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(settings) = serde_json::from_str(&content) {
                return settings;
            }
        }
    }

    LLMSettings::default()
}

/// Load app settings from disk (data folder)
pub fn load_app_settings() -> AppSettings {
    let settings_dir = get_settings_dir();
    let settings_path = settings_dir.join("app.json");

    // Also try old location for migration
    let old_path = get_data_dir().join("app_settings.json");

    if settings_path.exists() {
        load_app_settings_from(&settings_path)
    } else if old_path.exists() {
        // Migrate old settings to new location
        if let Ok(content) = std::fs::read_to_string(&old_path) {
            let _ = std::fs::create_dir_all(&settings_dir);
            let _ = std::fs::write(&settings_path, &content);
        }
        load_app_settings_from(&old_path)
    } else {
        AppSettings::default()
    }
}

/// Save app settings to disk (data folder)
pub fn save_app_settings(settings: &AppSettings) -> anyhow::Result<()> {
    // Validate settings
    if settings.font_size.is_empty() {
        return Err(anyhow::anyhow!("Font size cannot be empty"));
    }
    if !["small", "medium", "large"].contains(&settings.font_size.as_str()) {
        return Err(anyhow::anyhow!("Invalid font size: {}. Must be 'small', 'medium', or 'large'", settings.font_size));
    }
    if settings.language.is_empty() {
        return Err(anyhow::anyhow!("Language cannot be empty"));
    }
    
    let settings_dir = get_settings_dir();
    std::fs::create_dir_all(&settings_dir)?;

    let settings_path = settings_dir.join("app.json");
    let content = serde_json::to_string_pretty(settings)?;
    std::fs::write(&settings_path, content)?;

    log::info!("App settings saved to: {}", settings_path.display());
    Ok(())
}

/// Load LLM settings from disk (data folder)
pub fn load_llm_settings() -> LLMSettings {
    let settings_dir = get_settings_dir();
    let settings_path = settings_dir.join("llm.json");

    // Also try old location for migration
    let old_path = get_data_dir().join("llm_settings.json");

    if settings_path.exists() {
        load_llm_settings_from(&settings_path)
    } else if old_path.exists() {
        // Migrate old settings to new location
        if let Ok(content) = std::fs::read_to_string(&old_path) {
            let _ = std::fs::create_dir_all(&settings_dir);
            let _ = std::fs::write(&settings_path, &content);
        }
        load_llm_settings_from(&old_path)
    } else {
        LLMSettings::default()
    }
}

/// Save LLM settings to disk (data folder)
pub fn save_llm_settings(settings: &LLMSettings) -> anyhow::Result<()> {
    // Validate settings
    if settings.temperature <= 0.0 {
        return Err(anyhow::anyhow!("Temperature must be greater than 0, got {}", settings.temperature));
    }
    if settings.temperature > 2.0 {
        log::warn!("Temperature {} is very high, may produce incoherent output", settings.temperature);
    }
    if settings.max_tokens < 1 {
        return Err(anyhow::anyhow!("Max tokens must be at least 1, got {}", settings.max_tokens));
    }
    if settings.context_length < 128 {
        return Err(anyhow::anyhow!("Context length must be at least 128, got {}", settings.context_length));
    }
    if settings.top_p < 0.0 || settings.top_p > 1.0 {
        return Err(anyhow::anyhow!("Top P must be between 0 and 1, got {}", settings.top_p));
    }
    if settings.min_p < 0.0 || settings.min_p > 1.0 {
        return Err(anyhow::anyhow!("Min P must be between 0 and 1, got {}", settings.min_p));
    }
    if settings.cpu_threads < 1 {
        return Err(anyhow::anyhow!("CPU threads must be at least 1, got {}", settings.cpu_threads));
    }
    if settings.gpu_layers < 0 {
        return Err(anyhow::anyhow!("GPU layers cannot be negative, got {}", settings.gpu_layers));
    }
    
    let settings_dir = get_settings_dir();
    std::fs::create_dir_all(&settings_dir)?;

    let settings_path = settings_dir.join("llm.json");
    let content = serde_json::to_string_pretty(settings)?;
    std::fs::write(&settings_path, content)?;

    log::info!("LLM settings saved to: {}", settings_path.display());
    Ok(())
}

/// Get the models directory path
pub fn get_models_dir() -> PathBuf {
    get_data_dir().join("models")
}

/// Get the cache directory path
pub fn get_cache_dir() -> PathBuf {
    get_data_dir().join("cache")
}

/// Get the logs directory path
pub fn get_logs_dir() -> PathBuf {
    get_data_dir().join("logs")
}

/// Get the errors directory path
pub fn get_errors_dir() -> PathBuf {
    get_data_dir().join("errors")
}

/// Get the exports directory path
pub fn get_exports_dir() -> PathBuf {
    get_data_dir().join("exports")
}

/// Get the backups directory path
pub fn get_backups_dir() -> PathBuf {
    get_data_dir().join("backups")
}

/// Write an error report to the errors directory
pub fn write_error_report(error_type: &str, error: &anyhow::Error) -> PathBuf {
    let errors_dir = get_errors_dir();
    let _ = std::fs::create_dir_all(&errors_dir);

    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
    let filename = format!("{}_{}.log", error_type, timestamp);
    let error_path = errors_dir.join(&filename);

    let content = format!(
        "=== NexaStory Error Report ===\n\
         Time: {}\n\
         Type: {}\n\
         \n\
         Error: {}\n\
         \n\
         Backtrace:\n{:?}\n",
        chrono::Local::now().to_rfc3339(),
        error_type,
        error,
        error.backtrace()
    );

    let _ = std::fs::write(&error_path, content);
    log::error!("Error report written to: {}", error_path.display());

    error_path
}
