//! NexaStory - AI-powered creative writing assistant
//!
//! This is the main library for the Tauri backend, providing:
//! - SQLite database for projects, chapters, characters, etc.
//! - GGUF model loading and inference via llama.cpp
//! - Hardware detection and optimization
//! - Automatic backup system
//! - Cache system for generated content and queries

pub mod backup;
pub mod cache;
pub mod commands;
pub mod database;
pub mod enrichment;
pub mod llm;
pub mod memory;
pub mod models;
pub mod settings;

use std::sync::Arc;
use tauri::Manager;
use llm::LlmState;
use settings::AppState;

/// Get the directory where the executable is located
fn get_exe_directory() -> std::path::PathBuf {
    std::env::current_exe()
        .expect("Failed to get executable path")
        .parent()
        .expect("Failed to get parent directory")
        .to_path_buf()
}

/// Setup data directory structure in the same folder as the executable
fn setup_data_directories() -> std::path::PathBuf {
    let exe_dir = get_exe_directory();
    let data_dir = exe_dir.join("data");

    // Create main data directory
    std::fs::create_dir_all(&data_dir)
        .expect("Failed to create data directory");

    // Create subdirectories
    let subdirs = ["models", "cache", "logs", "errors", "exports", "backups", "settings"];
    for subdir in subdirs {
        std::fs::create_dir_all(data_dir.join(subdir))
            .expect(&format!("Failed to create {} directory", subdir));
    }

    data_dir
}

/// Run the Tauri application
pub fn run() {
    // Setup data directories first (before logger)
    let data_dir = setup_data_directories();
    let logs_dir = data_dir.join("logs");
    let errors_dir = data_dir.join("errors");

    // Set up panic hook to write crash logs
    let errors_dir_clone = errors_dir.clone();
    std::panic::set_hook(Box::new(move |panic_info| {
        let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
        let crash_file = errors_dir_clone.join(format!("crash_{}.log", timestamp));

        let message = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };

        let location = panic_info.location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "Unknown location".to_string());

        let content = format!(
            "=== NexaStory Crash Report ===\n\
            Time: {}\n\
            \n\
            Panic: {}\n\
            Location: {}\n\
            \n\
            Please report this issue at:\n\
            https://github.com/nexastory/nexastory/issues\n",
            chrono::Local::now().to_rfc3339(),
            message,
            location
        );

        // Write crash log
        let _ = std::fs::write(&crash_file, &content);

        // Also print to stderr
        eprintln!("{}", content);
        eprintln!("Crash log written to: {}", crash_file.display());
    }));

    // Initialize logger with dual output (stdout + file) (H6)
    let log_file_path = logs_dir.join("nexastory.log");

    // Create a simple file logger that also outputs to stdout
    // We use env_logger's Pipe target to write to both stdout and a log file
    use std::io::{self, Write, BufWriter};

    /// A writer that tees output to both stdout and a file
    struct TeeWriter {
        stdout: io::Stdout,
        file: BufWriter<std::fs::File>,
    }

    impl Write for TeeWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let _ = self.stdout.write(buf);
            self.file.write(buf)
        }
        fn flush(&mut self) -> io::Result<()> {
            self.stdout.flush()?;
            self.file.flush()
        }
    }

    // NEW-9 fix: Simple log rotation - if log file exceeds 10 MB, truncate it before opening
    if let Ok(metadata) = std::fs::metadata(&log_file_path) {
        if metadata.len() > 10 * 1024 * 1024 {
            if let Ok(content) = std::fs::read_to_string(&log_file_path) {
                let keep_from = content.len().saturating_sub(5 * 1024 * 1024);
                let truncated = &content[keep_from..];
                let _ = std::fs::write(&log_file_path, truncated);
            }
        }
    }

    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file_path)
        .expect("Failed to open log file");

    let tee = TeeWriter {
        stdout: io::stdout(),
        file: BufWriter::new(log_file),
    };

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .target(env_logger::Target::Pipe(Box::new(tee)))
        .format_timestamp_secs()
        .init();

    log::info!("=== NexaStory Starting ===");
    log::info!("Data directory: {}", data_dir.display());
    log::info!("Log file: {}", log_file_path.display());

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(move |app| {
            log::info!("Setting up application...");

            // Initialize database
            let _app_handle = app.handle();

            // Database path in data directory
            let db_path = data_dir.join("nexastory.db");
            let db_url = format!("sqlite:{}?mode=rwc", db_path.display());

            log::info!("Database path: {}", db_path.display());

            tauri::async_runtime::block_on(async {
                database::init_database(&db_url)
                    .await
                    .expect("Failed to initialize database");
            });

            log::info!("Database initialized");

            // Load saved settings
            let settings_dir = data_dir.join("settings");
            let app_settings = settings::load_app_settings_from(&settings_dir.join("app.json"));
            let llm_settings = settings::load_llm_settings_from(&settings_dir.join("llm.json"));

            // (M1) Use OnceLock instead of deprecated set_var
            crate::settings::set_data_dir(data_dir.clone());

            // Create app state with database URL
            let app_state = Arc::new(AppState::new(db_url.clone(), app_settings, llm_settings));

            // Create LLM state
            let llm_state = Arc::new(LlmState::new());

            // Store states in app
            app.manage(app_state);
            app.manage(llm_state);

            log::info!("=== NexaStory Ready ===");

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Project commands
            commands::get_projects,
            commands::get_project,
            commands::create_project,
            commands::update_project,
            commands::delete_project,
            // Chapter commands
            commands::get_chapters,
            commands::get_chapter,
            commands::create_chapter,
            commands::update_chapter,
            commands::delete_chapter,
            // Character commands
            commands::get_characters,
            commands::create_character,
            commands::update_character,
            commands::delete_character,
            // Location commands
            commands::get_locations,
            commands::create_location,
            commands::update_location,
            commands::delete_location,
            // Lore note commands
            commands::get_lore_notes,
            commands::create_lore_note,
            commands::update_lore_note,
            commands::delete_lore_note,
            // Project settings commands
            commands::get_project_settings,
            commands::update_project_settings,
            // Preset commands
            commands::get_presets,
            commands::create_preset,
            commands::delete_preset,
            // LLM commands
            commands::get_available_models,
            commands::load_model,
            commands::unload_model,
            commands::generate_text,
            commands::stop_generation,
            commands::get_llm_settings,
            commands::update_llm_settings,
            commands::get_hardware_info,
            commands::select_models_directory,
            commands::select_model_file,
            commands::scan_models_directory,
            commands::get_model_info,
            commands::delete_model,
            // App settings commands
            commands::get_app_settings,
            commands::update_app_settings,
            // Export/Import commands
            commands::export_project,
            commands::import_project,
            commands::export_all_projects,
            // Memory optimization commands
            commands::get_memory_info,
            commands::get_recommended_memory_settings,
            commands::get_batch_config,
            commands::update_batch_config,
            commands::clear_context_window,
            commands::get_context_window_stats,
            // Duo Model commands (Speculative Decoding)
            commands::load_draft_model,
            commands::unload_draft_model,
            commands::set_duo_model_enabled,
            commands::get_duo_model_status,
            // CPU Optimization commands
            commands::get_cpu_optimizations,
            // Backup commands
            commands::create_backup,
            commands::list_backups,
            commands::restore_backup,
            commands::delete_backup,
            commands::cleanup_backups,
            commands::save_export_to_file,
            commands::get_backups_directory,
            commands::get_exports_directory,
            // Cache commands
            commands::cache_store,
            commands::cache_get,
            commands::cache_exists,
            commands::cache_remove,
            commands::cache_clear_type,
            commands::cache_clear_all,
            commands::cache_cleanup_expired,
            commands::cache_get_stats,
            commands::cache_list,
            commands::cache_generation,
            commands::find_cached_generation,
            commands::cache_db_query,
            commands::find_cached_db_query,
            commands::cache_embedding,
            commands::find_cached_embedding,
            commands::get_cache_directory,
            commands::get_cache_size,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
