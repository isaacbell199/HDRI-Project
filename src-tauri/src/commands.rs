//! Tauri commands for frontend-backend communication

use std::sync::Arc;
use tauri::{AppHandle, State};

use crate::database;
use crate::llm::{self, LlmState};
use crate::memory::MemoryInfo;
use crate::models::*;
use crate::settings::{self, AppState};

// ============================================================================
// Project Commands
// ============================================================================

#[tauri::command]
pub async fn get_projects() -> Result<Vec<ProjectWithCounts>, String> {
    database::get_projects().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_project(id: String) -> Result<Option<Project>, String> {
    database::get_project(&id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn create_project(data: CreateProjectRequest) -> Result<Project, String> {
    // Validate input
    if data.name.trim().is_empty() {
        return Err("Project name cannot be empty".to_string());
    }
    if data.name.len() > 255 {
        return Err("Project name too long (max 255 characters)".to_string());
    }
    
    database::create_project(data).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn update_project(id: String, data: Project) -> Result<Project, String> {
    // Validate input
    if data.name.trim().is_empty() {
        return Err("Project name cannot be empty".to_string());
    }
    if data.name.len() > 255 {
        return Err("Project name too long (max 255 characters)".to_string());
    }
    if id.trim().is_empty() {
        return Err("Project ID cannot be empty".to_string());
    }
    
    database::update_project(&id, data).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_project(id: String) -> Result<(), String> {
    database::delete_project(&id).await.map_err(|e| e.to_string())
}

// ============================================================================
// Chapter Commands
// ============================================================================

#[tauri::command]
pub async fn get_chapters(project_id: String) -> Result<Vec<Chapter>, String> {
    database::get_chapters(&project_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_chapter(id: String) -> Result<Option<Chapter>, String> {
    database::get_chapter(&id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn create_chapter(data: CreateChapterRequest) -> Result<Chapter, String> {
    // Validate input
    if data.title.trim().is_empty() {
        return Err("Chapter title cannot be empty".to_string());
    }
    if data.title.len() > 255 {
        return Err("Chapter title too long (max 255 characters)".to_string());
    }
    if data.project_id.trim().is_empty() {
        return Err("Project ID cannot be empty".to_string());
    }
    
    database::create_chapter(data).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn update_chapter(id: String, data: Chapter) -> Result<Chapter, String> {
    // Validate input
    if data.title.trim().is_empty() {
        return Err("Chapter title cannot be empty".to_string());
    }
    if id.trim().is_empty() {
        return Err("Chapter ID cannot be empty".to_string());
    }
    
    database::update_chapter(&id, data).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_chapter(id: String) -> Result<(), String> {
    database::delete_chapter(&id).await.map_err(|e| e.to_string())
}

// ============================================================================
// Character Commands
// ============================================================================

#[tauri::command]
pub async fn get_characters(project_id: String) -> Result<Vec<Character>, String> {
    database::get_characters(&project_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn create_character(data: CreateCharacterRequest) -> Result<Character, String> {
    // NEW-8 fix: Validate required fields
    if data.name.trim().is_empty() {
        return Err("Character name cannot be empty".to_string());
    }
    if data.name.len() > 255 {
        return Err("Character name too long (max 255 characters)".to_string());
    }
    database::create_character(data).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn update_character(id: String, data: Character) -> Result<Character, String> {
    database::update_character(&id, data).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_character(id: String) -> Result<(), String> {
    database::delete_character(&id).await.map_err(|e| e.to_string())
}

// ============================================================================
// Location Commands
// ============================================================================

#[tauri::command]
pub async fn get_locations(project_id: String) -> Result<Vec<Location>, String> {
    database::get_locations(&project_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn create_location(data: CreateLocationRequest) -> Result<Location, String> {
    // NEW-8 fix: Validate required fields
    if data.name.trim().is_empty() {
        return Err("Location name cannot be empty".to_string());
    }
    if data.name.len() > 255 {
        return Err("Location name too long (max 255 characters)".to_string());
    }
    database::create_location(data).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn update_location(id: String, data: Location) -> Result<Location, String> {
    database::update_location(&id, data).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_location(id: String) -> Result<(), String> {
    database::delete_location(&id).await.map_err(|e| e.to_string())
}

// ============================================================================
// Lore Note Commands
// ============================================================================

#[tauri::command]
pub async fn get_lore_notes(project_id: String) -> Result<Vec<LoreNote>, String> {
    database::get_lore_notes(&project_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn create_lore_note(data: CreateLoreNoteRequest) -> Result<LoreNote, String> {
    // NEW-8 fix: Validate required fields
    if data.title.trim().is_empty() {
        return Err("Lore note title cannot be empty".to_string());
    }
    if data.title.len() > 255 {
        return Err("Lore note title too long (max 255 characters)".to_string());
    }
    database::create_lore_note(data).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn update_lore_note(id: String, data: LoreNote) -> Result<LoreNote, String> {
    database::update_lore_note(&id, data).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_lore_note(id: String) -> Result<(), String> {
    database::delete_lore_note(&id).await.map_err(|e| e.to_string())
}

// ============================================================================
// Project Settings Commands
// ============================================================================

#[tauri::command]
pub async fn get_project_settings(project_id: String) -> Result<Option<ProjectSettings>, String> {
    database::get_project_settings(&project_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn update_project_settings(project_id: String, data: ProjectSettings) -> Result<ProjectSettings, String> {
    database::update_project_settings(&project_id, data).await.map_err(|e| e.to_string())
}

// ============================================================================
// Preset Commands
// ============================================================================

#[tauri::command]
pub async fn get_presets() -> Result<Vec<GenerationPreset>, String> {
    database::get_presets().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn create_preset(data: CreatePresetRequest) -> Result<GenerationPreset, String> {
    database::create_preset(data).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_preset(id: String) -> Result<(), String> {
    database::delete_preset(&id).await.map_err(|e| e.to_string())
}

// ============================================================================
// LLM Commands
// ============================================================================

#[tauri::command]
pub async fn get_available_models(
    app_state: State<'_, Arc<AppState>>,
) -> Result<Vec<ModelInfo>, String> {
    let models_dir = app_state.models_directory.read().clone();

    let path = if let Some(ref path) = models_dir {
        path.clone()
    } else {
        let settings = app_state.app_settings.read();
        if !settings.models_directory.is_empty() {
            settings.models_directory.clone()
        } else {
            // Default to data/models directory
            settings::get_models_dir().to_string_lossy().to_string()
        }
    };

    llm::scan_models_directory(&path).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn load_model(
    path: String,
    app_state: State<'_, Arc<AppState>>,
    llm_state: State<'_, Arc<LlmState>>,
) -> Result<String, String> {
    let settings = app_state.llm_settings.read().clone();

    llm::load_model(&llm_state, &path, &settings)
        .map(|_| {
            // Also update the model_path in app_state settings
            let mut app_settings = app_state.llm_settings.write();
            app_settings.model_path = Some(path.clone());

            format!("Model loaded: {}", path)
        })
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn unload_model(
    llm_state: State<'_, Arc<LlmState>>,
) -> Result<(), String> {
    llm::unload_model(&llm_state);
    Ok(())
}

#[tauri::command]
pub async fn generate_text(
    request: GenerationRequest,
    app_handle: AppHandle,
    llm_state: State<'_, Arc<LlmState>>,
) -> Result<(), String> {
    let llm_state = llm_state.inner().clone();

    tokio::task::spawn_blocking(move || {
        llm::generate_text(&llm_state, &app_handle, request)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Generation task failed: {}", e))?
}

#[tauri::command]
pub async fn stop_generation(
    llm_state: State<'_, Arc<LlmState>>,
) -> Result<(), String> {
    llm::stop_generation(&llm_state);
    Ok(())
}

#[tauri::command]
pub async fn get_llm_settings(
    app_state: State<'_, Arc<AppState>>,
) -> Result<LLMSettings, String> {
    Ok(app_state.llm_settings.read().clone())
}

#[tauri::command]
pub async fn update_llm_settings(
    settings: LLMSettings,
    app_state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    {
        let mut current = app_state.llm_settings.write();
        *current = settings.clone();
    }

    // Save settings to disk
    settings::save_llm_settings(&settings).map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
pub fn get_hardware_info() -> HardwareInfo {
    llm::get_hardware_info()
}

#[tauri::command]
pub async fn select_models_directory(
    app_handle: AppHandle,
    app_state: State<'_, Arc<AppState>>,
) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;

    // Note: blocking_pick_folder() is used intentionally here.
    // Tauri v2's async dialog picker has different semantics.
    // The blocking call is safe because it only blocks this command's task,
    // not the entire async runtime.
    let folder = app_handle
        .dialog()
        .file()
        .blocking_pick_folder();

    if let Some(ref path) = folder {
        let path_str = path.to_string();
        let mut models_dir = app_state.models_directory.write();
        *models_dir = Some(path_str.clone());

        // Also update app settings
        let mut settings = app_state.app_settings.write();
        settings.models_directory = path_str;
    }

    Ok(folder.map(|p| p.to_string()))
}

#[tauri::command]
pub async fn select_model_file(
    app_handle: AppHandle,
    title: Option<String>,
) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;

    let dialog = app_handle
        .dialog()
        .file()
        .add_filter("GGUF Model", &["gguf"])
        .set_title(title.as_deref().unwrap_or("Select Model File"));

    // Note: blocking_pick_file() is used intentionally here.
    // Tauri v2's async dialog picker has different semantics.
    // The blocking call is safe because it only blocks this command's task,
    // not the entire async runtime.
    let file = dialog.blocking_pick_file();

    Ok(file.map(|p| p.to_string()))
}

#[tauri::command]
pub async fn scan_models_directory(path: String) -> Result<Vec<ModelInfo>, String> {
    llm::scan_models_directory(&path).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_model_info(path: String) -> Result<ModelInfo, String> {
    llm::get_model_info(&path).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_model(path: String) -> Result<(), String> {
    let path_buf = std::path::PathBuf::from(&path);

    // Validate file extension
    if path_buf.extension().map(|e| e.to_string_lossy().to_lowercase()) != Some("gguf".to_string()) {
        return Err("Only .gguf model files can be deleted".to_string());
    }

    // Validate path is within known directories (models dir or app data dir)
    let models_dir = crate::settings::get_models_dir();
    let canonical_path = path_buf.canonicalize().map_err(|e| format!("Invalid path: {}", e))?;
    let canonical_models = models_dir.canonicalize().unwrap_or_else(|_| models_dir.clone());

    if !canonical_path.starts_with(&canonical_models) {
        return Err("Model file must be within the models directory".to_string());
    }

    std::fs::remove_file(&path).map_err(|e| e.to_string())
}

// ============================================================================
// App Settings Commands
// ============================================================================

#[tauri::command]
pub async fn get_app_settings(
    app_state: State<'_, Arc<AppState>>,
) -> Result<AppSettings, String> {
    Ok(app_state.app_settings.read().clone())
}

#[tauri::command]
pub async fn update_app_settings(
    settings: AppSettings,
    app_state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    {
        let mut current = app_state.app_settings.write();
        *current = settings.clone();
    }

    // Save settings to disk
    settings::save_app_settings(&settings).map_err(|e| e.to_string())?;

    Ok(())
}

// ============================================================================
// Export/Import Commands
// ============================================================================

#[tauri::command]
pub async fn export_project(project_id: String) -> Result<String, String> {
    // Get project data
    let project = database::get_project(&project_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Project not found")?;

    let chapters = database::get_chapters(&project_id)
        .await
        .map_err(|e| e.to_string())?;

    let characters = database::get_characters(&project_id)
        .await
        .map_err(|e| e.to_string())?;

    let locations = database::get_locations(&project_id)
        .await
        .map_err(|e| e.to_string())?;

    let lore_notes = database::get_lore_notes(&project_id)
        .await
        .map_err(|e| e.to_string())?;

    let settings = database::get_project_settings(&project_id)
        .await
        .map_err(|e| e.to_string())?;

    // Create export data
    let export_data = serde_json::json!({
        "project": project,
        "chapters": chapters,
        "characters": characters,
        "locations": locations,
        "lore_notes": lore_notes,
        "settings": settings,
    });

    serde_json::to_string_pretty(&export_data).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn import_project(filepath: String) -> Result<Project, String> {
    // NEW-1 fix: Validate path is within allowed directories (exports or user-selected)
    let path_buf = std::path::PathBuf::from(&filepath);
    
    // Validate file extension
    if path_buf.extension().map(|e| e.to_string_lossy().to_lowercase()) != Some("json".to_string()) {
        return Err("Import file must be a .json file".to_string());
    }
    
    // Canonicalize and validate path is within allowed directories
    let canonical_path = path_buf.canonicalize().map_err(|e| format!("Invalid path: {}", e))?;
    let exports_dir = crate::settings::get_exports_dir().canonicalize().unwrap_or_else(|_| crate::settings::get_exports_dir());
    let home_dir = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("/"));
    let downloads_dir = home_dir.join("Downloads");
    let desktop_dir = home_dir.join("Desktop");
    
    let is_allowed = canonical_path.starts_with(&exports_dir)
        || canonical_path.starts_with(&downloads_dir)
        || canonical_path.starts_with(&desktop_dir)
        || canonical_path.starts_with(&home_dir);
    
    if !is_allowed {
        return Err("Import file must be within allowed directories (exports, home, downloads, desktop)".to_string());
    }
    
    // Validate file size (max 50 MB for imports)
    let metadata = std::fs::metadata(&filepath).map_err(|e| e.to_string())?;
    if metadata.len() > 50 * 1024 * 1024 {
        return Err("Import file too large (max 50 MB)".to_string());
    }

    let content = std::fs::read_to_string(&filepath).map_err(|e| e.to_string())?;
    let data: serde_json::Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;

    // Parse and import project
    let project: Project = serde_json::from_value(data["project"].clone())
        .map_err(|e| e.to_string())?;

    // Create the project
    let create_request = CreateProjectRequest {
        name: project.name.clone(),
        description: project.description.clone(),
        genre: project.genre.clone(),
    };

    let new_project = database::create_project(create_request)
        .await
        .map_err(|e| e.to_string())?;

    let project_id = new_project.id.clone();

    // Import chapters
    if let Ok(chapters) = serde_json::from_value::<Vec<Chapter>>(data["chapters"].clone()) {
        for chapter in chapters {
            let chapter_request = CreateChapterRequest {
                project_id: project_id.clone(),
                title: chapter.title,
                content: chapter.content,
            };
            if let Err(e) = database::create_chapter(chapter_request).await {
                log::warn!("Failed to import chapter: {}", e);
            }
        }
    }

    // Import characters
if let Ok(characters) = serde_json::from_value::<Vec<Character>>(data["characters"].clone()) {
    for character in characters {
        let character_request = CreateCharacterRequest {
            project_id: project_id.clone(),
            name: character.name,
            age: character.age,
            gender: character.gender,
            role: character.role,
            background: Some(String::new()),
        };
        if let Err(e) = database::create_character(character_request).await {
            log::warn!("Failed to import character: {}", e);
        }
    }
}

    // Import locations
    if let Ok(locations) = serde_json::from_value::<Vec<Location>>(data["locations"].clone()) {
        for location in locations {
            let location_request = CreateLocationRequest {
                project_id: project_id.clone(),
                name: location.name,
                location_type: location.location_type,
                description: location.description,
            };
            if let Err(e) = database::create_location(location_request).await {
                log::warn!("Failed to import location: {}", e);
            }
        }
    }

    // Import lore notes
    if let Ok(lore_notes) = serde_json::from_value::<Vec<LoreNote>>(data["lore_notes"].clone()) {
        for lore_note in lore_notes {
            let lore_request = CreateLoreNoteRequest {
                project_id: project_id.clone(),
                title: lore_note.title,
                category: lore_note.category,
                content: lore_note.content,
            };
            if let Err(e) = database::create_lore_note(lore_request).await {
                log::warn!("Failed to import lore note: {}", e);
            }
        }
    }

    log::info!("Successfully imported project '{}' with all related data", new_project.name);
    Ok(new_project)
}

#[tauri::command]
pub async fn export_all_projects() -> Result<String, String> {
    let projects = database::get_projects().await.map_err(|e| e.to_string())?;

    let mut export_data = Vec::new();

    for p in projects {
        if let Ok(chapters) = database::get_chapters(&p.id).await {
            if let Ok(characters) = database::get_characters(&p.id).await {
                if let Ok(locations) = database::get_locations(&p.id).await {
                    if let Ok(lore_notes) = database::get_lore_notes(&p.id).await {
                        export_data.push(serde_json::json!({
                            "project": p,
                            "chapters": chapters,
                            "characters": characters,
                            "locations": locations,
                            "lore_notes": lore_notes,
                        }));
                    }
                }
            }
        }
    }

    serde_json::to_string_pretty(&export_data).map_err(|e| e.to_string())
}

// ============================================================================
// Memory Optimization Commands
// ============================================================================

/// Get current memory information
#[tauri::command]
pub fn get_memory_info() -> MemoryInfo {
    llm::get_memory_info()
}

/// Get recommended memory settings for current system
#[tauri::command]
pub fn get_recommended_memory_settings() -> serde_json::Value {
    let (context_tokens, batch_size, batch_delay, profile) = llm::get_recommended_memory_settings();

    serde_json::json!({
        "maxContextTokens": context_tokens,
        "generationBatchSize": batch_size,
        "batchDelayMs": batch_delay,
        "memoryProfile": profile
    })
}

/// Get current batch configuration
#[tauri::command]
pub async fn get_batch_config(
    llm_state: State<'_, Arc<LlmState>>,
) -> Result<serde_json::Value, String> {
    let config = llm_state.batch_config.read().clone();

    Ok(serde_json::json!({
        "tokensPerBatch": config.tokens_per_batch,
        "batchDelayMs": config.batch_delay_ms,
        "adaptiveBatchSize": config.adaptive_batch_size,
        "targetCpuPercent": config.target_cpu_percent
    }))
}

/// Update batch configuration
#[tauri::command]
pub async fn update_batch_config(
    tokens_per_batch: usize,
    batch_delay_ms: u64,
    adaptive: bool,
    target_cpu: f32,
    llm_state: State<'_, Arc<LlmState>>,
) -> Result<(), String> {
    // NEW-15 fix: Validate parameters
    if tokens_per_batch == 0 || tokens_per_batch > 1000 {
        return Err("tokens_per_batch must be between 1 and 1000".to_string());
    }
    if batch_delay_ms > 10000 {
        return Err("batch_delay_ms must be at most 10000".to_string());
    }
    if target_cpu <= 0.0 || target_cpu > 100.0 {
        return Err("target_cpu must be between 0.0 (exclusive) and 100.0".to_string());
    }
    
    let mut config = llm_state.batch_config.write();
    config.tokens_per_batch = tokens_per_batch;
    config.batch_delay_ms = batch_delay_ms;
    config.adaptive_batch_size = adaptive;
    config.target_cpu_percent = target_cpu;

    log::info!(
        "Batch config updated: {} tokens/batch, {}ms delay, adaptive: {}, target CPU: {}%",
        tokens_per_batch,
        batch_delay_ms,
        adaptive,
        target_cpu
    );

    Ok(())
}

/// Clear the context window
#[tauri::command]
pub async fn clear_context_window(
    llm_state: State<'_, Arc<LlmState>>,
) -> Result<(), String> {
    llm_state.clear_context();
    log::info!("Context window cleared");
    Ok(())
}

/// Get context window stats
#[tauri::command]
pub async fn get_context_window_stats(
    llm_state: State<'_, Arc<LlmState>>,
) -> Result<serde_json::Value, String> {
    let window = llm_state.context_window.read();

    Ok(serde_json::json!({
        "tokenCount": window.token_count(),
        "chunkCount": window.chunk_count(),
        "isOverCapacity": window.is_over_capacity()
    }))
}

// ============================================================================
// Duo Model Commands (Speculative Decoding)
// ============================================================================

/// Load draft model for speculative decoding
#[tauri::command]
pub async fn load_draft_model(
    path: String,
    llm_state: State<'_, Arc<LlmState>>,
) -> Result<String, String> {
    llm::load_draft_model(&llm_state, &path)
        .map(|_| format!("Draft model loaded: {}", path))
        .map_err(|e| e.to_string())
}

/// Unload draft model
#[tauri::command]
pub async fn unload_draft_model(
    llm_state: State<'_, Arc<LlmState>>,
) -> Result<(), String> {
    llm::unload_draft_model(&llm_state);
    Ok(())
}

/// Enable or disable duo model mode
#[tauri::command]
pub async fn set_duo_model_enabled(
    enabled: bool,
    llm_state: State<'_, Arc<LlmState>>,
) -> Result<(), String> {
    llm::set_duo_model_enabled(&llm_state, enabled);
    Ok(())
}

/// Get duo model status
#[tauri::command]
pub async fn get_duo_model_status(
    llm_state: State<'_, Arc<LlmState>>,
) -> Result<serde_json::Value, String> {
    Ok(llm::get_duo_model_status(&llm_state))
}

// ============================================================================
// CPU Optimization Commands
// ============================================================================

/// Get CPU optimization info (AVX, AVX2, AVX-512, FMA)
#[tauri::command]
pub fn get_cpu_optimizations() -> serde_json::Value {
    serde_json::json!({
        "hasAvx": llm::is_avx_available(),
        "hasAvx2": llm::is_avx2_available(),
        "hasAvx512": llm::is_avx512_available(),
        "hasFma": llm::is_fma_available(),
        "optimizationLevel": llm::get_optimization_info()
    })
}

// ============================================================================
// Backup Commands
// ============================================================================

use crate::backup::{self, BackupInfo};

/// Create a manual backup
#[tauri::command]
pub async fn create_backup(
    app_state: State<'_, Arc<AppState>>,
) -> Result<BackupInfo, String> {
    let db_url = app_state.db_url.clone();
    backup::create_backup(&db_url, false)
        .await
        .map_err(|e| e.to_string())
}

/// List all available backups
#[tauri::command]
pub fn list_backups() -> Result<Vec<BackupInfo>, String> {
    backup::list_backups()
        .map_err(|e| e.to_string())
}

/// Restore database from a backup
#[tauri::command]
pub async fn restore_backup(
    backup_filename: String,
    app_state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let db_url = app_state.db_url.clone();
    backup::restore_backup(&db_url, &backup_filename)
        .await
        .map_err(|e| e.to_string())
}

/// Delete a backup file
#[tauri::command]
pub fn delete_backup(backup_filename: String) -> Result<(), String> {
    backup::delete_backup(&backup_filename)
        .map_err(|e| e.to_string())
}

/// Clean up old automatic backups (keep last N)
#[tauri::command]
pub fn cleanup_backups(keep_count: usize) -> Result<usize, String> {
    backup::cleanup_old_backups(keep_count)
        .map_err(|e| e.to_string())
}

/// Save export content to exports folder
#[tauri::command]
pub fn save_export_to_file(filename: String, content: String) -> Result<String, String> {
    backup::save_export(&filename, &content)
        .map_err(|e| e.to_string())
}

/// Get backups directory path
#[tauri::command]
pub fn get_backups_directory() -> String {
    backup::get_backups_dir()
        .to_string_lossy()
        .to_string()
}

/// Get exports directory path
#[tauri::command]
pub fn get_exports_directory() -> String {
    backup::get_exports_dir()
        .to_string_lossy()
        .to_string()
}

// ============================================================================
// Cache Commands
// ============================================================================

use crate::cache::{self, CacheEntry, CacheEntryInfo, CacheStats, CacheType};

/// Store an item in the cache
#[tauri::command]
pub fn cache_store(
    cache_type: String,
    id: String,
    content: String,
    input_hash: String,
    ttl_seconds: u64,
    project_id: Option<String>,
    tags: Vec<String>,
) -> Result<CacheEntry, String> {
    let ct = match cache_type.as_str() {
        "generation" => CacheType::Generation,
        "dbquery" => CacheType::DbQuery,
        "embedding" => CacheType::Embedding,
        "session" => CacheType::Session,
        _ => return Err(format!("Unknown cache type: {}", cache_type)),
    };

    cache::cache_store(ct, &id, &content, &input_hash, ttl_seconds, project_id, tags)
        .map_err(|e| e.to_string())
}

/// Retrieve an item from the cache
#[tauri::command]
pub fn cache_get(
    cache_type: String,
    id: String,
) -> Result<Option<CacheEntry>, String> {
    let ct = match cache_type.as_str() {
        "generation" => CacheType::Generation,
        "dbquery" => CacheType::DbQuery,
        "embedding" => CacheType::Embedding,
        "session" => CacheType::Session,
        _ => return Err(format!("Unknown cache type: {}", cache_type)),
    };

    cache::cache_get(ct, &id)
        .map_err(|e| e.to_string())
}

/// Check if a cache entry exists
#[tauri::command]
pub fn cache_exists(
    cache_type: String,
    id: String,
) -> Result<bool, String> {
    let ct = match cache_type.as_str() {
        "generation" => CacheType::Generation,
        "dbquery" => CacheType::DbQuery,
        "embedding" => CacheType::Embedding,
        "session" => CacheType::Session,
        _ => return Err(format!("Unknown cache type: {}", cache_type)),
    };

    Ok(cache::cache_exists(ct, &id))
}

/// Remove an item from the cache
#[tauri::command]
pub fn cache_remove(
    cache_type: String,
    id: String,
) -> Result<bool, String> {
    let ct = match cache_type.as_str() {
        "generation" => CacheType::Generation,
        "dbquery" => CacheType::DbQuery,
        "embedding" => CacheType::Embedding,
        "session" => CacheType::Session,
        _ => return Err(format!("Unknown cache type: {}", cache_type)),
    };

    cache::cache_remove(ct, &id)
        .map_err(|e| e.to_string())
}

/// Clear all entries of a specific cache type
#[tauri::command]
pub fn cache_clear_type(cache_type: String) -> Result<u64, String> {
    let ct = match cache_type.as_str() {
        "generation" => CacheType::Generation,
        "dbquery" => CacheType::DbQuery,
        "embedding" => CacheType::Embedding,
        "session" => CacheType::Session,
        _ => return Err(format!("Unknown cache type: {}", cache_type)),
    };

    cache::cache_clear_type(ct)
        .map_err(|e| e.to_string())
}

/// Clear all cache entries
#[tauri::command]
pub fn cache_clear_all() -> Result<u64, String> {
    cache::cache_clear_all()
        .map_err(|e| e.to_string())
}

/// Clean up expired cache entries
#[tauri::command]
pub fn cache_cleanup_expired() -> Result<u64, String> {
    cache::cache_cleanup_expired()
        .map_err(|e| e.to_string())
}

/// Get cache statistics
#[tauri::command]
pub fn cache_get_stats() -> Result<CacheStats, String> {
    cache::cache_get_stats()
        .map_err(|e| e.to_string())
}

/// List cache entries of a specific type
#[tauri::command]
pub fn cache_list(cache_type: String) -> Result<Vec<CacheEntryInfo>, String> {
    let ct = match cache_type.as_str() {
        "generation" => CacheType::Generation,
        "dbquery" => CacheType::DbQuery,
        "embedding" => CacheType::Embedding,
        "session" => CacheType::Session,
        _ => return Err(format!("Unknown cache type: {}", cache_type)),
    };

    cache::cache_list(ct)
        .map_err(|e| e.to_string())
}

/// Store a generated content in cache (convenience function)
#[tauri::command]
pub fn cache_generation(
    prompt_hash: String,
    generated_text: String,
    project_id: Option<String>,
    model_name: Option<String>,
) -> Result<CacheEntry, String> {
    cache::cache_generation(&prompt_hash, &generated_text, project_id, model_name)
        .map_err(|e| e.to_string())
}

/// Find a cached generation by input hash
#[tauri::command]
pub fn find_cached_generation(input_hash: String) -> Result<Option<CacheEntry>, String> {
    cache::find_cached_generation(&input_hash)
        .map_err(|e| e.to_string())
}

/// Store a DB query result in cache
#[tauri::command]
pub fn cache_db_query(
    query_hash: String,
    result_json: String,
    ttl_seconds: u64,
) -> Result<CacheEntry, String> {
    cache::cache_db_query(&query_hash, &result_json, ttl_seconds)
        .map_err(|e| e.to_string())
}

/// Find a cached DB query result
#[tauri::command]
pub fn find_cached_db_query(query_hash: String) -> Result<Option<CacheEntry>, String> {
    cache::find_cached_db_query(&query_hash)
        .map_err(|e| e.to_string())
}

/// Store an embedding in cache
#[tauri::command]
pub fn cache_embedding(
    text_hash: String,
    embedding_json: String,
    model_name: String,
) -> Result<CacheEntry, String> {
    cache::cache_embedding(&text_hash, &embedding_json, &model_name)
        .map_err(|e| e.to_string())
}

/// Find a cached embedding
#[tauri::command]
pub fn find_cached_embedding(text_hash: String) -> Result<Option<CacheEntry>, String> {
    cache::find_cached_embedding(&text_hash)
        .map_err(|e| e.to_string())
}

/// Get cache directory path
#[tauri::command]
pub fn get_cache_directory() -> String {
    cache::get_cache_directory_path()
}

/// Get cache size summary
#[tauri::command]
pub fn get_cache_size() -> Result<(u64, u64), String> {
    cache::get_cache_size()
        .map_err(|e| e.to_string())
}

/// Enforce cache size limits
#[tauri::command]
pub fn enforce_cache_limits() -> Result<u64, String> {
    cache::enforce_cache_limits()
        .map_err(|e| e.to_string())
}
