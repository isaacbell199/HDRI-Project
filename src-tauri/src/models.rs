//! Data models for NexaStory

use serde::{Deserialize, Serialize};

// ============================================================================
// Project Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
#[serde(rename_all = "camelCase")]
pub struct Project {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub cover_image: Option<String>,
    pub genre: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectWithCounts {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub cover_image: Option<String>,
    pub genre: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub chapter_count: i64,
    pub character_count: i64,
    pub location_count: i64,
    pub lore_note_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateProjectRequest {
    pub name: String,
    pub description: Option<String>,
    pub genre: Option<String>,
}

// ============================================================================
// Chapter Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
#[serde(rename_all = "camelCase")]
pub struct Chapter {
    pub id: String,
    pub project_id: String,
    pub title: String,
    pub content: Option<String>,
    #[serde(rename = "orderIndex")]
    pub order_index: i32,
    pub word_count: i32,
    pub status: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateChapterRequest {
    pub project_id: String,
    pub title: String,
    pub content: Option<String>,
}

// ============================================================================
// Character Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
#[serde(rename_all = "camelCase")]
pub struct Character {
    pub id: String,
    pub project_id: String,
    pub name: String,
    pub age: Option<String>,
    pub gender: Option<String>,
    pub role: Option<String>,
    pub occupation: Option<String>,
    pub appearance: Option<String>,
    pub distinguishing_features: Option<String>,
    pub personality: Option<String>,
    pub traits: Option<String>,
    pub flaws: Option<String>,
    pub fears: Option<String>,
    pub desires: Option<String>,
    pub background: Option<String>,
    pub relationships: Option<String>,
    pub skills: Option<String>,
    pub arc: Option<String>,
    pub motivation: Option<String>,
    pub conflicts: Option<String>,
    pub speech_pattern: Option<String>,
    pub catchphrases: Option<String>,
    pub notes: Option<String>,
    pub avatar: Option<String>,
    pub color: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateCharacterRequest {
    pub project_id: String,
    pub name: String,
    #[serde(default)]
    pub age: Option<String>,
    #[serde(default)]
    pub gender: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    /// H5 FIX: Add background field to create request so World View can save character descriptions
    #[serde(default)]
    pub background: Option<String>,
}

// ============================================================================
// Location Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
#[serde(rename_all = "camelCase")]
pub struct Location {
    pub id: String,
    pub project_id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub location_type: Option<String>,
    pub description: Option<String>,
    pub atmosphere: Option<String>,
    pub features: Option<String>,
    pub history: Option<String>,
    pub notes: Option<String>,
    pub image: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateLocationRequest {
    pub project_id: String,
    pub name: String,
    #[serde(rename = "type")]
    #[serde(default)]
    pub location_type: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

// ============================================================================
// Lore Note Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
#[serde(rename_all = "camelCase")]
pub struct LoreNote {
    pub id: String,
    pub project_id: String,
    pub title: String,
    pub category: Option<String>,
    pub content: Option<String>,
    pub tags: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateLoreNoteRequest {
    pub project_id: String,
    pub title: String,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
}

// ============================================================================
// Project Settings Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSettings {
    pub id: String,
    pub project_id: String,
    pub target_word_count: i32,
    pub daily_word_goal: i32,
    pub auto_save: bool,
    pub auto_save_interval: i32,
    pub context_paragraphs: i32,
    pub genres: Option<String>,
    pub themes: Option<String>,
    pub target_audience: Option<String>,
    pub writing_style: Option<String>,
    pub narrative_pov: Option<String>,
    pub content_rating: Option<String>,
    pub content_warnings: Option<String>,
    pub tone_preferences: Option<String>,
    pub time_period: Option<String>,
    pub world_type: Option<String>,
    pub language: Option<String>,
    pub language_style: Option<String>,
    pub adult_content: Option<String>,
    pub adult_intensity: Option<String>,
}

// ============================================================================
// Generation Preset Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
#[serde(rename_all = "camelCase")]
pub struct GenerationPreset {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub preset_type: String,
    pub positive_prompt: Option<String>,
    pub negative_prompt: Option<String>,
    pub selected_tone: Option<String>,
    pub custom_tone_instruction: Option<String>,
    pub custom_style_name: Option<String>,
    pub custom_style_instruction: Option<String>,
    pub custom_genre_name: Option<String>,
    pub custom_genre_instruction: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreatePresetRequest {
    pub name: String,
    #[serde(rename = "type")]
    pub preset_type: String,
    #[serde(default)]
    pub positive_prompt: Option<String>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub selected_tone: Option<String>,
    #[serde(default)]
    pub custom_tone_instruction: Option<String>,
    #[serde(default)]
    pub custom_style_name: Option<String>,
    #[serde(default)]
    pub custom_style_instruction: Option<String>,
    #[serde(default)]
    pub custom_genre_name: Option<String>,
    #[serde(default)]
    pub custom_genre_instruction: Option<String>,
}

// ============================================================================
// Duo Model Config (Speculative Decoding)
// ============================================================================

/// Configuration for the Duo Model (Speculative Decoding) feature
/// This allows using a smaller "draft" model to predict tokens faster,
/// while a larger "main" model verifies and corrects the predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DuoModelConfig {
    /// Enable or disable duo model mode
    pub enabled: bool,
    /// Path to the draft model (smaller, faster model)
    pub draft_model_path: Option<String>,
    /// Number of tokens the draft model speculates before verification
    pub draft_tokens: i32,
    /// Acceptance threshold for draft predictions (0.0 - 1.0)
    /// Higher values = more strict verification
    pub acceptance_threshold: f32,
    /// Use same tokenizer for both models (required for speculative decoding)
    pub shared_tokenizer: bool,
}

impl Default for DuoModelConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            draft_model_path: None,
            draft_tokens: 4,
            acceptance_threshold: 0.9,
            shared_tokenizer: true,
        }
    }
}

// ============================================================================
// Dynamic Sampling Config (Adaptive Parameters)
// ============================================================================

/// Scene types for dynamic sampling adaptation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SceneType {
    /// Action scenes - fast, direct, impactful
    Action,
    /// Dialogue scenes - natural, coherent
    Dialogue,
    /// Description scenes - rich, detailed
    Description,
    /// Sensation scenes - sensory details (smell, touch, etc.)
    Sensation,
    /// Emotional scenes - expressive, intense
    Emotion,
    /// Default/unknown scene type
    Default,
}

impl Default for SceneType {
    fn default() -> Self {
        SceneType::Default
    }
}

/// Configuration for Dynamic Sampling (Adaptive Parameters)
///
/// This module provides intelligent parameter adjustment based on scene content.
/// When Duo Model is enabled, the Draft Model uses FIXED parameters while
/// the Main Model uses DYNAMIC parameters for creative adaptation.
///
/// Key strategy:
/// - Min-P is used as the primary adaptation mechanism (more stable than temperature)
/// - Temperature changes are smoothed to avoid disrupting Duo Model synchronization
/// - Draft Model always uses stable parameters to maintain prediction accuracy
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DynamicSamplingConfig {
    /// Enable or disable dynamic sampling
    pub enabled: bool,

    /// Min-P base value (filters improbable tokens based on model confidence)
    /// Range: 0.0 - 1.0, recommended: 0.02 - 0.1
    pub min_p: f32,

    /// Min-P for action scenes (higher = more direct, less creative)
    /// Recommended: 0.08
    pub min_p_action: f32,

    /// Min-P for dialogue scenes (balanced)
    /// Recommended: 0.05
    pub min_p_dialogue: f32,

    /// Min-P for description scenes (slightly creative)
    /// Recommended: 0.04
    pub min_p_description: f32,

    /// Min-P for sensation scenes (most creative, sensory details)
    /// Recommended: 0.02
    pub min_p_sensation: f32,

    /// Min-P for emotional scenes (expressive)
    /// Recommended: 0.03
    pub min_p_emotion: f32,

    /// Temperature adjustment for action scenes
    /// Base temperature + this adjustment = action temperature
    pub temp_adjust_action: f32,

    /// Temperature adjustment for dialogue scenes
    pub temp_adjust_dialogue: f32,

    /// Temperature adjustment for description scenes
    pub temp_adjust_description: f32,

    /// Temperature adjustment for sensation scenes
    pub temp_adjust_sensation: f32,

    /// Temperature adjustment for emotional scenes
    pub temp_adjust_emotion: f32,

    /// Transition rate for smooth parameter changes (0.0 - 1.0)
    /// Higher = faster transition, lower = smoother
    /// Recommended: 0.05 (changes over ~20 tokens)
    pub transition_rate: f32,

    /// Number of tokens for complete transition (smoothing window)
    pub smoothing_window: i32,

    // === Draft Model Parameters (FIXED for Duo Model) ===
    /// Draft model temperature (always fixed for prediction stability)
    pub draft_temperature: f32,

    /// Draft model top-p (always fixed)
    pub draft_top_p: f32,

    /// Draft model min-p (always fixed)
    pub draft_min_p: f32,

    /// Draft model repeat penalty (always fixed)
    pub draft_repeat_penalty: f32,
}

impl Default for DynamicSamplingConfig {
    fn default() -> Self {
        Self {
            enabled: false,  // Disabled by default for backward compatibility

            // Min-P values (primary adaptation mechanism)
            min_p: 0.05,
            min_p_action: 0.08,
            min_p_dialogue: 0.05,
            min_p_description: 0.04,
            min_p_sensation: 0.02,
            min_p_emotion: 0.03,

            // Temperature adjustments (secondary, smaller changes)
            temp_adjust_action: -0.1,      // Slightly lower for action
            temp_adjust_dialogue: 0.0,     // Base temperature
            temp_adjust_description: 0.0,  // Base temperature
            temp_adjust_sensation: 0.1,    // Slightly higher for creativity
            temp_adjust_emotion: 0.1,      // Slightly higher for expressiveness

            // Smoothing parameters
            transition_rate: 0.05,
            smoothing_window: 20,

            // Draft model FIXED parameters (for Duo Model stability)
            draft_temperature: 0.8,
            draft_top_p: 0.9,
            draft_min_p: 0.05,
            draft_repeat_penalty: 1.07,
        }
    }
}

// ============================================================================
// LLM Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LLMSettings {
    pub model_path: Option<String>,
    pub cpu_threads: i32,
    pub batch_size: i32,
    pub context_length: i32,
    pub seed: i32,
    pub gpu_layers: i32,
    pub use_gpu: bool,
    pub gpu_device: Option<i32>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    /// Min-P sampling parameter (filters improbable tokens)
    /// Range: 0.0 - 1.0, higher = more restrictive
    pub min_p: f32,
    pub max_tokens: i32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub flash_attention: bool,
    pub cache_type: String,
    pub stop_sequences: Vec<String>,
    pub max_context_tokens: i32,
    pub generation_batch_size: i32,
    pub batch_delay_ms: i32,
    pub adaptive_batch_sizing: bool,
    pub target_cpu_percent: f32,
    pub enable_context_compression: bool,
    pub memory_profile: String,
    /// Duo Model (Speculative Decoding) configuration
    pub duo_model_config: DuoModelConfig,
    /// Dynamic Sampling (Adaptive Parameters) configuration
    pub dynamic_sampling: DynamicSamplingConfig,
}

impl Default for LLMSettings {
    fn default() -> Self {
        Self {
            model_path: None,
            cpu_threads: num_cpus::get() as i32,
            batch_size: 512,
            context_length: 4096,
            seed: -1,
            gpu_layers: 0,
            use_gpu: false,
            gpu_device: None,
            temperature: 0.75,  // Optimized for coherent story generation
            top_p: 0.92,        // Slightly higher for better variety
            top_k: 50,          // Increased for more options
            min_p: 0.03,        // Lower to allow more creative tokens
            max_tokens: 500,
            repeat_penalty: 1.12,  // Increased to prevent repetition loops
            frequency_penalty: 0.5,
            presence_penalty: 0.4,
            use_mmap: true,
            use_mlock: false,
            flash_attention: false,
            cache_type: "f16".to_string(),
            stop_sequences: vec![],
            max_context_tokens: 2048,
            generation_batch_size: 50,
            batch_delay_ms: 50,
            adaptive_batch_sizing: true,
            target_cpu_percent: 70.0,
            enable_context_compression: true,
            memory_profile: "balanced".to_string(),
            duo_model_config: DuoModelConfig::default(),
            dynamic_sampling: DynamicSamplingConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareInfo {
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub total_memory_gb: f64,
    pub available_memory_gb: f64,
    pub has_gpu: bool,
    pub gpu_name: Option<String>,
    pub gpu_memory_gb: Option<f64>,
    pub recommended_threads: usize,
    pub recommended_batch_size: usize,
    pub best_backend: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    /// Unique identifier for the model (usually the file path or a generated ID)
    pub id: String,
    pub name: String,
    pub path: String,
    pub size_mb: u64,
    pub parameters: Option<String>,
    pub quantization: Option<String>,
    pub context_length: usize,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            path: String::new(),
            size_mb: 0,
            parameters: None,
            quantization: None,
            context_length: 4096,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationRequest {
    pub mode: String,
    pub text: String,
    pub context: Option<String>,
    #[serde(default)]
    pub characters: Vec<Character>,
    #[serde(default)]
    pub locations: Vec<Location>,
    #[serde(default)]
    pub lore_notes: Vec<LoreNote>,
    pub project_settings: Option<ProjectSettings>,
    pub selected_character_id: Option<String>,
    pub positive_prompt: Option<String>,
    pub negative_prompt: Option<String>,
    pub custom_style_name: Option<String>,
    pub custom_style_instruction: Option<String>,
    pub custom_genre_name: Option<String>,
    pub custom_genre_instruction: Option<String>,
    pub selected_tone: Option<String>,
    pub custom_tone_instruction: Option<String>,
    /// Theme name (e.g., "Love & Relationships", "Survival")
    pub theme_name: Option<String>,
    /// Theme instruction / description for detailed theme context
    pub theme_instruction: Option<String>,
    /// Language mode: "fr-fr", "fr-en", "en-fr", "en-en" (input -> output)
    pub language_mode: Option<String>,
    pub start_phrase: Option<String>,
    pub end_phrase: Option<String>,
    #[serde(default = "default_true")]
    pub stream: bool,
    /// Model-specific system prompt to guide generation behavior
    pub system_prompt: Option<String>,
    /// Sampling parameters (can override settings)
    pub temperature: Option<f32>,
    pub max_tokens: Option<i32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

fn default_true() -> bool { true }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationChunk {
    pub content: String,
    pub done: bool,
}

// ============================================================================
// App Settings Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSettings {
    pub is_dark_mode: bool,
    pub font_size: String,
    pub auto_check_updates: bool,
    pub models_directory: String,
    pub default_export_directory: String,
    pub language: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        // Default to empty string - the app will use data/models directory
        // which is created next to the executable
        Self {
            is_dark_mode: false,
            font_size: "medium".to_string(),
            auto_check_updates: true,
            models_directory: String::new(),  // Empty = use data/models next to exe
            default_export_directory: String::new(),  // Empty = use data/exports next to exe
            language: "en".to_string(),
        }
    }
}
