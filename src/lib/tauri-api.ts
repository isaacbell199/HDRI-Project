/**
 * Tauri API Wrapper for NexaStory Desktop
 * 
 * This module provides a unified interface for communicating with the Tauri backend.
 * It replaces fetch('/api/...') calls with Tauri invoke() commands.
 * 
 * When not running in Tauri, all functions return mock data for demo mode.
 */

// ============================================================================
// Types
// ============================================================================

export interface Project {
  id: string
  name: string
  description: string | null
  coverImage: string | null
  genre: string | null
  createdAt: string
  updatedAt: string
}

export interface ProjectWithCounts extends Project {
  chapterCount: number
  characterCount: number
  locationCount: number
  loreNoteCount: number
  wordCount?: number
}

export interface Chapter {
  id: string
  projectId: string
  title: string
  content: string | null
  orderIndex: number
  wordCount: number
  status: string
  createdAt: string
  updatedAt: string
}

export interface Character {
  id: string
  projectId: string
  name: string
  age: string | null
  gender: string | null
  role: string | null
  occupation: string | null
  appearance: string | null
  distinguishingFeatures: string | null
  personality: string | null
  traits: string | null
  flaws: string | null
  fears: string | null
  desires: string | null
  background: string | null
  relationships: string | null
  skills: string | null
  arc: string | null
  motivation: string | null
  conflicts: string | null
  speechPattern: string | null
  catchphrases: string | null
  notes: string | null
  avatar: string | null
  color: string | null
  createdAt: string
  updatedAt: string
}

export interface Location {
  id: string
  projectId: string
  name: string
  type: string | null
  description: string | null
  atmosphere: string | null
  features: string | null
  history: string | null
  notes: string | null
  image: string | null
  createdAt: string
  updatedAt: string
}

export interface LoreNote {
  id: string
  projectId: string
  title: string
  category: string | null
  content: string | null
  tags: string | null
  createdAt: string
  updatedAt: string
}

export interface ProjectSettings {
  id: string
  projectId: string
  targetWordCount: number
  dailyWordGoal: number
  autoSave: boolean
  autoSaveInterval: number
  contextParagraphs: number
  genres: string | null
  themes: string | null
  targetAudience: string | null
  writingStyle: string | null
  narrativePov: string | null
  contentRating: string | null
  contentWarnings: string | null
  tonePreferences: string | null
  timePeriod: string | null
  worldType: string | null
  language: string | null
  languageStyle: string | null
  adultContent: string | null
  adultIntensity: string | null
}

export interface GenerationPreset {
  id: string
  name: string
  type: string
  positivePrompt: string | null
  negativePrompt: string | null
  selectedTone: string | null
  customToneInstruction: string | null
  customStyleName: string | null
  customStyleInstruction: string | null
  customGenreName: string | null
  customGenreInstruction: string | null
  createdAt: string
  updatedAt: string
}

export interface DuoModelConfig {
  enabled: boolean
  draftModelPath: string | null
  draftTokens: number
  acceptanceThreshold: number
  sharedTokenizer: boolean
}

// ============================================================================
// Dynamic Sampling Types
// ============================================================================

export type SceneType = 'action' | 'dialogue' | 'description' | 'sensation' | 'emotion' | 'default'

export interface DynamicSamplingConfig {
  enabled: boolean
  minP: number
  minPAction: number
  minPDialogue: number
  minPDescription: number
  minPSensation: number
  minPEmotion: number
  tempAdjustAction: number
  tempAdjustDialogue: number
  tempAdjustDescription: number
  tempAdjustSensation: number
  tempAdjustEmotion: number
  transitionRate: number
  smoothingWindow: number
  draftTemperature: number
  draftTopP: number
  draftMinP: number
  draftRepeatPenalty: number
}

export interface LLMSettings {
  modelPath: string | null
  cpuThreads: number
  batchSize: number
  contextLength: number
  seed: number
  gpuLayers: number
  useGpu: boolean
  gpuDevice: number | null
  temperature: number
  topP: number
  topK: number
  minP: number
  maxTokens: number
  repeatPenalty: number
  frequencyPenalty: number
  presencePenalty: number
  useMmap: boolean
  useMlock: boolean
  flashAttention: boolean
  cacheType: string
  stopSequences: string[]
  maxContextTokens: number
  generationBatchSize: number
  batchDelayMs: number
  adaptiveBatchSizing: boolean
  targetCpuPercent: number
  enableContextCompression: boolean
  memoryProfile: string
  duoModelConfig: DuoModelConfig
  dynamicSampling: DynamicSamplingConfig
}

export interface AppSettings {
  isDarkMode: boolean
  fontSize: string
  autoCheckUpdates: boolean
  modelsDirectory: string
  defaultExportDirectory: string
  language: string
}

export interface HardwareInfo {
  cpuCores: number
  cpuThreads: number
  totalMemoryGb: number
  availableMemoryGb: number
  hasGpu: boolean
  gpuName: string | null
  gpuMemoryGb: number | null
  recommendedThreads: number
  recommendedBatchSize: number
  bestBackend: string | null
}

export interface ModelInfo {
  id: string
  name: string
  path: string
  sizeMb: number
  parameters: string | null
  quantization: string | null
  contextLength: number
}

// Language mode type: input language -> output language
export type LanguageMode = 'fr-fr' | 'fr-en' | 'en-fr' | 'en-en'

export interface GenerationRequest {
  mode: string
  text: string
  context?: string
  characters?: Character[]
  locations?: Location[]
  loreNotes?: LoreNote[]
  projectSettings?: ProjectSettings
  selectedCharacterId?: string
  positivePrompt?: string
  negativePrompt?: string
  customStyleName?: string
  customStyleInstruction?: string
  customGenreName?: string
  customGenreInstruction?: string
  selectedTone?: string
  customToneInstruction?: string
  /** Theme name (e.g., "Love & Relationships") */
  themeName?: string
  /** Theme instruction / description for detailed theme context */
  themeInstruction?: string
  languageMode?: LanguageMode  // FR→FR, FR→EN, EN→FR, EN→EN
  startPhrase?: string
  endPhrase?: string
  stream?: boolean
  temperature?: number
  maxTokens?: number
  systemPrompt?: string  // Model-specific system prompt
  // Sampling parameters from preset
  topP?: number
  topK?: number
  minP?: number
  repeatPenalty?: number
  frequencyPenalty?: number
  presencePenalty?: number
}

export interface GenerationChunk {
  content: string
  done: boolean
}

export interface MemoryInfo {
  totalGb: number
  availableGb: number
  usedGb: number
  usedPercent: number
  isUnderPressure: boolean
  recommendedContextTokens: number
  recommendedBatchSize: number
}

// ============================================================================
// Utility: Check if running in Tauri
// ============================================================================

export function isTauri(): boolean {
  return typeof window !== 'undefined' && '__TAURI__' in window
}

// ============================================================================
// Project Commands
// ============================================================================

export async function getProjects(): Promise<ProjectWithCounts[]> {
  if (!isTauri()) return []
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<ProjectWithCounts[]>('get_projects')
}

export async function getProject(id: string): Promise<Project | null> {
  if (!isTauri()) return null
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<Project | null>('get_project', { id })
}

export async function createProject(data: {
  name: string
  description?: string
  genre?: string
}): Promise<Project> {
  if (!isTauri()) throw new Error('Not in Tauri')
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<Project>('create_project', { data })
}

export async function updateProject(id: string, data: Project): Promise<Project> {
  if (!isTauri()) throw new Error('Not in Tauri')
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<Project>('update_project', { id, data })
}

export async function deleteProject(id: string): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('delete_project', { id })
}

// ============================================================================
// Chapter Commands
// ============================================================================

export async function getChapters(projectId: string): Promise<Chapter[]> {
  if (!isTauri()) return []
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<Chapter[]>('get_chapters', { projectId })
}

export async function getChapter(id: string): Promise<Chapter | null> {
  if (!isTauri()) return null
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<Chapter | null>('get_chapter', { id })
}

export async function createChapter(data: {
  projectId: string
  title: string
  content?: string
}): Promise<Chapter> {
  if (!isTauri()) throw new Error('Not in Tauri')
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<Chapter>('create_chapter', { data })
}

export async function updateChapter(id: string, data: Chapter): Promise<Chapter> {
  if (!isTauri()) throw new Error('Not in Tauri')
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<Chapter>('update_chapter', { id, data })
}

export async function deleteChapter(id: string): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('delete_chapter', { id })
}

// ============================================================================
// Character Commands
// ============================================================================

export async function getCharacters(projectId: string): Promise<Character[]> {
  if (!isTauri()) return []
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<Character[]>('get_characters', { projectId })
}

export async function createCharacter(data: {
  projectId: string
  name: string
  age?: string
  gender?: string
  role?: string
  /** H5 FIX: Character background/description */
  background?: string
}): Promise<Character> {
  if (!isTauri()) throw new Error('Not in Tauri')
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<Character>('create_character', { data })
}

export async function updateCharacter(id: string, data: Character): Promise<Character> {
  if (!isTauri()) throw new Error('Not in Tauri')
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<Character>('update_character', { id, data })
}

export async function deleteCharacter(id: string): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('delete_character', { id })
}

// ============================================================================
// Location Commands
// ============================================================================

export async function getLocations(projectId: string): Promise<Location[]> {
  if (!isTauri()) return []
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<Location[]>('get_locations', { projectId })
}

export async function createLocation(data: {
  projectId: string
  name: string
  type?: string
  description?: string
}): Promise<Location> {
  if (!isTauri()) throw new Error('Not in Tauri')
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<Location>('create_location', { data })
}

export async function updateLocation(id: string, data: Location): Promise<Location> {
  if (!isTauri()) throw new Error('Not in Tauri')
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<Location>('update_location', { id, data })
}

export async function deleteLocation(id: string): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('delete_location', { id })
}

// ============================================================================
// Lore Note Commands
// ============================================================================

export async function getLoreNotes(projectId: string): Promise<LoreNote[]> {
  if (!isTauri()) return []
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<LoreNote[]>('get_lore_notes', { projectId })
}

export async function createLoreNote(data: {
  projectId: string
  title: string
  category?: string
  content?: string
}): Promise<LoreNote> {
  if (!isTauri()) throw new Error('Not in Tauri')
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<LoreNote>('create_lore_note', { data })
}

export async function updateLoreNote(id: string, data: LoreNote): Promise<LoreNote> {
  if (!isTauri()) throw new Error('Not in Tauri')
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<LoreNote>('update_lore_note', { id, data })
}

export async function deleteLoreNote(id: string): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('delete_lore_note', { id })
}

// ============================================================================
// Project Settings Commands
// ============================================================================

export async function getProjectSettings(projectId: string): Promise<ProjectSettings | null> {
  if (!isTauri()) return null
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<ProjectSettings | null>('get_project_settings', { projectId })
}

export async function updateProjectSettings(projectId: string, data: ProjectSettings): Promise<ProjectSettings> {
  if (!isTauri()) throw new Error('Not in Tauri')
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<ProjectSettings>('update_project_settings', { projectId, data })
}

// ============================================================================
// Preset Commands
// ============================================================================

export async function getPresets(): Promise<GenerationPreset[]> {
  if (!isTauri()) return []
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<GenerationPreset[]>('get_presets')
}

export async function createPreset(data: {
  name: string
  type: string
  positivePrompt?: string
  negativePrompt?: string
}): Promise<GenerationPreset> {
  if (!isTauri()) throw new Error('Not in Tauri')
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<GenerationPreset>('create_preset', { data })
}

export async function deletePreset(id: string): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('delete_preset', { id })
}

// ============================================================================
// LLM Commands
// ============================================================================

export async function getAvailableModels(): Promise<ModelInfo[]> {
  if (!isTauri()) return []
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<ModelInfo[]>('get_available_models')
}

export async function loadModel(path: string): Promise<string> {
  if (!isTauri()) return 'Demo mode'
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<string>('load_model', { path })
}

export async function unloadModel(): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('unload_model')
}

export async function generateText(request: GenerationRequest): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('generate_text', { request })
}

export async function stopGeneration(): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('stop_generation')
}

export async function getLLMSettings(): Promise<LLMSettings> {
  if (!isTauri()) {
    return {
      modelPath: null,
      cpuThreads: 4,
      batchSize: 512,
      contextLength: 4096,
      seed: -1,
      gpuLayers: 0,
      useGpu: false,
      gpuDevice: null,
      temperature: 0.9,
      topP: 0.9,
      topK: 40,
      minP: 0.05,
      maxTokens: 500,
      repeatPenalty: 1.07,
      frequencyPenalty: 0.5,
      presencePenalty: 0.4,
      useMmap: true,
      useMlock: false,
      flashAttention: false,
      cacheType: 'f16',
      stopSequences: [],
      maxContextTokens: 2048,
      generationBatchSize: 50,
      batchDelayMs: 50,
      adaptiveBatchSizing: true,
      targetCpuPercent: 70,
      enableContextCompression: true,
      memoryProfile: 'balanced',
      duoModelConfig: {
        enabled: false,
        draftModelPath: null,
        draftTokens: 4,
        acceptanceThreshold: 0.9,
        sharedTokenizer: true
      },
      dynamicSampling: {
        enabled: false,
        minP: 0.05,
        minPAction: 0.08,
        minPDialogue: 0.05,
        minPDescription: 0.04,
        minPSensation: 0.02,
        minPEmotion: 0.03,
        tempAdjustAction: -0.1,
        tempAdjustDialogue: 0.0,
        tempAdjustDescription: 0.0,
        tempAdjustSensation: 0.1,
        tempAdjustEmotion: 0.1,
        transitionRate: 0.05,
        smoothingWindow: 20,
        draftTemperature: 0.8,
        draftTopP: 0.9,
        draftMinP: 0.05,
        draftRepeatPenalty: 1.07
      }
    }
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<LLMSettings>('get_llm_settings')
}

export async function updateLLMSettings(settings: LLMSettings): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('update_llm_settings', { settings })
}

export async function getHardwareInfo(): Promise<HardwareInfo> {
  if (!isTauri()) {
    return {
      cpuCores: 8,
      cpuThreads: 16,
      totalMemoryGb: 16,
      availableMemoryGb: 8,
      hasGpu: false,
      gpuName: null,
      gpuMemoryGb: null,
      recommendedThreads: 8,
      recommendedBatchSize: 512,
      bestBackend: 'CPU'
    }
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<HardwareInfo>('get_hardware_info')
}

export async function selectModelsDirectory(): Promise<string | null> {
  if (!isTauri()) return null
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<string | null>('select_models_directory')
}

export async function selectModelFile(title?: string): Promise<string | null> {
  if (!isTauri()) return null
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<string | null>('select_model_file', { title })
}

export async function scanModelsDirectory(path: string): Promise<ModelInfo[]> {
  if (!isTauri()) return []
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<ModelInfo[]>('scan_models_directory', { path })
}

export async function getModelInfo(path: string): Promise<ModelInfo> {
  if (!isTauri()) throw new Error('Not in Tauri')
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<ModelInfo>('get_model_info', { path })
}

export async function deleteModel(path: string): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('delete_model', { path })
}

// ============================================================================
// App Settings Commands
// ============================================================================

export async function getAppSettings(): Promise<AppSettings> {
  if (!isTauri()) {
    return {
      isDarkMode: false,
      fontSize: 'medium',
      autoCheckUpdates: true,
      modelsDirectory: '',
      defaultExportDirectory: '',
      language: 'en'
    }
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<AppSettings>('get_app_settings')
}

export async function updateAppSettings(settings: AppSettings): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('update_app_settings', { settings })
}

// ============================================================================
// Export/Import Commands
// ============================================================================

export async function exportProject(projectId: string): Promise<string> {
  if (!isTauri()) return ''
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<string>('export_project', { projectId })
}

export async function importProject(filepath: string): Promise<Project> {
  if (!isTauri()) throw new Error('Not in Tauri')
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<Project>('import_project', { filepath })
}

export async function exportAllProjects(): Promise<string> {
  if (!isTauri()) return ''
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<string>('export_all_projects')
}

// ============================================================================
// Memory Commands
// ============================================================================

export async function getMemoryInfo(): Promise<MemoryInfo> {
  if (!isTauri()) {
    return {
      totalGb: 16,
      availableGb: 8,
      usedGb: 8,
      usedPercent: 50,
      isUnderPressure: false,
      recommendedContextTokens: 2048,
      recommendedBatchSize: 50
    }
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<MemoryInfo>('get_memory_info')
}

export async function getRecommendedMemorySettings(): Promise<{
  maxContextTokens: number
  generationBatchSize: number
  batchDelayMs: number
  memoryProfile: string
}> {
  if (!isTauri()) {
    return {
      maxContextTokens: 2048,
      generationBatchSize: 50,
      batchDelayMs: 50,
      memoryProfile: 'balanced'
    }
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke('get_recommended_memory_settings')
}

export async function getBatchConfig(): Promise<{
  tokensPerBatch: number
  batchDelayMs: number
  adaptiveBatchSize: boolean
  targetCpuPercent: number
}> {
  if (!isTauri()) {
    return {
      tokensPerBatch: 50,
      batchDelayMs: 50,
      adaptiveBatchSize: true,
      targetCpuPercent: 70
    }
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke('get_batch_config')
}

export async function updateBatchConfig(
  tokensPerBatch: number,
  batchDelayMs: number,
  adaptive: boolean,
  targetCpu: number
): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke('update_batch_config', {
    tokensPerBatch,
    batchDelayMs,
    adaptive,
    targetCpu
  })
}

export async function clearContextWindow(): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke('clear_context_window')
}

export async function getContextWindowStats(): Promise<{
  tokenCount: number
  chunkCount: number
  isOverCapacity: boolean
}> {
  if (!isTauri()) {
    return {
      tokenCount: 0,
      chunkCount: 0,
      isOverCapacity: false
    }
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke('get_context_window_stats')
}

// ============================================================================
// Duo Model Commands (Speculative Decoding)
// ============================================================================

export async function loadDraftModel(path: string): Promise<string> {
  if (!isTauri()) return 'Demo mode'
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<string>('load_draft_model', { path })
}

export async function unloadDraftModel(): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('unload_draft_model')
}

export async function setDuoModelEnabled(enabled: boolean): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('set_duo_model_enabled', { enabled })
}

export async function getDuoModelStatus(): Promise<{
  enabled: boolean
  draftModelLoaded: boolean
  mainModelLoaded: boolean
  ready: boolean
  mainModelPath: string | null
  draftModelPath: string | null
  recommendedPairs: Array<{ main: string; draft: string }>
}> {
  if (!isTauri()) {
    return {
      enabled: false,
      draftModelLoaded: false,
      mainModelLoaded: false,
      ready: false,
      mainModelPath: null,
      draftModelPath: null,
      recommendedPairs: [
        { main: 'Llama 3.2 3B', draft: 'Llama 3.2 1B' },
        { main: 'Qwen 2.5 7B', draft: 'Qwen 2.5 1.5B' },
        { main: 'Mistral 7B', draft: 'Mistral 0.5B' }
      ]
    }
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke('get_duo_model_status')
}

// ============================================================================
// Event Listeners
// ============================================================================

export function onGenerationChunk(callback: (chunk: GenerationChunk) => void): () => void {
  if (!isTauri()) return () => {};

  let unlistenFn: (() => void) | null = null;
  let cancelled = false;

  const unlistenPromise = import('@tauri-apps/api/event').then(({ listen }) => {
    return listen<GenerationChunk>('generation-chunk', (event) => {
      callback(event.payload);
    });
  });

  unlistenPromise.then(fn => {
    unlistenFn = fn;
    // If already cancelled during the async gap, unlisten immediately
    if (cancelled) {
      fn();
    }
  });

  return () => {
    cancelled = true;
    if (unlistenFn) {
      unlistenFn();
    }
    // If listen hasn't resolved yet, the .then above will handle cleanup
  };
}

// ============================================================================
// CPU Optimization Commands
// ============================================================================

export interface CpuOptimizations {
  hasAvx: boolean
  hasAvx2: boolean
  hasAvx512: boolean
  hasFma: boolean
  optimizationLevel: string
}

export async function getCpuOptimizations(): Promise<CpuOptimizations> {
  if (!isTauri()) {
    return {
      hasAvx: true,
      hasAvx2: true,
      hasAvx512: false,
      hasFma: true,
      optimizationLevel: 'AVX2 (Excellent)'
    }
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<CpuOptimizations>('get_cpu_optimizations')
}

// ============================================================================
// Backup Commands
// ============================================================================

export interface BackupInfo {
  id: string
  filename: string
  createdAt: string
  sizeBytes: number
  projectCount: number
  chapterCount: number
  isAuto: boolean
}

export async function createBackup(): Promise<BackupInfo> {
  if (!isTauri()) {
    return {
      id: Date.now().toString(),
      filename: `manual_backup_${Date.now()}.db`,
      createdAt: new Date().toISOString(),
      sizeBytes: 1024 * 1024,
      projectCount: 0,
      chapterCount: 0,
      isAuto: false
    }
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<BackupInfo>('create_backup')
}

export async function listBackups(): Promise<BackupInfo[]> {
  if (!isTauri()) {
    return []
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<BackupInfo[]>('list_backups')
}

export async function restoreBackup(backupFilename: string): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('restore_backup', { backupFilename })
}

export async function deleteBackup(backupFilename: string): Promise<void> {
  if (!isTauri()) return
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<void>('delete_backup', { backupFilename })
}

export async function cleanupBackups(keepCount: number): Promise<number> {
  if (!isTauri()) return 0
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<number>('cleanup_backups', { keepCount })
}

export async function saveExportToFile(filename: string, content: string): Promise<string> {
  if (!isTauri()) return ''
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<string>('save_export_to_file', { filename, content })
}

export async function getBackupsDirectory(): Promise<string> {
  if (!isTauri()) return ''
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<string>('get_backups_directory')
}

export async function getExportsDirectory(): Promise<string> {
  if (!isTauri()) return ''
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<string>('get_exports_directory')
}

// ============================================================================
// Cache Commands
// ============================================================================

export type CacheType = 'generation' | 'dbquery' | 'embedding' | 'session'

export interface CacheEntry {
  id: string
  cacheType: CacheType
  content: string
  inputHash: string
  createdAt: number
  lastAccessed: number
  accessCount: number
  sizeBytes: number
  ttlSeconds: number
  projectId: string | null
  tags: string[]
}

export interface CacheEntryInfo {
  id: string
  cacheType: string
  createdAt: string
  lastAccessed: string
  accessCount: number
  sizeBytes: number
  projectId: string | null
  tags: string[]
}

export interface CacheStats {
  totalEntries: number
  totalSizeBytes: number
  entriesByType: Record<string, number>
  sizeByType: Record<string, number>
  hitCount: number
  missCount: number
  cacheDirectory: string
}

export async function cacheStore(
  cacheType: CacheType,
  id: string,
  content: string,
  inputHash: string,
  ttlSeconds: number = 0,
  projectId?: string,
  tags: string[] = []
): Promise<CacheEntry> {
  if (!isTauri()) {
    return {
      id,
      cacheType,
      content,
      inputHash,
      createdAt: Date.now() / 1000,
      lastAccessed: Date.now() / 1000,
      accessCount: 1,
      sizeBytes: content.length,
      ttlSeconds,
      projectId: projectId || null,
      tags
    }
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<CacheEntry>('cache_store', {
    cacheType,
    id,
    content,
    inputHash,
    ttlSeconds,
    projectId: projectId || null,
    tags
  })
}

export async function cacheGet(
  cacheType: CacheType,
  id: string
): Promise<CacheEntry | null> {
  if (!isTauri()) return null
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<CacheEntry | null>('cache_get', { cacheType, id })
}

export async function cacheExists(
  cacheType: CacheType,
  id: string
): Promise<boolean> {
  if (!isTauri()) return false
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<boolean>('cache_exists', { cacheType, id })
}

export async function cacheRemove(
  cacheType: CacheType,
  id: string
): Promise<boolean> {
  if (!isTauri()) return false
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<boolean>('cache_remove', { cacheType, id })
}

export async function cacheClearType(cacheType: CacheType): Promise<number> {
  if (!isTauri()) return 0
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<number>('cache_clear_type', { cacheType })
}

export async function cacheClearAll(): Promise<number> {
  if (!isTauri()) return 0
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<number>('cache_clear_all')
}

export async function cacheCleanupExpired(): Promise<number> {
  if (!isTauri()) return 0
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<number>('cache_cleanup_expired')
}

export async function cacheGetStats(): Promise<CacheStats> {
  if (!isTauri()) {
    return {
      totalEntries: 0,
      totalSizeBytes: 0,
      entriesByType: {},
      sizeByType: {},
      hitCount: 0,
      missCount: 0,
      cacheDirectory: ''
    }
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<CacheStats>('cache_get_stats')
}

export async function cacheList(cacheType: CacheType): Promise<CacheEntryInfo[]> {
  if (!isTauri()) return []
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<CacheEntryInfo[]>('cache_list', { cacheType })
}

// Convenience functions for specific cache types

export async function cacheGeneration(
  promptHash: string,
  generatedText: string,
  projectId?: string,
  modelName?: string
): Promise<CacheEntry> {
  if (!isTauri()) {
    return {
      id: `gen_${promptHash}`,
      cacheType: 'generation',
      content: generatedText,
      inputHash: promptHash,
      createdAt: Date.now() / 1000,
      lastAccessed: Date.now() / 1000,
      accessCount: 1,
      sizeBytes: generatedText.length,
      ttlSeconds: 604800, // 7 days
      projectId: projectId || null,
      tags: modelName ? ['llm', modelName] : ['llm']
    }
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<CacheEntry>('cache_generation', {
    promptHash,
    generatedText,
    projectId: projectId || null,
    modelName: modelName || null
  })
}

export async function findCachedGeneration(promptHash: string): Promise<CacheEntry | null> {
  if (!isTauri()) return null
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<CacheEntry | null>('find_cached_generation', { inputHash: promptHash })
}

export async function cacheDbQuery(
  queryHash: string,
  resultJson: string,
  ttlSeconds: number = 3600 // 1 hour default
): Promise<CacheEntry> {
  if (!isTauri()) {
    return {
      id: `dbq_${queryHash}`,
      cacheType: 'dbquery',
      content: resultJson,
      inputHash: queryHash,
      createdAt: Date.now() / 1000,
      lastAccessed: Date.now() / 1000,
      accessCount: 1,
      sizeBytes: resultJson.length,
      ttlSeconds,
      projectId: null,
      tags: ['db_query']
    }
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<CacheEntry>('cache_db_query', { queryHash, resultJson, ttlSeconds })
}

export async function findCachedDbQuery(queryHash: string): Promise<CacheEntry | null> {
  if (!isTauri()) return null
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<CacheEntry | null>('find_cached_db_query', { queryHash })
}

export async function cacheEmbedding(
  textHash: string,
  embeddingJson: string,
  modelName: string
): Promise<CacheEntry> {
  if (!isTauri()) {
    return {
      id: `emb_${textHash}`,
      cacheType: 'embedding',
      content: embeddingJson,
      inputHash: textHash,
      createdAt: Date.now() / 1000,
      lastAccessed: Date.now() / 1000,
      accessCount: 1,
      sizeBytes: embeddingJson.length,
      ttlSeconds: 2592000, // 30 days
      projectId: null,
      tags: ['embedding', modelName]
    }
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<CacheEntry>('cache_embedding', { textHash, embeddingJson, modelName })
}

export async function findCachedEmbedding(textHash: string): Promise<CacheEntry | null> {
  if (!isTauri()) return null
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<CacheEntry | null>('find_cached_embedding', { textHash })
}

export async function getCacheDirectory(): Promise<string> {
  if (!isTauri()) return ''
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<string>('get_cache_directory')
}

export async function getCacheSize(): Promise<{ entries: number; bytes: number }> {
  if (!isTauri()) return { entries: 0, bytes: 0 }
  const { invoke } = await import('@tauri-apps/api/core')
  const [entries, bytes] = await invoke<[number, number]>('get_cache_size')
  return { entries, bytes }
}

export async function enforceCacheLimits(): Promise<number> {
  if (!isTauri()) return 0
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke<number>('enforce_cache_limits')
}

// ============================================================================
// Session Cache (localStorage based - works without Tauri)
// ============================================================================

const SESSION_CACHE_KEY = 'nexastory_session_cache'

export interface SessionCacheData {
  lastProjectId: string | null
  lastChapterId: string | null
  lastView: string
  sidebarCollapsed: boolean
  editorFontSize: number
  theme: 'light' | 'dark'
  recentProjects: string[]
  lastBackupTime: string | null
  customSettings: Record<string, unknown>
}

const DEFAULT_SESSION_CACHE: SessionCacheData = {
  lastProjectId: null,
  lastChapterId: null,
  lastView: 'projects',
  sidebarCollapsed: false,
  editorFontSize: 16,
  theme: 'dark',
  recentProjects: [],
  lastBackupTime: null,
  customSettings: {}
}

export function getSessionCache(): SessionCacheData {
  if (typeof window === 'undefined') return DEFAULT_SESSION_CACHE
  try {
    const stored = localStorage.getItem(SESSION_CACHE_KEY)
    if (stored) {
      return { ...DEFAULT_SESSION_CACHE, ...JSON.parse(stored) }
    }
  } catch {
    // Ignore parse errors
  }
  return DEFAULT_SESSION_CACHE
}

export function setSessionCache(data: Partial<SessionCacheData>): void {
  if (typeof window === 'undefined') return
  try {
    const current = getSessionCache()
    const updated = { ...current, ...data }
    localStorage.setItem(SESSION_CACHE_KEY, JSON.stringify(updated))
  } catch {
    // Ignore storage errors
  }
}

export function clearSessionCache(): void {
  if (typeof window === 'undefined') return
  try {
    localStorage.removeItem(SESSION_CACHE_KEY)
  } catch {
    // Ignore storage errors
  }
}

export function addRecentProject(projectId: string): void {
  const cache = getSessionCache()
  const recent = [projectId, ...cache.recentProjects.filter(id => id !== projectId)].slice(0, 10)
  setSessionCache({ recentProjects: recent })
}
