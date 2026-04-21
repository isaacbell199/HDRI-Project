import { create } from 'zustand'
import { persist } from 'zustand/middleware'

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
  chapters: Chapter[]
  characters: Character[]
  locations: Location[]
  loreNotes: LoreNote[]
  settings: ProjectSettings | null
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

// Custom Style Types
export interface CustomStyle {
  id: string
  name: string
  description: string
  createdAt: string
}

export interface CustomPrompt {
  id: string
  name: string
  positivePrompt: string
  negativePrompt: string
  createdAt: string
}

export interface CustomTheme {
  id: string
  name: string
  description: string
  createdAt: string
}

export interface CustomCategory {
  id: string
  name: string
  description: string
  createdAt: string
}

// Generation Presets - Simplified settings for different writing styles
export interface GenerationPreset {
  id: string
  name: string
  description: string
  temperature: number
  topP: number
  topK: number
  minP: number
  repeatPenalty: number
  frequencyPenalty: number
  presencePenalty: number
  maxTokens: number
  isCustom?: boolean // Flag to identify custom presets
  createdAt?: string // For custom presets
}

// Built-in generation presets
export const GENERATION_PRESETS: GenerationPreset[] = [
  {
    id: 'classic-story',
    name: 'Classic Story',
    description: 'Balanced creativity for general storytelling',
    temperature: 0.75,
    topP: 0.92,
    topK: 50,
    minP: 0.03,
    repeatPenalty: 1.12,
    frequencyPenalty: 0.4,
    presencePenalty: 0.3,
    maxTokens: 400,
  },
  {
    id: 'dialogue',
    name: 'Natural Dialogue',
    description: 'Realistic and varied character conversations',
    temperature: 0.65,
    topP: 0.88,
    topK: 40,
    minP: 0.05,
    repeatPenalty: 1.15,
    frequencyPenalty: 0.5,
    presencePenalty: 0.4,
    maxTokens: 300,
  },
  {
    id: 'action',
    name: 'Action Scene',
    description: 'Dynamic and intense action sequences',
    temperature: 0.85,
    topP: 0.92,
    topK: 60,
    minP: 0.02,
    repeatPenalty: 1.12,
    frequencyPenalty: 0.35,
    presencePenalty: 0.25,
    maxTokens: 500,
  },
  {
    id: 'poetic',
    name: 'Poetic / Literary',
    description: 'Rich, evocative literary style',
    temperature: 0.88,
    topP: 0.95,
    topK: 50,
    minP: 0.02,
    repeatPenalty: 1.08,
    frequencyPenalty: 0.3,
    presencePenalty: 0.25,
    maxTokens: 400,
  },
  {
    id: 'precise',
    name: 'Precise & Coherent',
    description: 'More predictable and consistent output',
    temperature: 0.5,
    topP: 0.85,
    topK: 30,
    minP: 0.05,
    repeatPenalty: 1.18,
    frequencyPenalty: 0.5,
    presencePenalty: 0.4,
    maxTokens: 350,
  },
]

export interface CustomTone {
  id: string
  name: string
  description: string
  createdAt: string
}

// Custom Creative Options (created in Wizard)
export interface CreativeOption {
  id: string
  name: string
  description: string
  isCustom?: boolean
}

// Model info for global store
export interface ModelInfo {
  id: string
  name: string
  path: string
  sizeMb: number
  parameters: string
  quantization: string
  contextLength: number
  filename: string
  systemPrompt?: string // Custom system prompt to guide the model
}

// ============================================================================
// Persisted State Type
// ============================================================================

/// Type for the state that is persisted to localStorage
/// This ensures type safety for the partialize function
export interface PersistedState {
  // Generation parameters
  temperature: number
  topP: number
  topK: number
  minP: number
  repeatPenalty: number
  maxTokens: number
  // M10 fix: Include frequencyPenalty and presencePenalty in PersistedState
  frequencyPenalty: number
  presencePenalty: number
  // UI state
  sidebarCollapsed: boolean
  rightPanelOpen: boolean
  isDarkMode: boolean
  fontSize: 'sm' | 'md' | 'lg' | 'xl'
  // Generation options
  rewriteStyle: string
  tone: string
  pov: string
  // Custom creative content
  customStyles: CustomStyle[]
  customPrompts: CustomPrompt[]
  customThemes: CustomTheme[]
  customCategories: CustomCategory[]
  customTones: CustomTone[]
  // Selected IDs
  selectedStyleId: string | null
  selectedPromptId: string | null
  selectedThemeId: string | null
  selectedCategoryId: string | null
  selectedToneId: string | null
  // Dynamic sampling
  dynamicSamplingEnabled: boolean
  dynamicSamplingMinPAction: number
  dynamicSamplingMinPSensation: number
  dynamicSamplingTransitionRate: number
  // Language Mode (input -> output)
  languageMode: 'fr-fr' | 'fr-en' | 'en-fr' | 'en-en'
  // Presets
  selectedPresetId: string
  customPresets: GenerationPreset[]
  // LLM Settings (persisted for user convenience)
  cpuThreads: number
  gpuLayers: number
  useGpu: boolean
  batchSize: number
  contextLength: number
  useMmap: boolean
  useMlock: boolean
  flashAttention: boolean
  cacheType: string
  // Project Creative Settings
  projectCategory: string
  projectTone: string
  projectWritingStyle: string
  projectTheme: string
  projectPreset: string
  // Custom Creative Options
  projectCustomCategory: CreativeOption | null
  projectCustomTone: CreativeOption | null
  projectCustomStyle: CreativeOption | null
  projectCustomTheme: CreativeOption | null
}

// ============================================================================
// Store State Interface
// ============================================================================

// Duo Model Config (Speculative Decoding)
export interface DuoModelConfig {
  enabled: boolean
  draftModelPath: string | null
  draftTokens: number
  acceptanceThreshold: number
  sharedTokenizer: boolean
}

// Dynamic Sampling Config (Adaptive Parameters)
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

// View types
export type ViewType = 'projects' | 'editor' | 'world' | 'models' | 'settings'
export type WorldViewTab = 'characters' | 'locations' | 'lore'

// Store state
interface AppState {
  // Navigation
  currentView: ViewType
  setCurrentView: (view: ViewType) => void
  
  // Project
  currentProject: Project | null
  setCurrentProject: (project: Project | null) => void
  
  // Chapter
  currentChapter: Chapter | null
  setCurrentChapter: (chapter: Chapter | null) => void
  
  // World View Tab
  worldViewTab: WorldViewTab
  setWorldViewTab: (tab: WorldViewTab) => void
  
  // Editor
  editorContent: string
  setEditorContent: (content: string) => void
  isSaving: boolean
  setIsSaving: (saving: boolean) => void
  lastSaved: Date | null
  setLastSaved: (date: Date | null) => void
  
  // AI Generation
  generationMode: string
  setGenerationMode: (mode: string) => void
  
  // Generation Options
  rewriteStyle: string
  setRewriteStyle: (style: string) => void
  tone: string
  setTone: (tone: string) => void
  pov: string
  setPov: (pov: string) => void
  selectedCharacterId: string | null
  setSelectedCharacterId: (id: string | null) => void
  isGenerating: boolean
  setIsGenerating: (generating: boolean) => void
  generatedText: string
  setGeneratedText: (text: string) => void
  
  // Generation Parameters
  temperature: number
  setTemperature: (temp: number) => void
  topP: number
  setTopP: (p: number) => void
  topK: number
  setTopK: (k: number) => void
  minP: number
  setMinP: (minP: number) => void
  repeatPenalty: number
  setRepeatPenalty: (penalty: number) => void
  maxTokens: number
  setMaxTokens: (tokens: number) => void
  frequencyPenalty: number
  setFrequencyPenalty: (penalty: number) => void
  presencePenalty: number
  setPresencePenalty: (penalty: number) => void
  stopSequences: string
  setStopSequences: (sequences: string) => void
  
  // Generation Preset
  selectedPresetId: string
  setSelectedPresetId: (id: string) => void
  applyPreset: (preset: GenerationPreset) => void
  restorePresetFromId: (presetId: string) => void
  
  // Custom Generation Presets
  customPresets: GenerationPreset[]
  addCustomPreset: (preset: GenerationPreset) => void
  updateCustomPreset: (id: string, preset: Partial<GenerationPreset>) => void
  removeCustomPreset: (id: string) => void
  duplicatePreset: (preset: GenerationPreset) => void
  
  // Selected text for AI
  selectedText: string
  setSelectedText: (text: string) => void
  
  // Sidebar
  sidebarCollapsed: boolean
  setSidebarCollapsed: (collapsed: boolean) => void
  
  // Right Panel
  rightPanelOpen: boolean
  setRightPanelOpen: (open: boolean) => void
  
  // Theme
  isDarkMode: boolean
  setIsDarkMode: (isDark: boolean) => void
  
  // Font Size
  fontSize: 'sm' | 'md' | 'lg' | 'xl'
  setFontSize: (size: AppState['fontSize']) => void
  
  // LLM Settings
  cpuThreads: number
  setCpuThreads: (threads: number) => void
  batchSize: number
  setBatchSize: (size: number) => void
  contextLength: number
  setContextLength: (length: number) => void
  gpuLayers: number
  setGpuLayers: (layers: number) => void
  useGpu: boolean
  setUseGpu: (use: boolean) => void
  useMmap: boolean
  setUseMmap: (use: boolean) => void
  useMlock: boolean
  setUseMlock: (use: boolean) => void
  flashAttention: boolean
  setFlashAttention: (use: boolean) => void
  cacheType: string
  setCacheType: (type: string) => void
  modelPath: string | null
  setModelPath: (path: string | null) => void
  isModelLoaded: boolean
  setIsModelLoaded: (loaded: boolean) => void
  currentModelSystemPrompt: string | null  // System prompt for currently loaded model
  setCurrentModelSystemPrompt: (prompt: string | null) => void
  
  // Custom Writing Styles
  customStyles: CustomStyle[]
  addCustomStyle: (style: CustomStyle) => void
  removeCustomStyle: (id: string) => void
  selectedStyleId: string | null
  setSelectedStyleId: (id: string | null) => void
  
  // Custom Prompts
  customPrompts: CustomPrompt[]
  addCustomPrompt: (prompt: CustomPrompt) => void
  removeCustomPrompt: (id: string) => void
  selectedPromptId: string | null
  setSelectedPromptId: (id: string | null) => void
  
  // Custom Themes
  customThemes: CustomTheme[]
  addCustomTheme: (theme: CustomTheme) => void
  removeCustomTheme: (id: string) => void
  selectedThemeId: string | null
  setSelectedThemeId: (id: string | null) => void
  
  // Custom Categories
  customCategories: CustomCategory[]
  addCustomCategory: (category: CustomCategory) => void
  removeCustomCategory: (id: string) => void
  selectedCategoryId: string | null
  setSelectedCategoryId: (id: string | null) => void
  
  // Custom Tones
  customTones: CustomTone[]
  addCustomTone: (tone: CustomTone) => void
  removeCustomTone: (id: string) => void
  selectedToneId: string | null
  setSelectedToneId: (id: string | null) => void
  
  // Models
  models: ModelInfo[]
  setModels: (models: ModelInfo[]) => void
  addModel: (model: ModelInfo) => void
  removeModel: (id: string) => void
  selectedModelId: string | null
  setSelectedModelId: (id: string | null) => void
  
  // Duo Model (Speculative Decoding)
  duoModelEnabled: boolean
  setDuoModelEnabled: (enabled: boolean) => void
  duoModelDraftPath: string | null
  setDuoModelDraftPath: (path: string | null) => void
  duoModelDraftTokens: number
  setDuoModelDraftTokens: (tokens: number) => void
  duoModelAcceptanceThreshold: number
  setDuoModelAcceptanceThreshold: (threshold: number) => void
  
  // Dynamic Sampling (Adaptive Parameters)
  dynamicSamplingEnabled: boolean
  setDynamicSamplingEnabled: (enabled: boolean) => void
  dynamicSamplingMinP: number
  setDynamicSamplingMinP: (minP: number) => void
  dynamicSamplingMinPAction: number
  setDynamicSamplingMinPAction: (minP: number) => void
  dynamicSamplingMinPSensation: number
  setDynamicSamplingMinPSensation: (minP: number) => void
  dynamicSamplingTransitionRate: number
  setDynamicSamplingTransitionRate: (rate: number) => void
  
  // Language Mode (input -> output): FR→FR, FR→EN, EN→FR, EN→EN
  languageMode: 'fr-fr' | 'fr-en' | 'en-fr' | 'en-en'
  setLanguageMode: (mode: 'fr-fr' | 'fr-en' | 'en-fr' | 'en-en') => void
  
  // Project Creative Settings (from Wizard)
  projectCategory: string
  setProjectCategory: (category: string) => void
  projectTone: string
  setProjectTone: (tone: string) => void
  projectWritingStyle: string
  setProjectWritingStyle: (style: string) => void
  projectTheme: string
  setProjectTheme: (theme: string) => void
  projectPreset: string
  setProjectPreset: (preset: string) => void
  
  // Custom Creative Options (created in Wizard)
  projectCustomCategory: CreativeOption | null
  setProjectCustomCategory: (option: CreativeOption | null) => void
  projectCustomTone: CreativeOption | null
  setProjectCustomTone: (option: CreativeOption | null) => void
  projectCustomStyle: CreativeOption | null
  setProjectCustomStyle: (option: CreativeOption | null) => void
  projectCustomTheme: CreativeOption | null
  setProjectCustomTheme: (option: CreativeOption | null) => void
}

export const useStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Navigation
      currentView: 'projects',
      setCurrentView: (view) => set({ currentView: view }),
      
      // Project
      currentProject: null,
      setCurrentProject: (project) => set({ currentProject: project, currentChapter: null }),
      
      // Chapter
      currentChapter: null,
      setCurrentChapter: (chapter) => set({ currentChapter: chapter }),
      
      // World View Tab
      worldViewTab: 'characters',
      setWorldViewTab: (tab) => set({ worldViewTab: tab }),
      
      // Editor
      editorContent: '',
      setEditorContent: (content) => set({ editorContent: content }),
      isSaving: false,
      setIsSaving: (saving) => set({ isSaving: saving }),
      lastSaved: null,
      setLastSaved: (date) => set({ lastSaved: date }),
      
      // AI Generation
      generationMode: 'none',
      setGenerationMode: (mode) => set({ generationMode: mode }),
      
      // Generation Options
      rewriteStyle: 'more-descriptive',
      setRewriteStyle: (style) => set({ rewriteStyle: style }),
      tone: 'suspenseful',
      setTone: (tone) => set({ tone: tone }),
      pov: 'first',
      setPov: (pov) => set({ pov: pov }),
      selectedCharacterId: null,
      setSelectedCharacterId: (id) => set({ selectedCharacterId: id }),
      isGenerating: false,
      setIsGenerating: (generating) => set({ isGenerating: generating }),
      generatedText: '',
      setGeneratedText: (text) => set({ generatedText: text }),
      
      // Generation Parameters
      temperature: 0.7,
      setTemperature: (temp) => set({ temperature: temp }),
      topP: 0.9,
      setTopP: (p) => set({ topP: p }),
      topK: 40,
      setTopK: (k) => set({ topK: k }),
      minP: 0.05,
      setMinP: (minP) => set({ minP }),
      repeatPenalty: 1.12,
      setRepeatPenalty: (penalty) => set({ repeatPenalty: penalty }),
      maxTokens: 500,
      setMaxTokens: (tokens) => set({ maxTokens: tokens }),
      frequencyPenalty: 0,
      setFrequencyPenalty: (penalty) => set({ frequencyPenalty: penalty }),
      presencePenalty: 0,
      setPresencePenalty: (penalty) => set({ presencePenalty: penalty }),
      stopSequences: '',
      setStopSequences: (sequences) => set({ stopSequences: sequences }),
      
      // Generation Preset
      selectedPresetId: 'classic-story',
      setSelectedPresetId: (id) => set({ selectedPresetId: id, projectPreset: id }),
      applyPreset: (preset) => set({
        selectedPresetId: preset.id,
        projectPreset: preset.id,
        temperature: preset.temperature,
        topP: preset.topP,
        topK: preset.topK,
        minP: preset.minP,
        repeatPenalty: preset.repeatPenalty,
        maxTokens: preset.maxTokens,
        frequencyPenalty: preset.frequencyPenalty,
        presencePenalty: preset.presencePenalty,
      }),
      // Restore preset from projectPreset (called when loading a project)
      restorePresetFromId: (presetId) => {
        if (!presetId || presetId === 'none') return
        const allPresets = [...GENERATION_PRESETS, ...get().customPresets]
        const preset = allPresets.find(p => p.id === presetId)
        if (preset) {
          set({
            selectedPresetId: preset.id,
            temperature: preset.temperature,
            topP: preset.topP,
            topK: preset.topK,
            minP: preset.minP,
            repeatPenalty: preset.repeatPenalty,
            maxTokens: preset.maxTokens,
            frequencyPenalty: preset.frequencyPenalty,
            presencePenalty: preset.presencePenalty,
          })
        }
      },
      
      // Custom Generation Presets
      customPresets: [],
      addCustomPreset: (preset) => set((state) => ({ 
        customPresets: [...state.customPresets, preset] 
      })),
      updateCustomPreset: (id, updates) => set((state) => ({ 
        customPresets: state.customPresets.map(p => p.id === id ? { ...p, ...updates } : p) 
      })),
      removeCustomPreset: (id) => set((state) => ({ 
        customPresets: state.customPresets.filter(p => p.id !== id) 
      })),
      duplicatePreset: (preset) => set((state) => ({
        customPresets: [...state.customPresets, {
          ...preset,
          id: `custom-${Date.now()}`,
          name: `${preset.name} (Copy)`,
          isCustom: true,
          createdAt: new Date().toISOString(),
        }]
      })),
      
      // Selected text
      selectedText: '',
      setSelectedText: (text) => set({ selectedText: text }),
      
      // Sidebar
      sidebarCollapsed: false,
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
      
      // Right Panel
      rightPanelOpen: true,
      setRightPanelOpen: (open) => set({ rightPanelOpen: open }),
      
      // Theme
      isDarkMode: false,
      setIsDarkMode: (isDark) => set({ isDarkMode: isDark }),
      
      // Font Size
      fontSize: 'md',
      setFontSize: (size) => set({ fontSize: size }),
      
      // LLM Settings
      cpuThreads: 4,
      setCpuThreads: (threads) => set({ cpuThreads: threads }),
      batchSize: 512,
      setBatchSize: (size) => set({ batchSize: size }),
      contextLength: 4096,
      setContextLength: (length) => set({ contextLength: length }),
      gpuLayers: 0,
      setGpuLayers: (layers) => set({ gpuLayers: layers }),
      useGpu: false,
      setUseGpu: (use) => set({ useGpu: use }),
      useMmap: true,
      setUseMmap: (use) => set({ useMmap: use }),
      useMlock: false,
      setUseMlock: (use) => set({ useMlock: use }),
      flashAttention: false,
      setFlashAttention: (use) => set({ flashAttention: use }),
      cacheType: 'f16',
      setCacheType: (type) => set({ cacheType: type }),
      modelPath: null,
      setModelPath: (path) => set({ modelPath: path }),
      isModelLoaded: false,
      setIsModelLoaded: (loaded) => set({ isModelLoaded: loaded }),
      currentModelSystemPrompt: null,
      setCurrentModelSystemPrompt: (prompt) => set({ currentModelSystemPrompt: prompt }),
      
      // Custom Writing Styles
      customStyles: [],
      addCustomStyle: (style) => set((state) => ({ 
        customStyles: [...state.customStyles, style] 
      })),
      removeCustomStyle: (id) => set((state) => ({ 
        customStyles: state.customStyles.filter(s => s.id !== id) 
      })),
      selectedStyleId: null,
      setSelectedStyleId: (id) => set({ selectedStyleId: id }),
      
      // Custom Prompts
      customPrompts: [],
      addCustomPrompt: (prompt) => set((state) => ({ 
        customPrompts: [...state.customPrompts, prompt] 
      })),
      removeCustomPrompt: (id) => set((state) => ({ 
        customPrompts: state.customPrompts.filter(p => p.id !== id) 
      })),
      selectedPromptId: null,
      setSelectedPromptId: (id) => set({ selectedPromptId: id }),
      
      // Custom Themes
      customThemes: [],
      addCustomTheme: (theme) => set((state) => ({ 
        customThemes: [...state.customThemes, theme] 
      })),
      removeCustomTheme: (id) => set((state) => ({ 
        customThemes: state.customThemes.filter(t => t.id !== id) 
      })),
      selectedThemeId: null,
      setSelectedThemeId: (id) => set({ selectedThemeId: id }),
      
      // Custom Categories
      customCategories: [],
      addCustomCategory: (category) => set((state) => ({ 
        customCategories: [...state.customCategories, category] 
      })),
      removeCustomCategory: (id) => set((state) => ({ 
        customCategories: state.customCategories.filter(c => c.id !== id) 
      })),
      selectedCategoryId: null,
      setSelectedCategoryId: (id) => set({ selectedCategoryId: id }),
      
      // Custom Tones
      customTones: [],
      addCustomTone: (tone) => set((state) => ({ 
        customTones: [...state.customTones, tone] 
      })),
      removeCustomTone: (id) => set((state) => ({ 
        customTones: state.customTones.filter(t => t.id !== id) 
      })),
      selectedToneId: null,
      setSelectedToneId: (id) => set({ selectedToneId: id }),
      
      // Models
      models: [],
      setModels: (models) => set({ models }),
      addModel: (model) => set((state) => ({ 
        models: [...state.models, model] 
      })),
      removeModel: (id) => set((state) => ({ 
        models: state.models.filter(m => m.id !== id) 
      })),
      selectedModelId: null,
      setSelectedModelId: (id) => set({ selectedModelId: id }),
      
      // Duo Model (Speculative Decoding)
      duoModelEnabled: false,
      setDuoModelEnabled: (enabled) => set({ duoModelEnabled: enabled }),
      duoModelDraftPath: null,
      setDuoModelDraftPath: (path) => set({ duoModelDraftPath: path }),
      duoModelDraftTokens: 4,
      setDuoModelDraftTokens: (tokens) => set({ duoModelDraftTokens: tokens }),
      duoModelAcceptanceThreshold: 0.9,
      setDuoModelAcceptanceThreshold: (threshold) => set({ duoModelAcceptanceThreshold: threshold }),
      
      // Dynamic Sampling (Adaptive Parameters)
      dynamicSamplingEnabled: false,
      setDynamicSamplingEnabled: (enabled) => set({ dynamicSamplingEnabled: enabled }),
      dynamicSamplingMinP: 0.05,
      setDynamicSamplingMinP: (minP) => set({ dynamicSamplingMinP: minP }),
      dynamicSamplingMinPAction: 0.08,
      setDynamicSamplingMinPAction: (minP) => set({ dynamicSamplingMinPAction: minP }),
      dynamicSamplingMinPSensation: 0.02,
      setDynamicSamplingMinPSensation: (minP) => set({ dynamicSamplingMinPSensation: minP }),
      dynamicSamplingTransitionRate: 0.05,
      setDynamicSamplingTransitionRate: (rate) => set({ dynamicSamplingTransitionRate: rate }),
      
      // Language Mode (input -> output): FR→FR, FR→EN, EN→FR, EN→EN
      // Default: EN→EN (English input, English output)
      languageMode: 'en-en',
      setLanguageMode: (mode) => set({ languageMode: mode }),
      
      // Project Creative Settings (from Wizard)
      projectCategory: 'none',
      setProjectCategory: (category) => set({ projectCategory: category }),
      projectTone: 'none',
      setProjectTone: (tone) => set({ projectTone: tone }),
      projectWritingStyle: 'none',
      setProjectWritingStyle: (style) => set({ projectWritingStyle: style }),
      projectTheme: 'none',
      setProjectTheme: (theme) => set({ projectTheme: theme }),
      projectPreset: 'none',
      setProjectPreset: (preset) => set({ projectPreset: preset }),
      
      // Custom Creative Options (created in Wizard)
      projectCustomCategory: null,
      setProjectCustomCategory: (option) => set({ projectCustomCategory: option }),
      projectCustomTone: null,
      setProjectCustomTone: (option) => set({ projectCustomTone: option }),
      projectCustomStyle: null,
      setProjectCustomStyle: (option) => set({ projectCustomStyle: option }),
      projectCustomTheme: null,
      setProjectCustomTheme: (option) => set({ projectCustomTheme: option }),
    }),
    {
      name: 'nexastory-desktop-storage',
      partialize: (state): PersistedState => ({
        temperature: state.temperature,
        topP: state.topP,
        topK: state.topK,
        minP: state.minP,
        repeatPenalty: state.repeatPenalty,
        maxTokens: state.maxTokens,
        // M10 fix: Include frequencyPenalty and presencePenalty in persistence
        frequencyPenalty: state.frequencyPenalty,
        presencePenalty: state.presencePenalty,
        sidebarCollapsed: state.sidebarCollapsed,
        rightPanelOpen: state.rightPanelOpen,
        isDarkMode: state.isDarkMode,
        fontSize: state.fontSize,
        rewriteStyle: state.rewriteStyle,
        tone: state.tone,
        pov: state.pov,
        customStyles: state.customStyles,
        customPrompts: state.customPrompts,
        customThemes: state.customThemes,
        customCategories: state.customCategories,
        customTones: state.customTones,
        selectedStyleId: state.selectedStyleId,
        selectedPromptId: state.selectedPromptId,
        selectedThemeId: state.selectedThemeId,
        selectedCategoryId: state.selectedCategoryId,
        selectedToneId: state.selectedToneId,
        // NOTE: models are NOT persisted here - they are fetched dynamically from the backend
        // by scanning the models directory on app startup. This ensures fresh model data
        // is always shown instead of stale localStorage data after app restart.
        // DO NOT persist model loading states - these must be fetched from backend on startup
        // selectedModelId: NOT persisted - backend decides
        // isModelLoaded: NOT persisted - backend decides
        // modelPath: NOT persisted - backend decides
        // duoModelEnabled: NOT persisted - backend decides
        // duoModelDraftPath: NOT persisted - backend decides
        dynamicSamplingEnabled: state.dynamicSamplingEnabled,
        dynamicSamplingMinPAction: state.dynamicSamplingMinPAction,
        dynamicSamplingMinPSensation: state.dynamicSamplingMinPSensation,
        dynamicSamplingTransitionRate: state.dynamicSamplingTransitionRate,
        languageMode: state.languageMode,
        selectedPresetId: state.selectedPresetId,
        customPresets: state.customPresets,
        // LLM Settings for user convenience - persisted so users don't have to reconfigure
        cpuThreads: state.cpuThreads,
        gpuLayers: state.gpuLayers,
        useGpu: state.useGpu,
        batchSize: state.batchSize,
        contextLength: state.contextLength,
        useMmap: state.useMmap,
        useMlock: state.useMlock,
        flashAttention: state.flashAttention,
        cacheType: state.cacheType,
        // Project Creative Settings (from Wizard) - persist for generation
        projectCategory: state.projectCategory,
        projectTone: state.projectTone,
        projectWritingStyle: state.projectWritingStyle,
        projectTheme: state.projectTheme,
        projectPreset: state.projectPreset,
        // Custom Creative Options (created in Wizard)
        projectCustomCategory: state.projectCustomCategory,
        projectCustomTone: state.projectCustomTone,
        projectCustomStyle: state.projectCustomStyle,
        projectCustomTheme: state.projectCustomTheme,
      }),
    }
  )
)
