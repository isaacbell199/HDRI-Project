'use client'

import { useState, useEffect, useCallback } from 'react'
import { useStore } from '@/lib/store'
import { cn, getErrorMessage, getErrorStack } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import {
  Brain,
  Cpu,
  HardDrive,
  FolderOpen,
  RefreshCw,
  Play,
  Square,
  Check,
  Zap,
  Gauge,
  MemoryStick,
  Settings2,
  Activity,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Sparkles,
  Server,
  Download,
  Upload,
  Monitor,
  PcCase,
  Trash2,
  ChevronDown,
  ChevronUp,
  Search,
  X,
  FileCode,
  FolderSync,
  Cog,
  Wand2,
  AlertTriangle,
  Bug,
  Copy,
  Layers,
  Save,
  MessageSquare,
  Rocket,
  Target,
  ExternalLink,
  FileDown
} from 'lucide-react'
import { toast } from 'sonner'
import { motion, AnimatePresence } from 'framer-motion'
import {
  isTauri,
  getHardwareInfo,
  getAvailableModels,
  loadModel as tauriLoadModel,
  unloadModel as tauriUnloadModel,
  selectModelFile,
  getModelInfo,
  getCpuOptimizations,
  loadDraftModel as tauriLoadDraftModel,
  unloadDraftModel as tauriUnloadDraftModel,
  setDuoModelEnabled as tauriSetDuoModelEnabled,
  getDuoModelStatus as tauriGetDuoModelStatus,
  getLLMSettings,
  updateLLMSettings,
  type HardwareInfo as TauriHardwareInfo,
  type ModelInfo as TauriModelInfo,
  type CpuOptimizations
} from '@/lib/tauri-api'

// ============================================================================
// Types
// ============================================================================

interface HardwareInfoData {
  cpuCores: number
  cpuThreads: number
  cpuModel: string
  totalMemoryGb: number
  availableMemoryGb: number
  hasGpu: boolean
  gpuName: string
  gpuMemoryGb: number
  gpuVendor: 'nvidia' | 'amd' | 'intel' | 'apple' | 'unknown'
  recommendedThreads: number
  recommendedBatchSize: number
  recommendedGpuLayers: number
  supportsMmap: boolean
  supportsMlock: boolean
}

interface ModelInfoData {
  id: string
  name: string
  path: string
  sizeMb: number
  parameters: string
  quantization: string
  contextLength: number
  filename: string
  systemPrompt?: string
}

interface HFModel {
  id: string
  name: string
  author: string
  downloads: number
  tags: string[]
  size?: string
  ggufFiles?: string[]
  pipelineTag?: string
}

interface ErrorState {
  hasError: boolean
  message: string
  details: string
  timestamp: Date
  stack?: string
}

type GenerationMode = 'single' | 'duo'

// ============================================================================
// Component
// ============================================================================

export function ModelsView() {
  const store = useStore()

  // ========================================
  // Core State
  // ========================================

  const [activeTab, setActiveTab] = useState('models')
  const [isLoading, setIsLoading] = useState(false)
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [isDownloading, setIsDownloading] = useState(false)
  const [downloadProgress, setDownloadProgress] = useState(0)

  // Generation Mode - Single or Duo Model
  const [generationMode, setGenerationMode] = useState<GenerationMode>('single')

  // Error State
  const [errorState, setErrorState] = useState<ErrorState>({
    hasError: false,
    message: '',
    details: '',
    timestamp: new Date()
  })
  const [showErrorStack, setShowErrorStack] = useState(false)

  const clearError = useCallback(() => {
    setErrorState({ hasError: false, message: '', details: '', timestamp: new Date() })
    setShowErrorStack(false)
  }, [])

  const setError = useCallback((message: string, error?: unknown) => {
    const errDetails = error ? getErrorMessage(error) : ''
    const errStack = error ? getErrorStack(error) : undefined
    console.error('[ModelsView] Error:', message, error)
    setErrorState({ hasError: true, message, details: errDetails, stack: errStack, timestamp: new Date() })
    toast.error(message, { description: errDetails || undefined })
  }, [])

  // ========================================
  // Hardware State
  // ========================================

  const [hardwareInfo, setHardwareInfo] = useState<HardwareInfoData>({
    cpuCores: 4, cpuThreads: 8, cpuModel: 'Detecting...', totalMemoryGb: 16, availableMemoryGb: 12,
    hasGpu: false, gpuName: 'Not detected', gpuMemoryGb: 0, gpuVendor: 'unknown',
    recommendedThreads: 4, recommendedBatchSize: 512, recommendedGpuLayers: 0,
    supportsMmap: true, supportsMlock: false
  })

  const [cpuOptimizations, setCpuOptimizations] = useState<CpuOptimizations>({
    hasAvx: false, hasAvx2: false, hasAvx512: false, hasFma: false, optimizationLevel: 'Detecting...'
  })

  // ========================================
  // Model State
  // ========================================

  const [addModelTab, setAddModelTab] = useState<'disk' | 'huggingface'>('disk')
  const [modelPath, setModelPath] = useState('')
  const [hfSearchQuery, setHfSearchQuery] = useState('')
  const [hfSearchResults, setHfSearchResults] = useState<HFModel[]>([])
  const [isSearching, setIsSearching] = useState(false)

  // System prompts
  const [modelSystemPrompts, setModelSystemPrompts] = useState<Record<string, string>>({})
  const [savingPromptFor, setSavingPromptFor] = useState<string | null>(null)
  const [expandedModelId, setExpandedModelId] = useState<string | null>(null)

  // Settings
  const [perfSettingsOpen, setPerfSettingsOpen] = useState(true)
  const [advancedSettingsOpen, setAdvancedSettingsOpen] = useState(false)

  // Local settings
  const [useGpu, setUseGpu] = useState(store.useGpu)
  const [gpuLayers, setGpuLayers] = useState(store.gpuLayers)
  const [cpuThreads, setCpuThreads] = useState(store.cpuThreads)
  const [batchSize, setBatchSize] = useState(store.batchSize)
  const [contextLength, setContextLength] = useState(store.contextLength)
  const [useMmap, setUseMmap] = useState(store.useMmap)
  const [useMlock, setUseMlock] = useState(store.useMlock)
  const [flashAttention, setFlashAttention] = useState(store.flashAttention)
  const [cacheType, setCacheType] = useState(store.cacheType)

  // Duo Model State
  const [targetModelPath, setTargetModelPath] = useState('')
  const [draftModelPath, setDraftModelPath] = useState('')
  const [isLoadingTargetModel, setIsLoadingTargetModel] = useState(false)
  const [isLoadingDraftModel, setIsLoadingDraftModel] = useState(false)
  const [duoModelStatus, setDuoModelStatus] = useState({
    enabled: false, mainModelLoaded: false, draftModelLoaded: false, ready: false,
    mainModelPath: null as string | null, draftModelPath: null as string | null
  })

  // Global store
  const models = store.models
  const setModels = store.setModels
  const selectedModelId = store.selectedModelId
  const setSelectedModelId = store.setSelectedModelId

  // ========================================
  // Hardware Detection
  // ========================================

  const detectHardware = async () => {
    try {
      if (isTauri()) {
        const info = await getHardwareInfo()
        setHardwareInfo({
          cpuCores: Math.floor(info.cpuThreads / 2) || 4, cpuThreads: info.cpuThreads || 8,
          cpuModel: 'Detected via Tauri', totalMemoryGb: info.totalMemoryGb || 16,
          availableMemoryGb: info.availableMemoryGb || 12, hasGpu: info.hasGpu || false,
          gpuName: info.gpuName || 'Not detected', gpuMemoryGb: info.gpuMemoryGb || 0,
          gpuVendor: 'unknown', recommendedThreads: info.recommendedThreads || 4,
          recommendedBatchSize: info.recommendedBatchSize || 512,
          recommendedGpuLayers: info.hasGpu ? 35 : 0, supportsMmap: true, supportsMlock: false
        })
        try {
          const cpuOpts = await getCpuOptimizations()
          setCpuOptimizations(cpuOpts)
        } catch (e) { console.warn('Could not detect CPU optimizations:', e) }
        toast.success('Hardware detected successfully')
      } else {
        const cores = navigator.hardwareConcurrency || 4
        setHardwareInfo({
          cpuCores: Math.floor(cores / 2), cpuThreads: cores, cpuModel: 'Browser detection (limited)',
          totalMemoryGb: 8, availableMemoryGb: 6, hasGpu: false, gpuName: 'Not detectable',
          gpuMemoryGb: 0, gpuVendor: 'unknown', recommendedThreads: Math.max(1, Math.floor(cores / 2)),
          recommendedBatchSize: 512, recommendedGpuLayers: 0, supportsMmap: true, supportsMlock: false
        })
        setCpuOptimizations({ hasAvx: true, hasAvx2: true, hasAvx512: false, hasFma: true, optimizationLevel: 'AVX2 (Excellent)' })
        toast.info('Running in browser mode')
      }
    } catch (error) {
      setError('Hardware detection failed', error)
    }
  }

  // ========================================
  // Model Scanning
  // ========================================

  const scanModelsFolder = async () => {
    try {
      if (isTauri()) {
        const availableModels = await getAvailableModels()
        const mappedModels: ModelInfoData[] = (availableModels || []).map(m => ({
          id: m.id || m.path, name: m.name || m.path.split('/').pop() || 'Unknown', path: m.path,
          sizeMb: m.sizeMb || 0, parameters: m.parameters || 'Unknown', quantization: m.quantization || 'Unknown',
          contextLength: m.contextLength || 4096, filename: m.path.split('/').pop() || ''
        }))
        setModels(mappedModels)
      } else { setModels([]) }
    } catch (error) {
      console.warn('[ModelsView] Model scan failed:', error)
      setModels([])
    }
  }

  // ========================================
  // Model System Prompt (localStorage for Tauri desktop)
  // ========================================

  const MODEL_PROMPTS_KEY = 'nexastory_model_system_prompts'

  const loadModelSystemPrompts = async () => {
    try {
      // Use localStorage for Tauri desktop app
      const stored = localStorage.getItem(MODEL_PROMPTS_KEY)
      if (stored) {
        const promptsMap: Record<string, string> = JSON.parse(stored)
        setModelSystemPrompts(promptsMap)
      }
    } catch (error) {
      console.warn('[ModelsView] Failed to load model system prompts:', error)
    }
  }

  const handleSaveSystemPrompt = async (model: ModelInfoData) => {
    const prompt = modelSystemPrompts[model.path] || ''
    setSavingPromptFor(model.id)
    try {
      // Save to localStorage for Tauri desktop app
      const updatedPrompts = { ...modelSystemPrompts, [model.path]: prompt }
      if (!prompt.trim()) {
        delete updatedPrompts[model.path]
      }
      localStorage.setItem(MODEL_PROMPTS_KEY, JSON.stringify(updatedPrompts))
      setModelSystemPrompts(updatedPrompts)
      toast.success(`System prompt saved for "${model.name}"`)
      // Update current model system prompt if this is the loaded model
      if (store.modelPath === model.path) {
        store.setCurrentModelSystemPrompt(prompt || null)
      }
    } catch (error) {
      setError('Failed to save system prompt', error)
    } finally { setSavingPromptFor(null) }
  }

  const updateModelSystemPrompt = (modelPath: string, prompt: string) => {
    setModelSystemPrompts(prev => ({ ...prev, [modelPath]: prompt }))
  }

  // ========================================
  // Mode Switching
  // ========================================

  const handleModeChange = async (mode: GenerationMode) => {
    // If switching modes while models are loaded, warn user
    if (store.isModelLoaded || duoModelStatus.draftModelLoaded) {
      // Unload everything first
      await handleUnloadAllModels()
    }
    setGenerationMode(mode)
    toast.success(`Switched to ${mode === 'single' ? 'Single Model' : 'Duo Model'} mode`)
  }

  const handleUnloadAllModels = async () => {
    try {
      if (isTauri()) {
        if (duoModelStatus.draftModelLoaded) await tauriUnloadDraftModel()
        if (store.isModelLoaded) await tauriUnloadModel()
      }
      store.setModelPath(null)
      store.setIsModelLoaded(false)
      store.setSelectedModelId(null)
      store.setDuoModelEnabled(false)
      store.setCurrentModelSystemPrompt(null)
      setTargetModelPath('')
      setDraftModelPath('')
      await refreshDuoModelStatus()
    } catch (error) {
      console.error('[ModelsView] Failed to unload all models:', error)
    }
  }

  // ========================================
  // Single Model Loading
  // ========================================

  const handleLoadModel = async (modelId: string) => {
    const model = models.find(m => m.id === modelId)
    if (!model) { setError('Model not found', `Model ID: ${modelId}`); return }

    setIsModelLoading(true)
    clearError()

    try {
      if (isTauri()) {
        // Unload draft model if in duo mode and loaded
        if (generationMode === 'duo' && duoModelStatus.draftModelLoaded) {
          await tauriUnloadDraftModel()
        }

        // Get current LLM settings and update with local GPU settings
        // This ensures the model is loaded with the correct GPU configuration
        const currentSettings = await getLLMSettings()
        const updatedSettings = {
          ...currentSettings,
          useGpu: useGpu,
          gpuLayers: gpuLayers,
          gpuDevice: null,
          cpuThreads: cpuThreads,
          batchSize: batchSize,
          contextLength: contextLength,
          useMmap: useMmap,
          useMlock: useMlock,
          flashAttention: flashAttention,
          cacheType: cacheType,
        }
        await updateLLMSettings(updatedSettings)

        await tauriLoadModel(model.path)
        store.setModelPath(model.path)
        store.setIsModelLoaded(true)
        store.setUseGpu(useGpu)
        store.setGpuLayers(gpuLayers)
        store.setCpuThreads(cpuThreads)
        store.setBatchSize(batchSize)
        store.setContextLength(contextLength)
        store.setUseMmap(useMmap)
        store.setUseMlock(useMlock)
        store.setFlashAttention(flashAttention)
        store.setCacheType(cacheType)
        store.setSelectedModelId(modelId)

        const modelSystemPrompt = modelSystemPrompts[model.path] || null
        store.setCurrentModelSystemPrompt(modelSystemPrompt)

        if (generationMode === 'duo') {
          setTargetModelPath(model.path)
        }

        await refreshDuoModelStatus()
        toast.success(`Model "${model.name}" loaded successfully${modelSystemPrompt ? ' (with custom system prompt)' : ''}`)
      } else {
        store.setModelPath(model.path)
        store.setIsModelLoaded(true)
        store.setSelectedModelId(modelId)
        const modelSystemPrompt = modelSystemPrompts[model.path] || null
        store.setCurrentModelSystemPrompt(modelSystemPrompt)
        if (generationMode === 'duo') setTargetModelPath(model.path)
        toast.success(`Model "${model.name}" loaded (demo mode)`)
      }
    } catch (error) {
      setError(`Failed to load model "${model.name}"`, error)
    } finally { setIsModelLoading(false) }
  }

  const handleUnloadModel = async () => {
    try {
      if (isTauri()) await tauriUnloadModel()
      store.setModelPath(null)
      store.setIsModelLoaded(false)
      store.setSelectedModelId(null)
      store.setDuoModelEnabled(false)
      store.setCurrentModelSystemPrompt(null)
      setTargetModelPath('')
      await refreshDuoModelStatus()
      toast.success('Model unloaded')
    } catch (error) { setError('Failed to unload model', error) }
  }

  // ========================================
  // Duo Model Handlers
  // ========================================

  const refreshDuoModelStatus = async () => {
    if (!isTauri()) return
    try {
      const status = await tauriGetDuoModelStatus()
      setDuoModelStatus({
        enabled: status.enabled || false, mainModelLoaded: status.mainModelLoaded || false,
        draftModelLoaded: status.draftModelLoaded || false, ready: status.ready || false,
        mainModelPath: status.mainModelPath || null, draftModelPath: status.draftModelPath || null
      })
    } catch (error) { console.error('[ModelsView] Failed to get duo model status:', error) }
  }

  // ========================================
  // Initialization
  // ========================================

  useEffect(() => {
    let mounted = true
    const initializeView = async () => {
      setIsLoading(true)
      clearError()
      // Don't reset model state if a model is already loaded (C7 fix)
      // Only refresh the model list, not the entire state
      if (!store.isModelLoaded) {
        store.setModelPath(null)
        store.setIsModelLoaded(false)
        store.setSelectedModelId(null)
        store.setDuoModelEnabled(false)
        store.setDuoModelDraftPath(null)
      }

      try {
        await detectHardware()
        if (mounted) {
          await scanModelsFolder()
          await loadModelSystemPrompts()
        }
        if (mounted && isTauri()) {
          await refreshDuoModelStatus()
        }
      } catch (error) {
        if (mounted) setError('Failed to initialize Models view', error)
      } finally {
        if (mounted) setIsLoading(false)
      }
    }
    initializeView()
    return () => { mounted = false }
  }, [])

  const handleSelectTargetModel = async () => {
    try {
      if (isTauri()) {
        const path = await selectModelFile('Select Target Model (Main GGUF)')
        if (path) setTargetModelPath(path)
      }
    } catch (error) { setError('Failed to select target model file', error) }
  }

  const handleSelectDraftModel = async () => {
    try {
      if (isTauri()) {
        const path = await selectModelFile('Select Draft Model (Smaller GGUF)')
        if (path) setDraftModelPath(path)
      }
    } catch (error) { setError('Failed to select draft model file', error) }
  }

  const handleLoadTargetModel = async () => {
    if (!targetModelPath.trim()) { setError('Please select a target model'); return }
    setIsLoadingTargetModel(true)
    clearError()

    try {
      if (isTauri()) {
        await tauriLoadModel(targetModelPath)
        store.setModelPath(targetModelPath)
        store.setIsModelLoaded(true)
        const matchingModel = models.find(m => m.path === targetModelPath)
        if (matchingModel) store.setSelectedModelId(matchingModel.id)
        else store.setSelectedModelId(null)
        await refreshDuoModelStatus()
        toast.success('Target model loaded successfully')
      }
    } catch (error) { setError('Failed to load target model', error) }
    finally { setIsLoadingTargetModel(false) }
  }

  const handleUnloadTargetModel = async () => {
    try {
      if (isTauri()) await tauriUnloadModel()
      store.setModelPath(null)
      store.setIsModelLoaded(false)
      store.setSelectedModelId(null)
      store.setDuoModelEnabled(false)
      setTargetModelPath('')
      toast.success('Target model unloaded')
      await refreshDuoModelStatus()
    } catch (error) { setError('Failed to unload target model', error) }
  }

  const handleLoadDraftModel = async () => {
    if (!draftModelPath.trim()) { setError('Please select a draft model'); return }
    if (!duoModelStatus.mainModelLoaded) { setError('Please load the target model first'); return }

    setIsLoadingDraftModel(true)
    clearError()

    try {
      if (isTauri()) {
        await tauriLoadDraftModel(draftModelPath)
        await refreshDuoModelStatus()
        // AUTO-ENABLE speculative decoding when both models are loaded
        await tauriSetDuoModelEnabled(true)
        store.setDuoModelEnabled(true)
        await refreshDuoModelStatus()
        toast.success('Draft model loaded - Speculative Decoding ACTIVE! 🚀')
      }
    } catch (error) { setError('Failed to load draft model', error) }
    finally { setIsLoadingDraftModel(false) }
  }

  const handleUnloadDraftModel = async () => {
    try {
      if (isTauri()) {
        await tauriUnloadDraftModel()
        // Also disable duo model mode in store
        store.setDuoModelEnabled(false)
        setDraftModelPath('')
        toast.success('Draft model unloaded - Speculative Decoding disabled')
        await refreshDuoModelStatus()
      }
    } catch (error) { setError('Failed to unload draft model', error) }
  }

  const handleToggleDuoModel = async (enabled: boolean) => {
    try {
      if (isTauri()) {
        await tauriSetDuoModelEnabled(enabled)
        store.setDuoModelEnabled(enabled)
        toast.success(enabled ? 'Speculative decoding enabled' : 'Speculative decoding disabled')
        await refreshDuoModelStatus()
      }
    } catch (error) { setError('Failed to toggle duo model mode', error) }
  }

  // Auto-enable duo model mode when both models become ready
  useEffect(() => {
    const autoEnableDuoModel = async () => {
      if (isTauri() && duoModelStatus.mainModelLoaded && duoModelStatus.draftModelLoaded && !duoModelStatus.enabled) {
        try {
          await tauriSetDuoModelEnabled(true)
          store.setDuoModelEnabled(true)
          await refreshDuoModelStatus()
          toast.success('Speculative Decoding automatically enabled! 🚀')
        } catch (error) {
          console.error('[ModelsView] Failed to auto-enable duo model:', error)
        }
      }
    }
    autoEnableDuoModel()
  }, [duoModelStatus.mainModelLoaded, duoModelStatus.draftModelLoaded, duoModelStatus.enabled])

  // ========================================
  // Add Model from Disk
  // ========================================

  const handleSelectModelFile = async () => {
    try {
      if (isTauri()) {
        const path = await selectModelFile('Select a GGUF model file')
        if (path) setModelPath(path)
      } else { toast.info('Enter the path to your GGUF file manually') }
    } catch (error) { setError('Failed to select model file', error) }
  }

  const handleAddFromDisk = async () => {
    if (!modelPath.trim()) { setError('Please enter or select a file path'); return }
    setIsModelLoading(true)
    clearError()

    try {
      if (isTauri()) {
        const info = await getModelInfo(modelPath)
        const newModel: ModelInfoData = {
          id: info.id || info.path, name: info.name || modelPath.split('/').pop() || 'Imported model',
          path: info.path, sizeMb: info.sizeMb || 0, parameters: info.parameters || 'Unknown',
          quantization: info.quantization || 'Unknown', contextLength: info.contextLength || 4096,
          filename: modelPath.split('/').pop() || ''
        }
        store.addModel(newModel)
        setModelPath('')
        toast.success('Model added successfully')
      } else {
        const newModel: ModelInfoData = {
          id: Date.now().toString(), name: modelPath.split('/').pop() || 'Imported model',
          path: modelPath, sizeMb: 0, parameters: 'Unknown', quantization: 'Unknown',
          contextLength: 4096, filename: modelPath.split('/').pop() || ''
        }
        store.addModel(newModel)
        setModelPath('')
        toast.success('Model added (demo mode)')
      }
    } catch (error) { setError('Failed to add model', error) }
    finally { setIsModelLoading(false) }
  }

  // ========================================
  // HuggingFace Search
  // ========================================

  const searchHuggingFace = async () => {
    if (!hfSearchQuery.trim()) { setError('Please enter a search term'); return }
    setIsSearching(true)
    clearError()

    try {
      const response = await fetch(
        `https://huggingface.co/api/models?search=${encodeURIComponent(hfSearchQuery.trim())}&filter=gguf&sort=downloads&direction=-1&limit=20`
      )

      if (!response.ok) {
        if (response.status === 429) {
          setError('HuggingFace API rate limit reached. Please wait a moment and try again.')
          return
        }
        throw new Error(`API request failed with status ${response.status}`)
      }

      const data = await response.json()

      if (!Array.isArray(data) || data.length === 0) {
        setHfSearchResults([])
        toast.info('No GGUF models found for your search')
        return
      }

      const results: HFModel[] = data.map((item: Record<string, unknown>) => {
        const ggufFiles: string[] = (item.siblings as Array<Record<string, string>>)
          ?.map((s) => s.rfilename)
          ?.filter((name: string) => name.endsWith('.gguf')) || []

        return {
          id: (item.id as string) || (item.modelId as string) || '',
          name: ((item.id as string) || (item.modelId as string) || '').split('/').pop() || (item.id as string),
          author: (item.author as string) || '',
          downloads: (item.downloads as number) || 0,
          tags: (item.tags as string[]) || [],
          ggufFiles,
          pipelineTag: (item.pipeline_tag as string) || undefined,
        }
      })

      setHfSearchResults(results)
      toast.success(`Found ${results.length} model${results.length !== 1 ? 's' : ''}`)
    } catch (error) {
      if (error instanceof TypeError && error.message === 'Failed to fetch') {
        setError('Network error: Unable to reach HuggingFace. Check your internet connection.', error)
      } else {
        setError('Search failed. Please try again.', error)
      }
    } finally { setIsSearching(false) }
  }

  const downloadFromHuggingFace = async (modelId: string) => {
    clearError()

    try {
      const url = `https://huggingface.co/${modelId}`
      // Try Tauri shell.open first, fallback to window.open
      if (isTauri()) {
        try {
          const { open } = await import('@tauri-apps/plugin-shell')
          await open(url)
        } catch {
          window.open(url, '_blank')
        }
      } else {
        window.open(url, '_blank')
      }
      toast.success('Opening HuggingFace model page in browser', {
        description: 'Download the GGUF file manually and place it in your models directory, then add it via "From Disk".'
      })
    } catch (error) {
      setError('Failed to open model page', error)
    }
  }

  // ========================================
  // Settings
  // ========================================

  const applySettings = () => {
    try {
      store.setUseGpu(useGpu); store.setGpuLayers(gpuLayers); store.setCpuThreads(cpuThreads)
      store.setBatchSize(batchSize); store.setContextLength(contextLength); store.setUseMmap(useMmap)
      store.setUseMlock(useMlock); store.setFlashAttention(flashAttention); store.setCacheType(cacheType)
      toast.success('Settings applied')
    } catch (error) { setError('Failed to apply settings', error) }
  }

  const autoOptimize = () => {
    try {
      setCpuThreads(hardwareInfo.recommendedThreads); setBatchSize(hardwareInfo.recommendedBatchSize)
      setGpuLayers(hardwareInfo.recommendedGpuLayers); setUseGpu(hardwareInfo.hasGpu)
      toast.success('Settings auto-optimized')
    } catch (error) { setError('Auto-optimize failed', error) }
  }

  // ========================================
  // Helpers
  // ========================================

  const formatSize = (mb: number): string => mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`

  const formatDownloads = (count: number): string => {
    if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`
    if (count >= 1_000) return `${(count / 1_000).toFixed(count >= 100_000 ? 0 : 1)}K`
    return count.toString()
  }

  const getGpuColor = () => {
    switch (hardwareInfo.gpuVendor) {
      case 'nvidia': return 'text-emerald-500'
      case 'amd': return 'text-red-500'
      case 'intel': return 'text-sky-500'
      case 'apple': return 'text-gray-400'
      default: return 'text-muted-foreground'
    }
  }

  const copyErrorToClipboard = () => {
    const errorText = `[ModelsView Error - ${errorState.timestamp.toISOString()}]\nMessage: ${errorState.message}\nDetails: ${errorState.details}\n${errorState.stack ? `Stack:\n${errorState.stack}` : ''}`
    navigator.clipboard.writeText(errorText)
    toast.success('Error copied to clipboard')
  }

  // Get model name by path
  const getModelNameByPath = (path: string | null) => {
    if (!path) return 'Unknown'
    return models.find(m => m.path === path)?.name || path.split('/').pop() || 'Unknown'
  }

  // ========================================
  // Render
  // ========================================

  return (
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <header className="h-12 border-b border-border/50 flex items-center justify-between px-4 bg-gradient-to-r from-background to-muted/20 shrink-0">
        <div className="flex items-center gap-2">
          <Brain className="h-4 w-4 text-rose-500" />
          <h1 className="text-base font-semibold">Model Management</h1>
        </div>
        <div className="flex items-center gap-2">
          {/* Status Badge */}
          {store.isModelLoaded && (
            <Badge className={cn(
              "gap-1 text-xs",
              duoModelStatus.enabled ? "bg-purple-500 text-white" : "bg-emerald-500 text-white"
            )}>
              {duoModelStatus.enabled ? (
                <>
                  <Layers className="h-3 w-3" />
                  Duo Active
                </>
              ) : (
                <>
                  <CheckCircle2 className="h-3 w-3" />
                  Model Active
                </>
              )}
            </Badge>
          )}
          <Button variant="outline" size="sm" onClick={async () => {
            await scanModelsFolder()
            await loadModelSystemPrompts()
            detectHardware()
          }}>
            <RefreshCw className={cn("h-3.5 w-3.5", isLoading && "animate-spin")} />
          </Button>
        </div>
      </header>

      {/* Content */}
      <ScrollArea className="flex-1">
        <div className="p-4 max-w-5xl mx-auto space-y-4">

          {/* Error Alert */}
          <AnimatePresence>
            {errorState.hasError && (
              <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                <Alert variant="destructive" className="border-red-500/50 bg-red-500/10">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle className="flex items-center justify-between text-sm">
                    <span>Error</span>
                    <div className="flex gap-1">
                      <Button variant="ghost" size="sm" onClick={copyErrorToClipboard} className="h-6 px-2 text-xs"><Copy className="h-3 w-3 mr-1" />Copy</Button>
                      <Button variant="ghost" size="sm" onClick={clearError} className="h-6 px-2"><X className="h-3 w-3" /></Button>
                    </div>
                  </AlertTitle>
                  <AlertDescription className="text-xs mt-1">
                    <p className="font-medium">{errorState.message}</p>
                    {errorState.details && <p className="text-red-300 mt-1">{errorState.details}</p>}
                    {/* Collapsible Stack Trace Section */}
                    {errorState.stack && (
                      <div className="mt-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setShowErrorStack(!showErrorStack)}
                          className="h-6 px-2 text-xs text-red-400 hover:text-red-300"
                        >
                          <Bug className="h-3 w-3 mr-1" />
                          {showErrorStack ? 'Hide Stack Trace' : 'Show Stack Trace'}
                          {showErrorStack ? <ChevronUp className="h-3 w-3 ml-1" /> : <ChevronDown className="h-3 w-3 ml-1" />}
                        </Button>
                        {showErrorStack && (
                          <pre className="mt-2 p-2 bg-red-950/50 rounded text-[10px] text-red-200 overflow-x-auto max-h-32 overflow-y-auto font-mono whitespace-pre-wrap break-all">
                            {errorState.stack}
                          </pre>
                        )}
                      </div>
                    )}
                  </AlertDescription>
                </Alert>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Tabs */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
            <TabsList className="grid w-full grid-cols-3 bg-muted/30 p-0.5 h-9">
              <TabsTrigger value="models" className="gap-1.5 data-[state=active]:bg-background h-8 text-xs">
                <Brain className="h-3.5 w-3.5" />Models
              </TabsTrigger>
              <TabsTrigger value="settings" className="gap-1.5 data-[state=active]:bg-background h-8 text-xs">
                <Settings2 className="h-3.5 w-3.5" />Config
              </TabsTrigger>
              <TabsTrigger value="hardware" className="gap-1.5 data-[state=active]:bg-background h-8 text-xs">
                <PcCase className="h-3.5 w-3.5" />Hardware
              </TabsTrigger>
            </TabsList>

            {/* ==================== MODELS TAB ==================== */}
            <TabsContent value="models" className="space-y-4">

              {/* ========== MODE SELECTOR ========== */}
              <Card className="border-border/50">
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Cog className="h-4 w-4 text-muted-foreground" />
                    <Label className="text-sm font-medium">Generation Mode</Label>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    {/* Single Model Option */}
                    <button
                      onClick={() => handleModeChange('single')}
                      disabled={store.isModelLoaded && generationMode === 'duo'}
                      className={cn(
                        "relative flex flex-col items-center gap-2 p-4 rounded-xl border-2 transition-all text-center",
                        generationMode === 'single'
                          ? "border-emerald-500/50 bg-emerald-500/10"
                          : "border-border/50 bg-muted/20 hover:border-border hover:bg-muted/30",
                        store.isModelLoaded && generationMode === 'duo' && "opacity-50 cursor-not-allowed"
                      )}
                    >
                      {generationMode === 'single' && (
                        <div className="absolute top-2 right-2"><Check className="h-4 w-4 text-emerald-500" /></div>
                      )}
                      <div className={cn("p-2 rounded-lg", generationMode === 'single' ? "bg-emerald-500/20" : "bg-muted")}>
                        <Brain className={cn("h-5 w-5", generationMode === 'single' ? "text-emerald-500" : "text-muted-foreground")} />
                      </div>
                      <div>
                        <p className="font-medium text-sm">Single Model</p>
                        <p className="text-[10px] text-muted-foreground mt-0.5">Standard generation</p>
                      </div>
                    </button>

                    {/* Duo Model Option */}
                    <button
                      onClick={() => handleModeChange('duo')}
                      disabled={store.isModelLoaded && generationMode === 'single'}
                      className={cn(
                        "relative flex flex-col items-center gap-2 p-4 rounded-xl border-2 transition-all text-center",
                        generationMode === 'duo'
                          ? "border-purple-500/50 bg-purple-500/10"
                          : "border-border/50 bg-muted/20 hover:border-border hover:bg-muted/30",
                        store.isModelLoaded && generationMode === 'single' && "opacity-50 cursor-not-allowed"
                      )}
                    >
                      {generationMode === 'duo' && (
                        <div className="absolute top-2 right-2"><Check className="h-4 w-4 text-purple-500" /></div>
                      )}
                      <div className={cn("p-2 rounded-lg", generationMode === 'duo' ? "bg-purple-500/20" : "bg-muted")}>
                        <Layers className={cn("h-5 w-5", generationMode === 'duo' ? "text-purple-500" : "text-muted-foreground")} />
                      </div>
                      <div>
                        <p className="font-medium text-sm">Duo Model</p>
                        <p className="text-[10px] text-muted-foreground mt-0.5">Speculative decoding ~2x faster</p>
                      </div>
                    </button>
                  </div>
                  {(store.isModelLoaded || duoModelStatus.draftModelLoaded) && (
                    <p className="text-[10px] text-amber-500 mt-2 flex items-center gap-1">
                      <AlertCircle className="h-3 w-3" />
                      Unload models to switch mode
                    </p>
                  )}
                </CardContent>
              </Card>

              {/* ========== ACTIVE MODEL STATUS ========== */}
              <Card className="border-border/50">
                <CardHeader className="p-3 pb-0">
                  <CardTitle className="text-xs font-medium flex items-center gap-2">
                    <Zap className="h-3.5 w-3.5 text-amber-500" />
                    Active Status
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-3">
                  {generationMode === 'single' ? (
                    /* Single Model Status */
                    <div className="flex items-center justify-between p-3 rounded-lg bg-muted/30">
                      <div className="flex items-center gap-3">
                        <div className={cn("p-2 rounded-lg", store.isModelLoaded ? "bg-emerald-500/20" : "bg-muted")}>
                          <Brain className={cn("h-4 w-4", store.isModelLoaded ? "text-emerald-500" : "text-muted-foreground")} />
                        </div>
                        <div>
                          <p className="text-sm font-medium">{store.isModelLoaded ? getModelNameByPath(store.modelPath) : 'No model loaded'}</p>
                          <p className="text-[10px] text-muted-foreground">
                            {store.isModelLoaded ? (useGpu ? `GPU: ${gpuLayers} layers` : 'CPU mode') : 'Select a model below'}
                          </p>
                        </div>
                      </div>
                      {store.isModelLoaded && (
                        <Button variant="destructive" size="sm" onClick={handleUnloadModel} className="h-7 text-xs">
                          <Square className="h-3 w-3 mr-1" />Unload
                        </Button>
                      )}
                    </div>
                  ) : (
                    /* Duo Model Status */
                    <div className="space-y-2">
                      <div className="grid grid-cols-2 gap-2">
                        <div className={cn("p-2.5 rounded-lg border", duoModelStatus.mainModelLoaded ? "border-emerald-500/30 bg-emerald-500/5" : "border-border/30 bg-muted/20")}>
                          <div className="flex items-center gap-2">
                            <Target className={cn("h-3.5 w-3.5", duoModelStatus.mainModelLoaded ? "text-emerald-500" : "text-muted-foreground")} />
                            <span className="text-[10px] text-muted-foreground">Target</span>
                          </div>
                          <p className="text-xs font-medium mt-1 truncate">{duoModelStatus.mainModelLoaded ? getModelNameByPath(duoModelStatus.mainModelPath) : 'Not loaded'}</p>
                        </div>
                        <div className={cn("p-2.5 rounded-lg border", duoModelStatus.draftModelLoaded ? "border-cyan-500/30 bg-cyan-500/5" : "border-border/30 bg-muted/20")}>
                          <div className="flex items-center gap-2">
                            <Zap className={cn("h-3.5 w-3.5", duoModelStatus.draftModelLoaded ? "text-cyan-500" : "text-muted-foreground")} />
                            <span className="text-[10px] text-muted-foreground">Draft</span>
                          </div>
                          <p className="text-xs font-medium mt-1 truncate">{duoModelStatus.draftModelLoaded ? getModelNameByPath(duoModelStatus.draftModelPath) : 'Not loaded'}</p>
                        </div>
                      </div>
                      {duoModelStatus.mainModelLoaded && duoModelStatus.draftModelLoaded && (
                        <div className="flex items-center justify-between p-2 rounded-lg bg-purple-500/10 border border-purple-500/20">
                          <div className="flex items-center gap-2">
                            <Layers className="h-3.5 w-3.5 text-purple-500" />
                            <span className="text-xs">Speculative Decoding</span>
                          </div>
                          <Switch checked={duoModelStatus.enabled} onCheckedChange={handleToggleDuoModel} className="scale-75" />
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* ========== DUO MODEL SELECTORS (Only in Duo Mode) ========== */}
              {generationMode === 'duo' && (
                <Card className="border-border/50">
                  <CardHeader className="p-3 pb-0">
                    <CardTitle className="text-xs font-medium flex items-center gap-2">
                      <Layers className="h-3.5 w-3.5 text-purple-500" />
                      Duo Model Setup
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-3 space-y-3">
                    {/* Target Model Selector */}
                    <div className="space-y-1.5">
                      <Label className="text-[10px] text-muted-foreground flex items-center gap-1">
                        <Target className="h-3 w-3" />Target Model (Main)
                      </Label>
                      <div className="flex gap-1.5">
                        <Select value={targetModelPath} onValueChange={setTargetModelPath}>
                          <SelectTrigger className="h-8 text-xs"><SelectValue placeholder="Select from available models..." /></SelectTrigger>
                          <SelectContent>
                            {models.map(m => <SelectItem key={m.id} value={m.path} className="text-xs">{m.name} ({formatSize(m.sizeMb)})</SelectItem>)}
                          </SelectContent>
                        </Select>
                        <Button variant="outline" size="sm" onClick={handleSelectTargetModel} className="h-8 w-8 p-0"><FolderOpen className="h-3.5 w-3.5" /></Button>
                        {duoModelStatus.mainModelLoaded ? (
                          <Button variant="destructive" size="sm" onClick={handleUnloadTargetModel} className="h-8 text-xs"><Square className="h-3 w-3 mr-1" />Unload</Button>
                        ) : (
                          <Button size="sm" onClick={handleLoadTargetModel} disabled={!targetModelPath || isLoadingTargetModel} className="h-8 text-xs bg-gradient-to-r from-rose-600 to-pink-600 border-0">
                            {isLoadingTargetModel ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3 mr-1" />}Load
                          </Button>
                        )}
                      </div>
                    </div>

                    {/* Draft Model Selector */}
                    <div className="space-y-1.5">
                      <Label className="text-[10px] text-muted-foreground flex items-center gap-1">
                        <Zap className="h-3 w-3" />Draft Model (Smaller)
                      </Label>
                      <div className="flex gap-1.5">
                        <Select value={draftModelPath} onValueChange={setDraftModelPath} disabled={!duoModelStatus.mainModelLoaded}>
                          <SelectTrigger className="h-8 text-xs"><SelectValue placeholder={duoModelStatus.mainModelLoaded ? "Select draft model..." : "Load target first..."} /></SelectTrigger>
                          <SelectContent>
                            {models.filter(m => m.path !== targetModelPath).map(m => (
                              <SelectItem key={m.id} value={m.path} className="text-xs">{m.name} ({formatSize(m.sizeMb)})</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <Button variant="outline" size="sm" onClick={handleSelectDraftModel} disabled={!duoModelStatus.mainModelLoaded} className="h-8 w-8 p-0"><FolderOpen className="h-3.5 w-3.5" /></Button>
                        {duoModelStatus.draftModelLoaded ? (
                          <Button variant="destructive" size="sm" onClick={handleUnloadDraftModel} className="h-8 text-xs"><Square className="h-3 w-3 mr-1" />Unload</Button>
                        ) : (
                          <Button size="sm" onClick={handleLoadDraftModel} disabled={!draftModelPath || !duoModelStatus.mainModelLoaded || isLoadingDraftModel} className="h-8 text-xs bg-gradient-to-r from-cyan-600 to-teal-600 border-0">
                            {isLoadingDraftModel ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3 mr-1" />}Load
                          </Button>
                        )}
                      </div>
                      {!duoModelStatus.mainModelLoaded && (
                        <p className="text-[10px] text-amber-500 flex items-center gap-1">
                          <AlertCircle className="h-2.5 w-2.5" />Load target model first
                        </p>
                      )}
                    </div>

                    {/* Recommended Pairs Info */}
                    <div className="p-2 rounded-lg bg-purple-500/10 border border-purple-500/20">
                      <p className="text-[10px] font-medium text-purple-400 mb-1">Recommended Pairs</p>
                      <div className="grid grid-cols-3 gap-1 text-[9px] text-muted-foreground">
                        <span>Llama 3.2 3B + 1B</span>
                        <span>Qwen 2.5 7B + 1.5B</span>
                        <span>Mistral 7B + 0.5B</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* ========== ADD MODEL ========== */}
              <Collapsible>
                <CollapsibleTrigger asChild>
                  <Button variant="outline" className="w-full h-9 text-xs justify-between">
                    <span className="flex items-center gap-1.5"><Download className="h-3.5 w-3.5 text-sky-500" />Add Model</span>
                    <ChevronDown className="h-3.5 w-3.5" />
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <Card className="border-border/50 mt-2">
                    <CardContent className="p-3">
                      <Tabs value={addModelTab} onValueChange={(v) => setAddModelTab(v as 'disk' | 'huggingface')}>
                        <TabsList className="grid w-full grid-cols-2 h-8 mb-3">
                          <TabsTrigger value="disk" className="text-xs gap-1"><HardDrive className="h-3 w-3" />Disk</TabsTrigger>
                          <TabsTrigger value="huggingface" className="text-xs gap-1"><Download className="h-3 w-3" />HuggingFace</TabsTrigger>
                        </TabsList>

                        <TabsContent value="disk" className="space-y-2">
                          <div className="flex gap-1.5">
                            <Input value={modelPath} onChange={(e) => setModelPath(e.target.value)} placeholder="/path/to/model.gguf" className="h-8 text-xs flex-1" />
                            <Button variant="outline" size="sm" onClick={handleSelectModelFile} className="h-8 w-8 p-0"><FolderOpen className="h-3.5 w-3.5" /></Button>
                          </div>
                          <Button onClick={handleAddFromDisk} disabled={!modelPath.trim() || isModelLoading} className="w-full h-8 text-xs bg-gradient-to-r from-sky-600 to-cyan-600 border-0">
                            {isModelLoading ? <Loader2 className="h-3 w-3 mr-1.5 animate-spin" /> : <Upload className="h-3 w-3 mr-1.5" />}Add Model
                          </Button>
                        </TabsContent>

                        <TabsContent value="huggingface" className="space-y-2">
                          <div className="flex gap-1.5">
                            <Input value={hfSearchQuery} onChange={(e) => setHfSearchQuery(e.target.value)} placeholder="Search GGUF models (e.g. Llama, Mistral, Qwen)..." onKeyDown={(e) => e.key === 'Enter' && searchHuggingFace()} className="h-8 text-xs flex-1" />
                            <Button onClick={searchHuggingFace} disabled={isSearching} className="h-8 w-8 p-0">{isSearching ? <Loader2 className="h-3 w-3 animate-spin" /> : <Search className="h-3 w-3" />}</Button>
                          </div>
                          {hfSearchResults.length > 0 && (
                            <div className="space-y-1.5 max-h-64 overflow-y-auto">
                              {hfSearchResults.map(model => (
                                <div key={model.id} className="p-2.5 rounded-lg border border-border/30 hover:bg-muted/30 space-y-1.5">
                                  <div className="flex items-start justify-between gap-2">
                                    <div className="min-w-0 flex-1">
                                      <p className="text-xs font-medium truncate">{model.name}</p>
                                      <p className="text-[10px] text-muted-foreground">{model.author}</p>
                                    </div>
                                    <Button size="sm" variant="outline" onClick={() => downloadFromHuggingFace(model.id)} className="h-6 text-[10px] shrink-0 gap-1">
                                      <ExternalLink className="h-2.5 w-2.5" />Open
                                    </Button>
                                  </div>
                                  <div className="flex items-center gap-1.5 flex-wrap">
                                    {model.pipelineTag && (
                                      <Badge variant="secondary" className="text-[9px] h-4 px-1.5">{model.pipelineTag}</Badge>
                                    )}
                                    {model.ggufFiles && model.ggufFiles.length > 0 && (
                                      <Badge variant="outline" className="text-[9px] h-4 px-1.5 gap-0.5">
                                        <FileDown className="h-2.5 w-2.5" />{model.ggufFiles.length} GGUF
                                      </Badge>
                                    )}
                                    <span className="text-[9px] text-muted-foreground ml-auto">{formatDownloads(model.downloads)} downloads</span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                          {isSearching && hfSearchResults.length === 0 && (
                            <div className="text-center py-4 text-muted-foreground">
                              <Loader2 className="h-4 w-4 mx-auto animate-spin" />
                              <p className="text-[10px] mt-1">Searching HuggingFace...</p>
                            </div>
                          )}
                          {!isSearching && hfSearchResults.length === 0 && (
                            <div className="text-center py-3 text-muted-foreground">
                              <Download className="h-5 w-5 mx-auto mb-1 opacity-40" />
                              <p className="text-[10px]">Search for GGUF models on HuggingFace</p>
                            </div>
                          )}
                        </TabsContent>
                      </Tabs>
                    </CardContent>
                  </Card>
                </CollapsibleContent>
              </Collapsible>

              {/* ========== AVAILABLE MODELS ========== */}
              <Card className="border-border/50">
                <CardHeader className="p-3 pb-0">
                  <CardTitle className="text-xs font-medium flex items-center gap-2">
                    <FolderSync className="h-3.5 w-3.5 text-violet-500" />
                    Available Models
                    <Badge variant="outline" className="text-[10px] ml-auto">{models.length}</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-3">
                  {models.length === 0 ? (
                    <div className="text-center py-6 text-muted-foreground">
                      <HardDrive className="h-8 w-8 mx-auto mb-2 opacity-50" />
                      <p className="text-xs">No models found</p>
                    </div>
                  ) : (
                    <div className="space-y-1.5 max-h-[400px] overflow-y-auto">
                      {models.map((model) => {
                        const isExpanded = expandedModelId === model.id
                        const isActive = selectedModelId === model.id
                        const hasPrompt = modelSystemPrompts[model.path]?.trim()

                        return (
                          <div key={model.id} className={cn("rounded-lg border transition-all", isActive ? "border-emerald-500/30 bg-emerald-500/5" : "border-border/30 hover:border-border/50")}>
                            {/* Model Row */}
                            <div className="flex items-center gap-2 p-2">
                              <div className={cn("p-1.5 rounded-md", isActive ? "bg-emerald-500/20" : "bg-muted")}>
                                <Brain className={cn("h-3.5 w-3.5", isActive ? "text-emerald-500" : "text-muted-foreground")} />
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-1.5">
                                  <p className="text-xs font-medium truncate">{model.name}</p>
                                  {hasPrompt && <MessageSquare className="h-2.5 w-2.5 text-violet-500" />}
                                </div>
                                <p className="text-[10px] text-muted-foreground">
                                  {formatSize(model.sizeMb)} • {model.parameters} • {model.quantization}
                                </p>
                              </div>
                              {isActive && <Badge className="bg-emerald-500 text-white text-[9px] px-1.5 py-0">Active</Badge>}
                              <Button
                                size="sm"
                                variant={isActive ? "destructive" : "outline"}
                                onClick={() => isActive ? handleUnloadModel() : handleLoadModel(model.id)}
                                disabled={isModelLoading || (generationMode === 'duo' && !duoModelStatus.mainModelLoaded && !isActive && duoModelStatus.mainModelPath !== model.path)}
                                className="h-6 text-[10px] px-2"
                              >
                                {isModelLoading && !isActive ? <Loader2 className="h-2.5 w-2.5 animate-spin" /> : isActive ? <><Square className="h-2.5 w-2.5 mr-1" />Stop</> : <><Play className="h-2.5 w-2.5 mr-1" />Load</>}
                              </Button>
                              <Button variant="ghost" size="sm" onClick={() => setExpandedModelId(isExpanded ? null : model.id)} className="h-6 w-6 p-0">
                                {isExpanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                              </Button>
                            </div>

                            {/* Expanded System Prompt */}
                            <AnimatePresence>
                              {isExpanded && (
                                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                                  <div className="px-2 pb-2">
                                    <div className="p-2.5 rounded-lg bg-muted/30 space-y-2">
                                      <div className="flex items-center justify-between">
                                        <Label className="text-[10px] font-medium flex items-center gap-1">
                                          <MessageSquare className="h-3 w-3 text-violet-500" />System Prompt
                                        </Label>
                                        <Button size="sm" onClick={() => handleSaveSystemPrompt(model)} disabled={savingPromptFor === model.id} className="h-6 text-[10px] bg-gradient-to-r from-violet-600 to-purple-600 border-0">
                                          {savingPromptFor === model.id ? <Loader2 className="h-2.5 w-2.5 animate-spin" /> : <Save className="h-2.5 w-2.5 mr-1" />}Save
                                        </Button>
                                      </div>
                                      <textarea
                                        value={modelSystemPrompts[model.path] || ''}
                                        onChange={(e) => updateModelSystemPrompt(model.path, e.target.value)}
                                        placeholder="Guide the model's behavior..."
                                        className="w-full h-20 p-2 text-[11px] bg-background border border-border/50 rounded-md resize-none focus:outline-none focus:ring-1 focus:ring-primary/30"
                                      />
                                    </div>
                                  </div>
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* ==================== SETTINGS TAB ==================== */}
            <TabsContent value="settings" className="space-y-3">
              {/* Performance Settings */}
              <Collapsible open={perfSettingsOpen} onOpenChange={setPerfSettingsOpen}>
                <Card className="border-border/50">
                  <CollapsibleTrigger className="w-full">
                    <CardHeader className="p-3 pb-0">
                      <CardTitle className="text-xs font-medium flex items-center justify-between">
                        <span className="flex items-center gap-1.5"><Gauge className="h-3.5 w-3.5 text-amber-500" />Performance</span>
                        {perfSettingsOpen ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                      </CardTitle>
                    </CardHeader>
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <CardContent className="p-3 pt-2 space-y-3">
                      {/* GPU Settings */}
                      <div className="flex items-center justify-between">
                        <Label className="text-xs flex items-center gap-1.5"><Cpu className="h-3 w-3" />Use GPU</Label>
                        <Switch checked={useGpu} onCheckedChange={setUseGpu} disabled={!hardwareInfo.hasGpu} />
                      </div>
                      {useGpu && (
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs"><span>GPU Layers</span><span className="text-muted-foreground">{gpuLayers}</span></div>
                          <Slider value={[gpuLayers]} onValueChange={([v]) => setGpuLayers(v)} max={100} step={1} className="h-1.5" />
                        </div>
                      )}

                      <Separator className="my-2" />

                      {/* CPU Settings */}
                      <div className="grid grid-cols-2 gap-3">
                        <div className="space-y-1">
                          <Label className="text-xs">CPU Threads</Label>
                          <Input type="number" value={cpuThreads} onChange={(e) => setCpuThreads(parseInt(e.target.value) || 4)} className="h-7 text-xs" />
                        </div>
                        <div className="space-y-1">
                          <Label className="text-xs">Batch Size</Label>
                          <Input type="number" value={batchSize} onChange={(e) => setBatchSize(parseInt(e.target.value) || 512)} className="h-7 text-xs" />
                        </div>
                      </div>

                      <div className="space-y-1">
                        <Label className="text-xs">Context Length</Label>
                        <Input type="number" value={contextLength} onChange={(e) => setContextLength(parseInt(e.target.value) || 4096)} className="h-7 text-xs" />
                      </div>

                      <div className="flex gap-2">
                        <Button onClick={applySettings} className="flex-1 h-7 text-xs">Apply</Button>
                        <Button variant="outline" onClick={autoOptimize} className="h-7 text-xs"><Sparkles className="h-3 w-3 mr-1" />Auto</Button>
                      </div>
                    </CardContent>
                  </CollapsibleContent>
                </Card>
              </Collapsible>

              {/* Advanced Settings */}
              <Collapsible open={advancedSettingsOpen} onOpenChange={setAdvancedSettingsOpen}>
                <Card className="border-border/50">
                  <CollapsibleTrigger className="w-full">
                    <CardHeader className="p-3 pb-0">
                      <CardTitle className="text-xs font-medium flex items-center justify-between">
                        <span className="flex items-center gap-1.5"><Settings2 className="h-3.5 w-3.5 text-slate-500" />Advanced</span>
                        {advancedSettingsOpen ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                      </CardTitle>
                    </CardHeader>
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <CardContent className="p-3 pt-2 space-y-2">
                      <div className="grid grid-cols-2 gap-2">
                        <div className="flex items-center justify-between">
                          <Label className="text-xs">mmap</Label>
                          <Switch checked={useMmap} onCheckedChange={setUseMmap} className="scale-75" />
                        </div>
                        <div className="flex items-center justify-between">
                          <Label className="text-xs">mlock</Label>
                          <Switch checked={useMlock} onCheckedChange={setUseMlock} className="scale-75" />
                        </div>
                        <div className="flex items-center justify-between">
                          <Label className="text-xs">Flash Attn</Label>
                          <Switch checked={flashAttention} onCheckedChange={setFlashAttention} className="scale-75" />
                        </div>
                        <div className="space-y-0.5">
                          <Label className="text-xs">Cache Type</Label>
                          <Select value={cacheType} onValueChange={setCacheType}>
                            <SelectTrigger className="h-6 text-xs"><SelectValue /></SelectTrigger>
                            <SelectContent>
                              <SelectItem value="f16" className="text-xs">f16</SelectItem>
                              <SelectItem value="q8_0" className="text-xs">q8_0</SelectItem>
                              <SelectItem value="q4_0" className="text-xs">q4_0</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    </CardContent>
                  </CollapsibleContent>
                </Card>
              </Collapsible>
            </TabsContent>

            {/* ==================== HARDWARE TAB ==================== */}
            <TabsContent value="hardware" className="space-y-3">
              {/* System Overview */}
              <Card className="border-border/50">
                <CardHeader className="p-3 pb-0">
                  <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                    <Monitor className="h-3.5 w-3.5 text-sky-500" />System Overview
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-3">
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="p-2 rounded-lg bg-muted/30">
                      <Cpu className="h-4 w-4 mx-auto mb-1 text-amber-500" />
                      <p className="text-xs font-medium">{hardwareInfo.cpuThreads}</p>
                      <p className="text-[10px] text-muted-foreground">Threads</p>
                    </div>
                    <div className="p-2 rounded-lg bg-muted/30">
                      <MemoryStick className="h-4 w-4 mx-auto mb-1 text-emerald-500" />
                      <p className="text-xs font-medium">{hardwareInfo.totalMemoryGb} GB</p>
                      <p className="text-[10px] text-muted-foreground">RAM</p>
                    </div>
                    <div className="p-2 rounded-lg bg-muted/30">
                      <Server className={cn("h-4 w-4 mx-auto mb-1", hardwareInfo.hasGpu ? "text-purple-500" : "text-muted-foreground")} />
                      <p className="text-xs font-medium">{hardwareInfo.hasGpu ? 'Yes' : 'No'}</p>
                      <p className="text-[10px] text-muted-foreground">GPU</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* GPU Info */}
              {hardwareInfo.hasGpu && (
                <Card className="border-border/50">
                  <CardHeader className="p-3 pb-0">
                    <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                      <Server className={cn("h-3.5 w-3.5", getGpuColor())} />GPU
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-3">
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs"><span className="text-muted-foreground">Name</span><span>{hardwareInfo.gpuName}</span></div>
                      <div className="flex justify-between text-xs"><span className="text-muted-foreground">Memory</span><span>{hardwareInfo.gpuMemoryGb} GB</span></div>
                      <div className="flex justify-between text-xs"><span className="text-muted-foreground">Recommended Layers</span><span>{hardwareInfo.recommendedGpuLayers}</span></div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* CPU Optimizations */}
              <Card className="border-border/50">
                <CardHeader className="p-3 pb-0">
                  <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                    <Rocket className="h-3.5 w-3.5 text-cyan-500" />CPU Optimizations
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-muted-foreground">Level</span>
                    <Badge variant="outline" className="text-[10px]">{cpuOptimizations.optimizationLevel}</Badge>
                  </div>
                  <div className="grid grid-cols-4 gap-1">
                    {[
                      { label: 'AVX', value: cpuOptimizations.hasAvx },
                      { label: 'AVX2', value: cpuOptimizations.hasAvx2 },
                      { label: 'AVX512', value: cpuOptimizations.hasAvx512 },
                      { label: 'FMA', value: cpuOptimizations.hasFma },
                    ].map(opt => (
                      <div key={opt.label} className={cn("p-1.5 rounded text-center text-[10px]", opt.value ? "bg-emerald-500/20 text-emerald-500" : "bg-muted/30 text-muted-foreground")}>
                        {opt.label}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Memory */}
              <Card className="border-border/50">
                <CardHeader className="p-3 pb-0">
                  <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                    <MemoryStick className="h-3.5 w-3.5 text-emerald-500" />Memory
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-3">
                  <div className="space-y-2">
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs"><span>Available</span><span className="text-muted-foreground">{hardwareInfo.availableMemoryGb} / {hardwareInfo.totalMemoryGb} GB</span></div>
                      <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500" style={{ width: `${(hardwareInfo.availableMemoryGb / hardwareInfo.totalMemoryGb) * 100}%` }} />
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </ScrollArea>
    </div>
  )
}
