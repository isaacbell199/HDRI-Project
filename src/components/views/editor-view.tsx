'use client'

import { useState, useEffect, useRef, useCallback, useLayoutEffect } from 'react'
import { useStore, Chapter, CustomStyle, CustomPrompt, CustomTheme, CustomCategory, CustomTone, ModelInfo } from '@/lib/store'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
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
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { toast } from 'sonner'
import {
  Plus,
  ChevronDown,
  ChevronUp,
  FileText,
  Sparkles,
  Loader2,
  BookOpen,
  PanelRightClose,
  PanelRightOpen,
  Wand2,
  User,
  Settings2,
  FileEdit,
  X,
  CheckCircle,
  AlertCircle,
  AlignLeft,
  Target,
  Gauge,
  Users,
  Brain,
  Check,
  MessageSquare,
  PenTool,
  Palette,
  Save,
  FolderOpen,
  Trash2,
  Tag,
  Music,
  Zap,
  Settings,
  Cpu,
  Monitor,
  ArrowRight,
  GripHorizontal,
  Edit,
  Globe
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { AIAssistant } from '@/components/ai-assistant'
import { FloatingAITools } from '@/components/floating-ai-tools'
import {
  getChapters,
  createChapter,
  updateChapter,
  deleteChapter,
  generateText as tauriGenerateText,
  stopGeneration,
  onGenerationChunk,
  isTauri,
  getMemoryInfo,
  getCharacters,
  getLocations,
  getLoreNotes,
  type MemoryInfo,
  type Location as TauriLocation,
  type LoreNote as TauriLoreNote,
  type Character as TauriCharacter,
  type ProjectSettings as TauriProjectSettings,
} from '@/lib/tauri-api'

// Character colors for avatars
const characterColors = [
  'bg-violet-500',
  'bg-emerald-500',
  'bg-amber-500',
  'bg-rose-500',
  'bg-sky-500',
  'bg-fuchsia-500',
  'bg-teal-500',
  'bg-orange-500',
]

// Helper function to format time since last save
function formatTimeSince(date: Date): string {
  const now = new Date()
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000)
  
  if (seconds < 60) return 'just now'
  if (seconds < 120) return '1 min ago'
  if (seconds < 3600) return `${Math.floor(seconds / 60)} mins ago`
  if (seconds < 7200) return '1 hour ago'
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} hours ago`
  return date.toLocaleDateString()
}

// Simple session character type (for editor session)
interface SessionCharacter {
  id: string
  name: string
  description: string
  color: string
  isNarrator: boolean
  instructions: string
  startPhrase: string
  endPhrase: string
}

export function EditorView() {
  const { 
    currentProject, 
    setCurrentProject,
    currentChapter, 
    setCurrentChapter,
    isGenerating,
    setIsGenerating,
    temperature,
    setTemperature,
    maxTokens,
    setMaxTokens,
    topP,
    topK,
    minP,
    repeatPenalty,
    frequencyPenalty,
    presencePenalty,
    rightPanelOpen,
    setRightPanelOpen,
    fontSize,
    isModelLoaded,
    setIsModelLoaded,
    modelPath,
    setModelPath,
    currentModelSystemPrompt,  // System prompt for the loaded model
    useGpu,
    cpuThreads,
    gpuLayers,
    // Models
    models,
    setModels,
    selectedModelId,
    setSelectedModelId,
    setCurrentView,
    // Custom Styles
    customStyles,
    addCustomStyle,
    removeCustomStyle,
    selectedStyleId,
    setSelectedStyleId,
    // Custom Prompts
    customPrompts,
    addCustomPrompt,
    removeCustomPrompt,
    selectedPromptId,
    setSelectedPromptId,
    // Custom Themes
    customThemes,
    addCustomTheme,
    removeCustomTheme,
    selectedThemeId,
    setSelectedThemeId,
    // Custom Categories
    customCategories,
    addCustomCategory,
    removeCustomCategory,
    selectedCategoryId,
    setSelectedCategoryId,
    // Custom Tones
    customTones,
    addCustomTone,
    removeCustomTone,
    selectedToneId,
    setSelectedToneId,
    // Language Mode (input -> output)
    languageMode,
    setLanguageMode,
    // Project Creative Settings (from Wizard)
    projectCategory,
    projectTone,
    projectWritingStyle,
    projectTheme,
    projectPreset,
    restorePresetFromId,
    // Custom Creative Options (created in Wizard)
    projectCustomCategory,
    projectCustomTone,
    projectCustomStyle,
    projectCustomTheme,
  } = useStore()
  
  const [chapters, setChapters] = useState<Chapter[]>([])
  const [editorContent, setEditorContent] = useState('')
  const [chapterTitle, setChapterTitle] = useState('')
  const [isSaving, setIsSaving] = useState(false)
  const [lastSaved, setLastSaved] = useState<Date | null>(null)
  const [newChapterDialogOpen, setNewChapterDialogOpen] = useState(false)
  const [newChapterTitle, setNewChapterTitle] = useState('')
  const [chaptersSidebarOpen, setChaptersSidebarOpen] = useState(true)
  
  // Chapter editing in sidebar
  const [editingChapterId, setEditingChapterId] = useState<string | null>(null)
  const [editingChapterTitle, setEditingChapterTitle] = useState('')
  
  // Delete chapter dialog
  const [deleteChapterDialogOpen, setDeleteChapterDialogOpen] = useState(false)
  const [chapterToDelete, setChapterToDelete] = useState<Chapter | null>(null)
  
  // AI Guided section - Characters in session
  const [sessionCharacters, setSessionCharacters] = useState<SessionCharacter[]>([
    { 
      id: 'narrator', 
      name: 'Narrator', 
      description: 'The narrator tells the story', 
      color: 'bg-slate-500', 
      isNarrator: true,
      instructions: '',
      startPhrase: '',
      endPhrase: ''
    }
  ])
  const [addCharacterDialogOpen, setAddCharacterDialogOpen] = useState(false)
  const [activeCharacterId, setActiveCharacterId] = useState<string>('narrator')
  
  // Popover state for character menu
  const [openPopoverId, setOpenPopoverId] = useState<string | null>(null)
  
  // Add character dialog fields
  const [characterName, setCharacterName] = useState('')
  const [characterDescription, setCharacterDescription] = useState('')
  const [aiInstructions, setAiInstructions] = useState('')
  const [isGeneratingCharacter, setIsGeneratingCharacter] = useState(false)
  const [addCharacterTab, setAddCharacterTab] = useState<'manual' | 'ai'>('manual')
  
  // Generation state per character
  const [generatingCharacterId, setGeneratingCharacterId] = useState<string | null>(null)
  
  // Collapsible sections state - closed by default
  const [categorySectionOpen, setCategorySectionOpen] = useState(false)
  const [toneSectionOpen, setToneSectionOpen] = useState(false)
  const [styleSectionOpen, setStyleSectionOpen] = useState(false)
  const [promptSectionOpen, setPromptSectionOpen] = useState(false)
  const [themeSectionOpen, setThemeSectionOpen] = useState(false)
  const [paramsSectionOpen, setParamsSectionOpen] = useState(false)
  
  // Writing Style form state
  const [styleName, setStyleName] = useState('')
  const [styleDescription, setStyleDescription] = useState('')
  
  // Prompts form state
  const [promptName, setPromptName] = useState('')
  const [positivePrompt, setPositivePrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('')
  
  // Theme form state
  const [themeName, setThemeName] = useState('')
  const [themeDescription, setThemeDescription] = useState('')
  
  // Category form state
  const [categoryName, setCategoryName] = useState('')
  const [categoryDescription, setCategoryDescription] = useState('')
  
  // Tone form state
  const [toneName, setToneName] = useState('')
  const [toneDescription, setToneDescription] = useState('')

  // Model switching state
  const [isLoadingModel, setIsLoadingModel] = useState(false)

  // Loaded project data from Tauri backend (C1 fix: locations, loreNotes, characters, settings)
  const [loadedLocations, setLoadedLocations] = useState<TauriLocation[]>([])
  const [loadedLoreNotes, setLoadedLoreNotes] = useState<TauriLoreNote[]>([])
  const [loadedCharacters, setLoadedCharacters] = useState<TauriCharacter[]>([])
  const [loadedSettings, setLoadedSettings] = useState<TauriProjectSettings | null>(null)

  // AI Suggestions state
  const [lastGeneratedContent, setLastGeneratedContent] = useState<string | null>(null)
  const [lastGeneratedIndex, setLastGeneratedIndex] = useState<number | null>(null)
  const [aiSuggestionOpen, setAiSuggestionOpen] = useState<string | null>(null)
  const [isGeneratingSuggestion, setIsGeneratingSuggestion] = useState(false)
  const [suggestionContent, setSuggestionContent] = useState('')

  // Helper: Get creative settings values for generation (handles custom options)
  // H1/H2/C2 FIX: Now includes genre/tone/style instructions and proper theme separation
  const getCreativeSettings = useCallback(() => {
    // Category: use custom name if available, otherwise use the ID
    const categoryValue = projectCustomCategory?.name || (projectCategory !== 'none' ? projectCategory : undefined)
    // H1 FIX: Pass custom category description as genre instruction
    const categoryInstruction = projectCustomCategory?.description || undefined
    
    // Tone: use custom name if available, otherwise use the ID
    const toneValue = projectCustomTone?.name || (projectTone !== 'none' ? projectTone : undefined)
    // H2 FIX: Pass custom tone description as tone instruction
    const toneInstruction = projectCustomTone?.description || undefined
    
    // Writing Style: use custom name if available, otherwise use the ID
    const styleValue = projectCustomStyle?.name || (projectWritingStyle !== 'none' ? projectWritingStyle : undefined)
    // Pass custom style description as style instruction
    const styleInstruction = projectCustomStyle?.description || undefined
    
    // C2 FIX: Theme now has its own dedicated fields (name + instruction)
    // Previously theme was wrongly passed as customStyleInstruction
    const themeNameValue = projectCustomTheme?.name || (projectTheme !== 'none' ? projectTheme : undefined)
    const themeInstructionValue = projectCustomTheme?.description || undefined
    
    return {
      category: categoryValue,
      categoryInstruction,
      tone: toneValue,
      toneInstruction,
      style: styleValue,
      styleInstruction,
      themeName: themeNameValue,
      themeInstruction: themeInstructionValue,
    }
  }, [projectCategory, projectTone, projectWritingStyle, projectTheme, projectCustomCategory, projectCustomTone, projectCustomStyle, projectCustomTheme])

  // Memoize creative settings for JSX props (avoid calling function multiple times at render)
  const creativeSettings = getCreativeSettings()

  // Textarea resize state
  const [textareaHeight, setTextareaHeight] = useState(400)
  const [isResizingTextarea, setIsResizingTextarea] = useState(false)
  const textareaContainerRef = useRef<HTMLDivElement>(null)

  // Track if content has been generated
  const [hasGeneratedContent, setHasGeneratedContent] = useState(false)

  // Performance stats state
  const [ramUsage, setRamUsage] = useState({ used: 0, total: 0, percent: 0 })
  const [cpuUsage, setCpuUsage] = useState(0)
  const [tokensPerSecond, setTokensPerSecond] = useState(0)
  const [generatedTokensCount, setGeneratedTokensCount] = useState(0)
  const generationStartTimeRef = useRef<number | null>(null)
  const tokenCountRef = useRef(0)

  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const autoSaveTimerRef = useRef<NodeJS.Timeout | null>(null)
  const isAutoScrollingRef = useRef(false)
  const userHasScrolledRef = useRef(false)

  // Ref for saveChapter to avoid stale closure in auto-save timer (H8 fix)
  const saveChapterRef = useRef<() => Promise<void>>(undefined as unknown as () => Promise<void>)

  // Ref for editorContent to avoid stale closure in generation callbacks (L7 fix)
  const editorContentRef = useRef(editorContent)
  useEffect(() => {
    editorContentRef.current = editorContent
  }, [editorContent])

  // Ref for mouse event handlers cleanup (H9 fix)
  const mouseEventHandlersRef = useRef<{ move: ((e: MouseEvent) => void) | null; up: ((e: MouseEvent) => void) | null }>({ move: null, up: null })

  // Detect when user manually scrolls up (to stop auto-scroll)
  const handleTextareaScroll = useCallback(() => {
    if (!textareaRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = textareaRef.current
    // If user scrolled up (not at bottom), stop auto-scroll
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50
    if (!isAtBottom && (isGenerating || isGeneratingSuggestion)) {
      userHasScrolledRef.current = true
    }
  }, [isGenerating, isGeneratingSuggestion])

  // Reset user scroll flag when generation starts
  useEffect(() => {
    if (isGenerating || isGeneratingSuggestion) {
      userHasScrolledRef.current = false
      isAutoScrollingRef.current = true
    }
  }, [isGenerating, isGeneratingSuggestion])

  // Auto-scroll textarea during generation - use useLayoutEffect for immediate scroll after DOM update
  useLayoutEffect(() => {
    if ((isGenerating || isGeneratingSuggestion) && textareaRef.current && !userHasScrolledRef.current) {
      // Use requestAnimationFrame for smooth scroll after paint
      requestAnimationFrame(() => {
        if (textareaRef.current && !userHasScrolledRef.current) {
          textareaRef.current.scrollTop = textareaRef.current.scrollHeight
        }
      })
    }
  }, [editorContent, isGenerating, isGeneratingSuggestion])

  // Continuous scroll during generation as backup
  useEffect(() => {
    if ((isGenerating || isGeneratingSuggestion) && textareaRef.current) {
      const interval = setInterval(() => {
        if (textareaRef.current && isAutoScrollingRef.current && !userHasScrolledRef.current) {
          textareaRef.current.scrollTop = textareaRef.current.scrollHeight
        }
      }, 100)
      return () => clearInterval(interval)
    } else {
      isAutoScrollingRef.current = false
    }
  }, [isGenerating, isGeneratingSuggestion])

  // Handle textarea resize
  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizingTextarea(true)
    const startY = e.clientY
    const startHeight = textareaHeight

    const handleMouseMove = (e: MouseEvent) => {
      const newHeight = Math.max(200, Math.min(800, startHeight + (e.clientY - startY)))
      setTextareaHeight(newHeight)
    }

    const handleMouseUp = () => {
      setIsResizingTextarea(false)
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      mouseEventHandlersRef.current = { move: null, up: null }
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    mouseEventHandlersRef.current = { move: handleMouseMove, up: handleMouseUp }
  }, [textareaHeight])

  // Cleanup mouse event listeners on unmount (H9 fix)
  useEffect(() => {
    return () => {
      if (mouseEventHandlersRef.current.move) {
        document.removeEventListener('mousemove', mouseEventHandlersRef.current.move)
      }
      if (mouseEventHandlersRef.current.up) {
        document.removeEventListener('mouseup', mouseEventHandlersRef.current.up)
      }
      mouseEventHandlersRef.current = { move: null, up: null }
    }
  }, [])

  // Load selected items into forms
  useEffect(() => {
    if (selectedStyleId) {
      const style = customStyles.find(s => s.id === selectedStyleId)
      if (style) {
        setStyleName(style.name)
        setStyleDescription(style.description)
      }
    }
  }, [selectedStyleId, customStyles])
  
  useEffect(() => {
    if (selectedPromptId) {
      const prompt = customPrompts.find(p => p.id === selectedPromptId)
      if (prompt) {
        setPromptName(prompt.name)
        setPositivePrompt(prompt.positivePrompt)
        setNegativePrompt(prompt.negativePrompt)
      }
    }
  }, [selectedPromptId, customPrompts])
  
  useEffect(() => {
    if (selectedThemeId) {
      const theme = customThemes.find(t => t.id === selectedThemeId)
      if (theme) {
        setThemeName(theme.name)
        setThemeDescription(theme.description)
      }
    }
  }, [selectedThemeId, customThemes])
  
  useEffect(() => {
    if (selectedCategoryId) {
      const category = customCategories.find(c => c.id === selectedCategoryId)
      if (category) {
        setCategoryName(category.name)
        setCategoryDescription(category.description)
      }
    }
  }, [selectedCategoryId, customCategories])
  
  useEffect(() => {
    if (selectedToneId) {
      const tone = customTones.find(t => t.id === selectedToneId)
      if (tone) {
        setToneName(tone.name)
        setToneDescription(tone.description)
      }
    }
  }, [selectedToneId, customTones])

  // Update chapter in the chapters list
  const updateChapterInList = useCallback((chapterId: string, updates: Partial<Chapter>) => {
    setChapters(prev => prev.map(ch => 
      ch.id === chapterId 
        ? { 
            ...ch, 
            ...updates,
            wordCount: updates.content ? updates.content.split(/\s+/).filter(Boolean).length : ch.wordCount,
            updatedAt: new Date().toISOString()
          } 
        : ch
    ))
  }, [])

  // Fetch system stats (RAM, CPU)
  useEffect(() => {
    let mounted = true
    let interval: NodeJS.Timeout | null = null
    
    async function fetchSystemStats() {
      if (!mounted) return
      try {
        if (isTauri()) {
          const memInfo = await getMemoryInfo()
          if (mounted && memInfo) {
            setRamUsage({
              used: memInfo.usedGb || memInfo.totalGb - memInfo.availableGb,
              total: memInfo.totalGb,
              percent: memInfo.usedPercent || Math.round(((memInfo.totalGb - memInfo.availableGb) / memInfo.totalGb) * 100)
            })
          }
        } else {
          // Demo mode - simulate system stats
          if (mounted) {
            setRamUsage({
              used: 4.2 + Math.random() * 0.5,
              total: 16,
              percent: 28 + Math.floor(Math.random() * 5)
            })
          }
        }
      } catch (error) {
        console.warn('Failed to fetch system stats:', error)
      }
    }
    
    // Initial fetch
    fetchSystemStats()
    
    // Update every 2 seconds
    interval = setInterval(fetchSystemStats, 2000)
    
    return () => {
      mounted = false
      if (interval) clearInterval(interval)
    }
  }, [])

  // Load chapters from Tauri backend with fallback
  async function loadChapters() {
    if (!currentProject) return

    try {
      if (isTauri()) {
        try {
          const data = await getChapters(currentProject.id)
          if (data && data.length > 0) {
            setChapters(data)
            setCurrentChapter(data[0])
            setEditorContent(data[0].content || '')
            setChapterTitle(data[0].title)
          } else {
            // No chapters in database, create a default one
            setChapters([])
            const defaultChapter: Chapter = {
              id: Date.now().toString(),
              projectId: currentProject.id,
              title: 'Chapter 1',
              content: '',
              orderIndex: 0,
              wordCount: 0,
              status: 'draft',
              createdAt: new Date().toISOString(),
              updatedAt: new Date().toISOString()
            }
            setChapters([defaultChapter])
            setCurrentChapter(defaultChapter)
            setEditorContent('')
            setChapterTitle(defaultChapter.title)
          }
        } catch (tauriError) {
          console.warn('Failed to load chapters from backend, using empty state:', tauriError)
          // Fallback: start with empty chapters
          const defaultChapter: Chapter = {
            id: Date.now().toString(),
            projectId: currentProject.id,
            title: 'Chapter 1',
            content: '',
            orderIndex: 0,
            wordCount: 0,
            status: 'draft',
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
          }
          setChapters([defaultChapter])
          setCurrentChapter(defaultChapter)
          setEditorContent('')
          setChapterTitle(defaultChapter.title)
          toast.info('Started with empty chapter (database unavailable)')
        }
      } else {
        // Demo mode with mock data
        const mockChapters: Chapter[] = [
          {
            id: '1',
            projectId: currentProject.id,
            title: 'The Beginning',
            content: 'The morning sun cast long shadows across the ancient ruins as Elena made her way through the crumbling archways. Her heart raced with anticipation—tonight would change everything.\n\nShe had spent years studying the old texts, deciphering the cryptic messages left by the forgotten civilization. Now, finally, she stood at the threshold of discovery.\n\n"Are you certain about this?" Marcus called from behind, his voice echoing off the weathered stones.\n\nElena paused, her hand resting on the cool surface of the temple entrance. Was she certain? The question had haunted her dreams for months. But there was no turning back now.',
            orderIndex: 0,
            wordCount: 98,
            status: 'draft',
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
          },
          {
            id: '2',
            projectId: currentProject.id,
            title: 'Shadows of the Past',
            content: '',
            orderIndex: 1,
            wordCount: 0,
            status: 'draft',
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
          }
        ]
        setChapters(mockChapters)
        if (mockChapters.length > 0) {
          setCurrentChapter(mockChapters[0])
          setEditorContent(mockChapters[0].content || '')
          setChapterTitle(mockChapters[0].title)
        }
      }
    } catch (error) {
      console.error('Failed to load chapters:', error)
      toast.error('Failed to load chapters')
    }
  }

  // Load chapters when project changes
  useEffect(() => {
    if (currentProject) {
      loadChapters()
    }
  }, [currentProject?.id])

  // C1 FIX: Load locations, loreNotes, characters, and settings from Tauri backend
  // These are NOT returned by get_project() and were always empty before
  useEffect(() => {
    if (!currentProject) {
      setLoadedLocations([])
      setLoadedLoreNotes([])
      setLoadedCharacters([])
      setLoadedSettings(null)
      return
    }
    
    async function loadProjectData() {
      if (!isTauri()) return
      try {
        const [characters, locations, loreNotes] = await Promise.all([
          getCharacters(currentProject!.id).catch(() => []),
          getLocations(currentProject!.id).catch(() => []),
          getLoreNotes(currentProject!.id).catch(() => []),
        ])
        setLoadedCharacters(characters)
        setLoadedLocations(locations)
        setLoadedLoreNotes(loreNotes)
      } catch (error) {
        console.warn('Failed to load project data from backend:', error)
      }
    }
    loadProjectData()
  }, [currentProject?.id])

  // Restore preset when project preset changes
  useEffect(() => {
    if (projectPreset && projectPreset !== 'none') {
      restorePresetFromId(projectPreset)
    }
  }, [projectPreset])

  // Auto-save
  useEffect(() => {
    if (autoSaveTimerRef.current) {
      clearTimeout(autoSaveTimerRef.current)
    }
    if (currentChapter && editorContent !== currentChapter.content) {
      autoSaveTimerRef.current = setTimeout(() => {
        // Use ref to avoid stale saveChapter closure (H8 fix)
        saveChapterRef.current?.()
      }, 2000)
    }
    return () => {
      if (autoSaveTimerRef.current) {
        clearTimeout(autoSaveTimerRef.current)
      }
    }
  }, [editorContent, chapterTitle])

  // Cleanup auto-save timer on unmount to prevent memory leaks
  // and attempts to update state on an unmounted component
  useEffect(() => {
    return () => {
      if (autoSaveTimerRef.current) {
        clearTimeout(autoSaveTimerRef.current)
        autoSaveTimerRef.current = null
      }
    }
  }, [])

  // Save chapter
  const saveChapter = useCallback(async () => {
    if (!currentChapter) return
    setIsSaving(true)
    
    try {
      if (isTauri()) {
        await updateChapter(currentChapter.id, {
          ...currentChapter,
          content: editorContent,
          title: chapterTitle
        })
      }
      
      updateChapterInList(currentChapter.id, { 
        content: editorContent, 
        title: chapterTitle 
      })
      
      setLastSaved(new Date())
    } catch (error) {
      console.error('Failed to save chapter:', error)
    } finally {
      setIsSaving(false)
    }
  }, [currentChapter, editorContent, chapterTitle, updateChapterInList])

  // Keep saveChapterRef in sync (H8 fix)
  useEffect(() => {
    saveChapterRef.current = saveChapter
  }, [saveChapter])

  // Keyboard shortcuts (Ctrl+S for save)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+S or Cmd+S for save
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault()
        if (currentChapter && editorContent !== currentChapter.content) {
          saveChapter()
          toast.success('Chapter saved')
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [currentChapter, editorContent, saveChapter])

  async function handleCreateChapter() {
    if (!newChapterTitle.trim()) {
      toast.error('Chapter title is required')
      return
    }

    if (!currentProject) {
      toast.error('No project selected. Please open a project first.')
      return
    }

    // Generate chapter data
    const newChapterData: Chapter = {
      id: Date.now().toString(),
      projectId: currentProject.id,
      title: newChapterTitle,
      content: '',
      orderIndex: chapters.length,
      wordCount: 0,
      status: 'draft',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    }

    // Try Tauri backend, fallback to local-only mode
    try {
      if (isTauri()) {
        try {
          const newChapter = await createChapter({
            projectId: currentProject.id,
            title: newChapterTitle,
            content: ''
          })
          setChapters([...chapters, newChapter])
          setCurrentChapter(newChapter)
          setEditorContent('')
          setChapterTitle(newChapter.title)
          setHasGeneratedContent(false)
          toast.success('Chapter created and saved to database')
        } catch (tauriError) {
          console.warn('Tauri backend failed, using local fallback:', tauriError)
          // Fallback: create locally without backend
          setChapters([...chapters, newChapterData])
          setCurrentChapter(newChapterData)
          setEditorContent('')
          setChapterTitle(newChapterData.title)
          setHasGeneratedContent(false)
          toast.warning('Chapter created locally (database unavailable)')
        }
      } else {
        // Demo mode - local only
        setChapters([...chapters, newChapterData])
        setCurrentChapter(newChapterData)
        setEditorContent('')
        setChapterTitle(newChapterData.title)
        setHasGeneratedContent(false)
        toast.success('Chapter created (demo mode)')
      }

      setNewChapterDialogOpen(false)
      setNewChapterTitle('')
    } catch (error) {
      console.error('Failed to create chapter:', error)
      const errorMessage = error instanceof Error ? error.message : String(error)
      toast.error(`Failed to create chapter: ${errorMessage}`)
    }
  }

  function selectChapter(chapter: Chapter) {
    // Save current chapter before switching
    if (currentChapter && (editorContent !== currentChapter.content || chapterTitle !== currentChapter.title)) {
      updateChapterInList(currentChapter.id, { content: editorContent, title: chapterTitle })
    }
    
    setCurrentChapter(chapter)
    setEditorContent(chapter.content || '')
    setChapterTitle(chapter.title)
    setHasGeneratedContent(!!chapter.content && chapter.content.length > 0)
  }

  // Handle chapter title change
  function handleTitleChange(newTitle: string) {
    setChapterTitle(newTitle)
    if (currentChapter) {
      updateChapterInList(currentChapter.id, { title: newTitle })
      setCurrentChapter({ ...currentChapter, title: newTitle })
    }
  }

  // Start editing a chapter name in sidebar
  function startEditingChapter(chapter: Chapter, e: React.MouseEvent) {
    e.stopPropagation()
    setEditingChapterId(chapter.id)
    setEditingChapterTitle(chapter.title)
  }

  // Save edited chapter name
  function saveEditedChapterName() {
    if (editingChapterId && editingChapterTitle.trim()) {
      updateChapterInList(editingChapterId, { title: editingChapterTitle.trim() })
      
      // Update current chapter if it's the one being edited
      if (currentChapter?.id === editingChapterId) {
        setCurrentChapter({ ...currentChapter, title: editingChapterTitle.trim() })
        setChapterTitle(editingChapterTitle.trim())
      }
      
      toast.success('Chapter renamed')
    }
    setEditingChapterId(null)
    setEditingChapterTitle('')
  }

  // Cancel editing chapter name
  function cancelEditingChapter() {
    setEditingChapterId(null)
    setEditingChapterTitle('')
  }

  // Start delete chapter confirmation
  function confirmDeleteChapter(chapter: Chapter, e: React.MouseEvent) {
    e.stopPropagation()
    setChapterToDelete(chapter)
    setDeleteChapterDialogOpen(true)
  }

  // Delete chapter
  async function handleDeleteChapter() {
    if (!chapterToDelete) return
    
    try {
      if (isTauri()) {
        await deleteChapter(chapterToDelete.id)
      }
      
      // Remove from local state
      setChapters(chapters.filter(ch => ch.id !== chapterToDelete.id))
      
      // If deleted chapter was current, select another
      if (currentChapter?.id === chapterToDelete.id) {
        const remainingChapters = chapters.filter(ch => ch.id !== chapterToDelete.id)
        if (remainingChapters.length > 0) {
          setCurrentChapter(remainingChapters[0])
          setEditorContent(remainingChapters[0].content || '')
          setChapterTitle(remainingChapters[0].title)
        } else {
          setCurrentChapter(null)
          setEditorContent('')
          setChapterTitle('')
        }
      }
      
      toast.success('Chapter deleted')
    } catch (error) {
      console.error('Failed to delete chapter:', error)
      toast.error('Failed to delete chapter')
    } finally {
      setDeleteChapterDialogOpen(false)
      setChapterToDelete(null)
    }
  }

  // Build comprehensive context for AI generation
  const buildGenerationContext = useCallback(() => {
    const CONTEXT_WINDOW = 8000 // Increased from 3000 to 8000 characters
    const MAX_PREVIOUS_CHAPTERS_CONTENT = 3000

    let context = ''

    // 1. Add previous chapters content (for story continuity)
    const previousChapters = chapters.filter(ch =>
      ch.id !== currentChapter?.id && ch.content && ch.content.length > 0
    )

    if (previousChapters.length > 0) {
      context += '[PREVIOUS CHAPTERS SUMMARY]\n'
      // Get the last 2-3 chapters for context
      const recentChapters = previousChapters.slice(-3)
      for (const ch of recentChapters) {
        const chapterContent = ch.content || ''
        // Summarize if too long (take beginning and end)
        if (chapterContent.length > MAX_PREVIOUS_CHAPTERS_CONTENT) {
          const beginning = chapterContent.slice(0, 1000)
          const ending = chapterContent.slice(-1000)
          context += `Chapter "${ch.title}":\n${beginning}\n...[content abbreviated]...\n${ending}\n\n`
        } else {
          context += `Chapter "${ch.title}":\n${chapterContent}\n\n`
        }
      }
      context += '\n'
    }

    // 2. Add current chapter content (larger window)
    const currentContent = editorContent
    if (currentContent.length > CONTEXT_WINDOW) {
      context += `[CURRENT CHAPTER - LAST ${CONTEXT_WINDOW} CHARS]\n`
      context += currentContent.slice(-CONTEXT_WINDOW)
    } else {
      context += '[CURRENT CHAPTER]\n'
      context += currentContent
    }

    return context
  }, [chapters, currentChapter, editorContent])

  // Generate text using Tauri
  async function handleGenerateText() {
    if (!isModelLoaded) {
      toast.error('Please load an AI model first (Models tab)')
      return
    }

    setIsGenerating(true)

    // Reset stats for new generation
    generationStartTimeRef.current = Date.now()
    tokenCountRef.current = 0
    setGeneratedTokensCount(0)
    setTokensPerSecond(0)

    try {
      if (isTauri()) {
        // Listen for generation chunks
        const unsubscribe = onGenerationChunk((chunk) => {
          if (chunk.done) {
            setIsGenerating(false)
            setHasGeneratedContent(true)
            const startIndex = editorContentRef.current.length
            setLastGeneratedIndex(startIndex)
            toast.success('Generation complete!')
            // Reset tokens/sec after a delay
            setTimeout(() => setTokensPerSecond(0), 2000)
          } else {
            setEditorContent(prev => prev + chunk.content)

            // Count tokens (approximate: 1 chunk ≈ 1 token)
            tokenCountRef.current += 1
            setGeneratedTokensCount(tokenCountRef.current)

            // Calculate tokens per second
            if (generationStartTimeRef.current) {
              const elapsedSeconds = (Date.now() - generationStartTimeRef.current) / 1000
              if (elapsedSeconds > 0) {
                const tps = Math.round((tokenCountRef.current / elapsedSeconds) * 10) / 10
                setTokensPerSecond(tps)
              }
            }
          }
        })

        const creativeSettings = getCreativeSettings()

        // Build comprehensive context (includes previous chapters)
        const generationContext = buildGenerationContext()

        // C1 FIX: Combine characters from loaded backend data + project + session
        const allCharacters = [
          // Priority: loadedCharacters from backend > currentProject.characters from store
          ...(loadedCharacters.length > 0 ? loadedCharacters : (currentProject?.characters || [])),
          ...sessionCharacters
            .filter(sc => sc.id !== 'narrator') // Exclude narrator from list
            .map(sc => ({
              id: sc.id,
              projectId: currentProject?.id || '',
              name: sc.name,
              age: null,
              gender: null,
              role: sc.isNarrator ? 'narrator' : null,
              occupation: null,
              appearance: null,
              distinguishingFeatures: null,
              personality: null,
              traits: null,
              flaws: null,
              fears: null,
              desires: null,
              background: sc.description || null,
              relationships: null,
              skills: null,
              arc: null,
              motivation: null,
              conflicts: null,
              speechPattern: null,
              catchphrases: null,
              notes: sc.instructions || null,
              avatar: null,
              color: sc.color || null,
              createdAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            }))
        ]
        
        // C1 FIX: Use loadedLocations from backend (was always empty before)
        const allLocations = loadedLocations.length > 0 ? loadedLocations : (currentProject?.locations || [])

        // Get project settings
        const projectSettings = loadedSettings || currentProject?.settings || null

        // C1 FIX: Use loadedLoreNotes from backend (was always empty before - Story Beginning was dead)
        const allLoreNotes = loadedLoreNotes.length > 0 ? loadedLoreNotes : (currentProject?.loreNotes || [])

        await tauriGenerateText({
          mode: 'continue',
          text: editorContent,
          context: generationContext, // Use comprehensive context with previous chapters
          temperature,
          maxTokens,
          topP,
          topK,
          minP,
          repeatPenalty,
          frequencyPenalty,
          presencePenalty,
          languageMode,
          systemPrompt: currentModelSystemPrompt || undefined,  // Pass model's system prompt
          // Pass characters and locations for context enrichment
          characters: allCharacters,
          locations: allLocations,
          loreNotes: allLoreNotes,
          // Pass prompts (positive/negative)
          positivePrompt: positivePrompt || undefined,
          negativePrompt: negativePrompt || undefined,
          projectSettings: projectSettings ? {
            id: projectSettings.id,
            projectId: projectSettings.projectId,
            targetWordCount: projectSettings.targetWordCount,
            dailyWordGoal: projectSettings.dailyWordGoal,
            autoSave: projectSettings.autoSave,
            autoSaveInterval: projectSettings.autoSaveInterval,
            contextParagraphs: projectSettings.contextParagraphs,
            genres: projectSettings.genres,
            themes: projectSettings.themes,
            targetAudience: projectSettings.targetAudience,
            writingStyle: projectSettings.writingStyle,
            narrativePov: projectSettings.narrativePov,
            contentRating: projectSettings.contentRating,
            contentWarnings: projectSettings.contentWarnings,
            tonePreferences: projectSettings.tonePreferences,
            timePeriod: projectSettings.timePeriod,
            worldType: projectSettings.worldType,
            language: projectSettings.language,
            languageStyle: projectSettings.languageStyle,
            adultContent: projectSettings.adultContent,
            adultIntensity: projectSettings.adultIntensity,
          } : undefined,
          // Project Creative Settings (from Wizard) - 100% influence on generation
          // H1/H2 FIX: Now pass all instructions properly
          selectedTone: creativeSettings.tone,
          customToneInstruction: creativeSettings.toneInstruction,
          customStyleName: creativeSettings.style,
          customStyleInstruction: creativeSettings.styleInstruction,
          customGenreName: creativeSettings.category,
          customGenreInstruction: creativeSettings.categoryInstruction,
          // C2 FIX: Theme now has its own dedicated fields
          themeName: creativeSettings.themeName,
          themeInstruction: creativeSettings.themeInstruction,
        })
        
        // Cleanup listener after some time
        setTimeout(() => {
          unsubscribe()
        }, 60000)
      } else {
        // Demo mode
        await new Promise(resolve => setTimeout(resolve, 2000))
        const generatedText = `\n\nThe wind whispered through the ancient stones, carrying with it echoes of forgotten voices. Elena pressed forward, her determination burning brighter than ever before.`
        const startIndex = editorContentRef.current.length
        setEditorContent(prev => prev + generatedText)
        setLastGeneratedContent(generatedText)
        setLastGeneratedIndex(startIndex)
        setHasGeneratedContent(true)
        setIsGenerating(false)
        // Simulate stats for demo
        setTokensPerSecond(28.5)
        setGeneratedTokensCount(35)
        toast.success('Text generated!')
      }
    } catch (error) {
      console.error('Generation failed:', error)
      toast.error('Generation failed')
      setIsGenerating(false)
    }
  }
  
  // Generate content for a specific character
  async function generateForCharacter(characterId: string) {
    if (!isModelLoaded) {
      toast.error('Please load an AI model first')
      return
    }
    
    const character = sessionCharacters.find(c => c.id === characterId)
    if (!character) return
    
    if (!character.instructions.trim() && !character.startPhrase.trim()) {
      toast.error('Please add instructions or a start phrase')
      return
    }
    
    setGeneratingCharacterId(characterId)

    try {
      if (isTauri()) {
        // Use Tauri API for generation
        const unsubscribe = onGenerationChunk((chunk) => {
          if (chunk.done) {
            setGeneratingCharacterId(null)
            setOpenPopoverId(null)
            toast.success(`Content generated for ${character.name}`)
          } else {
            setEditorContent(prev => prev + chunk.content)
          }
        })

        const startText = character.startPhrase ? `\n\n${character.startPhrase}\n` : '\n\n'

        setEditorContent(prev => prev + startText)

        const settings = getCreativeSettings()

        // Build context for character generation
        const charContext = buildGenerationContext()

        // Build character data for context
        const characterData = {
          id: character.id,
          projectId: currentProject?.id || '',
          name: character.name,
          age: null,
          gender: null,
          role: character.isNarrator ? 'narrator' : null,
          occupation: null,
          appearance: null,
          distinguishingFeatures: null,
          personality: null,
          traits: null,
          flaws: null,
          fears: null,
          desires: null,
          background: character.description || null,
          relationships: null,
          skills: null,
          arc: null,
          motivation: null,
          conflicts: null,
          speechPattern: null,
          catchphrases: null,
          notes: character.instructions || null,
          avatar: null,
          color: character.color || null,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        }
        
        // C1 FIX: Combine all characters for context (use loaded data from backend)
        const allCharacters = [
          characterData,
          ...(loadedCharacters.length > 0 ? loadedCharacters : (currentProject?.characters || [])).filter(c => c.id !== characterId),
        ]
        
        await tauriGenerateText({
          mode: 'character',
          text: editorContent,
          context: charContext, // Use comprehensive context
          selectedCharacterId: characterId,
          startPhrase: character.startPhrase,
          endPhrase: character.endPhrase,
          temperature,
          maxTokens,
          topP,
          topK,
          minP,
          repeatPenalty,
          frequencyPenalty,
          presencePenalty,
          languageMode,
          // Pass characters for context enrichment
          characters: allCharacters,
          // C1 FIX: Use loaded locations and lore from backend
          locations: loadedLocations.length > 0 ? loadedLocations : (currentProject?.locations || []),
          loreNotes: loadedLoreNotes.length > 0 ? loadedLoreNotes : (currentProject?.loreNotes || []),
          // Pass prompts (positive/negative)
          positivePrompt: positivePrompt || undefined,
          negativePrompt: negativePrompt || undefined,
          // Project Creative Settings (from Wizard) - 100% influence on generation
          // H1/H2 FIX: Now pass all instructions properly
          selectedTone: settings.tone,
          customToneInstruction: settings.toneInstruction,
          customStyleName: settings.style,
          customStyleInstruction: settings.styleInstruction,
          customGenreName: settings.category,
          customGenreInstruction: settings.categoryInstruction,
          // C2 FIX: Theme now has its own dedicated fields
          themeName: settings.themeName,
          themeInstruction: settings.themeInstruction,
        })
        
        setTimeout(() => {
          unsubscribe()
        }, 60000)
      } else {
        // Demo mode
        await new Promise(resolve => setTimeout(resolve, 1500))
        const startText = character.startPhrase ? `\n\n${character.startPhrase}\n` : '\n\n'
        const endText = character.endPhrase ? `\n${character.endPhrase}` : ''
        const simulatedContent = `${startText}[AI-generated content for ${character.name}]${endText}`
        
        setEditorContent(prev => prev + simulatedContent)
        setOpenPopoverId(null)
        toast.success('Content generated (simulation mode)')
      }
    } catch (error) {
      const startText = character.startPhrase ? `\n\n${character.startPhrase}\n` : '\n\n'
      const endText = character.endPhrase ? `\n${character.endPhrase}` : ''
      const simulatedContent = `${startText}[AI-generated content for ${character.name}]${endText}`
      
      setEditorContent(prev => prev + simulatedContent)
      setOpenPopoverId(null)
      toast.success('Content generated (simulation mode)')
    } finally {
      setGeneratingCharacterId(null)
    }
  }
  
  // Add character manually
  function addCharacterManually() {
    if (!characterName.trim()) {
      toast.error('Character name is required')
      return
    }
    
    const newCharacter: SessionCharacter = {
      id: Date.now().toString(),
      name: characterName,
      description: characterDescription,
      color: characterColors[sessionCharacters.length % characterColors.length],
      isNarrator: false,
      instructions: '',
      startPhrase: '',
      endPhrase: ''
    }
    
    setSessionCharacters([...sessionCharacters, newCharacter])
    setCharacterName('')
    setCharacterDescription('')
    setAddCharacterDialogOpen(false)
    toast.success(`Character "${characterName}" added`)
  }
  
  // Generate character with AI
  async function generateCharacterWithAI() {
    if (!aiInstructions.trim()) {
      toast.error('Please enter instructions to generate the character')
      return
    }
    
    setIsGeneratingCharacter(true)
    
    try {
      // For now, simulate character generation (could be extended to use Tauri AI)
      const newCharacter: SessionCharacter = {
        id: Date.now().toString(),
        name: 'Generated Character',
        description: aiInstructions,
        color: characterColors[sessionCharacters.length % characterColors.length],
        isNarrator: false,
        instructions: '',
        startPhrase: '',
        endPhrase: ''
      }
      
      setSessionCharacters([...sessionCharacters, newCharacter])
      setAiInstructions('')
      setAddCharacterDialogOpen(false)
      toast.success('Character created')
    } finally {
      setIsGeneratingCharacter(false)
    }
  }
  
  // Remove character from session
  function removeCharacter(characterId: string) {
    if (characterId === 'narrator') {
      toast.error('The narrator cannot be deleted')
      return
    }
    setSessionCharacters(sessionCharacters.filter(c => c.id !== characterId))
    if (activeCharacterId === characterId) {
      setActiveCharacterId('narrator')
    }
    toast.success('Character deleted')
  }
  
  // Update character fields
  function updateCharacterFields(characterId: string, field: keyof SessionCharacter, value: string) {
    setSessionCharacters(prev => prev.map(c => 
      c.id === characterId ? { ...c, [field]: value } : c
    ))
  }
  
  // Save functions
  function saveStyle() {
    if (!styleName.trim()) {
      toast.error('Style name is required')
      return
    }
    const newStyle: CustomStyle = {
      id: Date.now().toString(),
      name: styleName,
      description: styleDescription,
      createdAt: new Date().toISOString()
    }
    addCustomStyle(newStyle)
    setSelectedStyleId(newStyle.id)
    toast.success(`Style "${styleName}" saved`)
  }
  
  function savePrompt() {
    if (!promptName.trim()) {
      toast.error('Prompt name is required')
      return
    }
    const newPrompt: CustomPrompt = {
      id: Date.now().toString(),
      name: promptName,
      positivePrompt,
      negativePrompt,
      createdAt: new Date().toISOString()
    }
    addCustomPrompt(newPrompt)
    setSelectedPromptId(newPrompt.id)
    toast.success(`Prompt "${promptName}" saved`)
  }
  
  function saveTheme() {
    if (!themeName.trim()) {
      toast.error('Theme name is required')
      return
    }
    const newTheme: CustomTheme = {
      id: Date.now().toString(),
      name: themeName,
      description: themeDescription,
      createdAt: new Date().toISOString()
    }
    addCustomTheme(newTheme)
    setSelectedThemeId(newTheme.id)
    toast.success(`Theme "${themeName}" saved`)
  }
  
  function saveCategory() {
    if (!categoryName.trim()) {
      toast.error('Category name is required')
      return
    }
    const newCategory: CustomCategory = {
      id: Date.now().toString(),
      name: categoryName,
      description: categoryDescription,
      createdAt: new Date().toISOString()
    }
    addCustomCategory(newCategory)
    setSelectedCategoryId(newCategory.id)
    toast.success(`Category "${categoryName}" saved`)
  }
  
  function saveTone() {
    if (!toneName.trim()) {
      toast.error('Tone name is required')
      return
    }
    const newTone: CustomTone = {
      id: Date.now().toString(),
      name: toneName,
      description: toneDescription,
      createdAt: new Date().toISOString()
    }
    addCustomTone(newTone)
    setSelectedToneId(newTone.id)
    toast.success(`Tone "${toneName}" saved`)
  }
  
  // Clear form functions
  function clearStyleForm() {
    setStyleName('')
    setStyleDescription('')
    setSelectedStyleId(null)
  }
  
  function clearPromptForm() {
    setPromptName('')
    setPositivePrompt('')
    setNegativePrompt('')
    setSelectedPromptId(null)
  }
  
  function clearThemeForm() {
    setThemeName('')
    setThemeDescription('')
    setSelectedThemeId(null)
  }
  
  function clearCategoryForm() {
    setCategoryName('')
    setCategoryDescription('')
    setSelectedCategoryId(null)
  }
  
  function clearToneForm() {
    setToneName('')
    setToneDescription('')
    setSelectedToneId(null)
  }

  // AI Suggestion functions
  async function generateSuggestion(type: string) {
    if (!isModelLoaded) {
      toast.error('Please load an AI model first')
      return
    }

    setIsGeneratingSuggestion(true)
    setAiSuggestionOpen(type)
    setSuggestionContent('')

    const prompts: Record<string, string> = {
      'next': 'What happens next in this story? Suggest a logical and compelling continuation.',
      'conflict': 'Identify and develop a potential conflict in this scene. Create dramatic tension.',
      'action': 'Suggest a dynamic and engaging action scene based on the current context.',
      'scenario': 'Develop an alternative scenario or an unexpected direction for the story.'
    }

    try {
      if (isTauri()) {
        // Use Tauri API for suggestion generation
        const unsubscribe = onGenerationChunk((chunk) => {
          if (chunk.done) {
            setIsGeneratingSuggestion(false)
          } else {
            setSuggestionContent(prev => prev + chunk.content)
          }
        })
        
        const suggestionSettings = getCreativeSettings()

        // Build context for suggestion (includes previous chapters)
        const suggestionContext = buildGenerationContext()

        await tauriGenerateText({
          mode: 'suggestion',
          text: editorContent,
          context: `${suggestionContext}\n\n[SUGGESTION REQUEST]\n${prompts[type]}`,
          temperature,
          maxTokens: 800,
          topP,
          topK,
          minP,
          repeatPenalty,
          frequencyPenalty,
          presencePenalty,
          languageMode,
          // Pass lore notes for world context
          loreNotes: currentProject?.loreNotes || [],
          // Pass prompts (positive/negative)
          positivePrompt: positivePrompt || undefined,
          negativePrompt: negativePrompt || undefined,
          // Project Creative Settings (from Wizard) - 100% influence on generation
          selectedTone: suggestionSettings.tone,
          customStyleName: suggestionSettings.style,
          customGenreName: suggestionSettings.category,
          customGenreInstruction: suggestionSettings.categoryInstruction,
          customStyleInstruction: suggestionSettings.styleInstruction,
          themeName: suggestionSettings.themeName,
          themeInstruction: suggestionSettings.themeInstruction,
        })
        
        setTimeout(() => {
          unsubscribe()
        }, 60000)
      } else {
        // Simulate suggestion for demo
        await new Promise(resolve => setTimeout(resolve, 1500))
        const suggestions: Record<string, string> = {
          'next': `**Suggestion: What happens next**

Suddenly, the wind picks up, making the window shutters bang. Elena feels a presence behind her - someone is watching her from the shadows. She turns around slowly, her heart pounding.

"You shouldn't be here," whispers a voice she would recognize anywhere. It's Marcus, but his face is twisted with an expression she has never seen before - a mixture of fear and wild determination.`,
          'conflict': `**Suggestion: Dramatic conflict**

A conflict emerges between Elena's expectations and the reality she discovers. Marcus is hiding something essential - a secret that could change everything she thinks she knows about her quest.

The clues are there, subtle but undeniable: his hesitations, his evasive glances, his unfinished sentences. Elena must choose between her trust in him and her thirst for truth.`,
          'action': `**Suggestion: Action scene**

Without warning, the ground begins to shake. Debris falls from the ancient ceiling. Elena leaps to the side just before a massive stone beam crashes where she was standing.

"Quick! Follow me!" Marcus shouts, pulling her by the arm. They run through the collapsing corridors, dust burning their eyes. Behind them, the ancient temple devours itself.`,
          'scenario': `**Suggestion: Alternative scenario**

What if everything wasn't what it seemed? The "ruins" Elena is exploring are actually a complex device left by an advanced civilization - a kind of living library that chooses its visitors.

The texts she deciphered are not warnings, but invitations. The real treasure isn't material: it's knowledge itself, preserved through the ages for those who would know how to seek it.`
        }
        setSuggestionContent(suggestions[type] || 'Suggestion generated.')
        setIsGeneratingSuggestion(false)
      }
    } catch (error) {
      // Simulate suggestion for demo
      const suggestions: Record<string, string> = {
        'next': `**Suggestion: What happens next**

Suddenly, the wind picks up, making the window shutters bang. Elena feels a presence behind her - someone is watching her from the shadows. She turns around slowly, her heart pounding.`,
        'conflict': `**Suggestion: Dramatic conflict**

A conflict emerges between Elena's expectations and the reality she discovers.`,
        'action': `**Suggestion: Action scene**

Without warning, the ground begins to shake. Debris falls from the ancient ceiling.`,
        'scenario': `**Suggestion: Alternative scenario**

What if everything wasn't what it seemed?`
      }
      setSuggestionContent(suggestions[type] || 'Suggestion generated.')
      setIsGeneratingSuggestion(false)
    }
  }

  // Apply suggestion to editor
  function applySuggestion() {
    if (suggestionContent) {
      const startIndex = editorContentRef.current.length
      setEditorContent(prev => prev + '\n\n' + suggestionContent)
      setLastGeneratedContent(suggestionContent)
      setLastGeneratedIndex(startIndex)
      setHasGeneratedContent(true)
      setAiSuggestionOpen(null)
      setSuggestionContent('')
      toast.success('Suggestion applied to text')
    }
  }

  // Delete last generated content
  function deleteLastGeneration() {
    if (lastGeneratedContent && lastGeneratedIndex !== null) {
      setEditorContent(prev => prev.slice(0, lastGeneratedIndex))
      setLastGeneratedContent(null)
      setLastGeneratedIndex(null)
      toast.success('Last generation deleted')
    } else {
      toast.error('No recent generation to delete')
    }
  }

  // Regenerate last instruction
  async function regenerateLastInstruction() {
    if (!lastGeneratedContent) {
      toast.error('No recent generation to regenerate')
      return
    }

    // First delete the last generation
    if (lastGeneratedIndex !== null) {
      setEditorContent(prev => prev.slice(0, lastGeneratedIndex))
    }

    // Then regenerate
    setIsGeneratingSuggestion(true)
    setAiSuggestionOpen('regenerate')

    try {
      if (isTauri()) {
        const unsubscribe = onGenerationChunk((chunk) => {
          if (chunk.done) {
            setIsGeneratingSuggestion(false)
            setAiSuggestionOpen(null)
            toast.success('Content regenerated successfully')
          } else {
            setEditorContent(prev => prev + chunk.content)
          }
        })
        
        const startIndex = editorContentRef.current.length
        setLastGeneratedIndex(startIndex)

        // Build context for regeneration
        const regenerateContext = buildGenerationContext()

        await tauriGenerateText({
          mode: 'regenerate',
          text: editorContent,
          context: `${regenerateContext}\n\n[REGENERATION REQUEST]\nRegenerate the previous content with a different and creative approach.`,
          temperature: temperature + 0.1,
          maxTokens: 800,
          topP,
          topK,
          minP,
          repeatPenalty,
          frequencyPenalty,
          presencePenalty,
          languageMode,
          // Pass lore notes for world context
          loreNotes: currentProject?.loreNotes || [],
          // Pass prompts (positive/negative)
          positivePrompt: positivePrompt || undefined,
          negativePrompt: negativePrompt || undefined,
        })
        
        setTimeout(() => {
          unsubscribe()
        }, 60000)
      } else {
        // Simulate regeneration
        await new Promise(resolve => setTimeout(resolve, 1500))
        const newContent = `**New regenerated version**

The setting sun casts long shadows across the broken columns. Elena stops, feeling the air change around her - a palpable tension, like before a storm.

The inscriptions on the walls seem to pulse with their own light, revealing symbols she hadn't noticed before. Her heart races: she is about to discover something extraordinary.`
        const startIndex = editorContentRef.current.length
        setEditorContent(prev => prev + '\n\n' + newContent)
        setLastGeneratedContent(newContent)
        setLastGeneratedIndex(startIndex)
        setHasGeneratedContent(true)
        toast.success('Content regenerated (demo mode)')
        setIsGeneratingSuggestion(false)
        setAiSuggestionOpen(null)
      }
    } catch (error) {
      // Simulate regeneration
      const newContent = `**New regenerated version**

The setting sun casts long shadows across the broken columns. Elena stops, feeling the air change around her.`
      const startIndex = editorContentRef.current.length
      setEditorContent(prev => prev + '\n\n' + newContent)
      setLastGeneratedContent(newContent)
      setLastGeneratedIndex(startIndex)
      setHasGeneratedContent(true)
      toast.success('Content regenerated (demo mode)')
      setIsGeneratingSuggestion(false)
      setAiSuggestionOpen(null)
    }
  }

  // Stream handler for real-time token generation
  const handleStreamChunk = useCallback((chunk: string) => {
    setEditorContent(prev => prev + chunk)
    setHasGeneratedContent(true)
  }, [])

  // Word count
  const wordCount = editorContent.split(/\s+/).filter(Boolean).length
  const fontSizeClass = {
    'sm': 'text-sm',
    'md': 'text-base',
    'lg': 'text-lg',
    'xl': 'text-xl'
  }[fontSize]

  if (!currentProject) {
    return (
      <div className="flex-1 flex items-center justify-center bg-gradient-to-br from-background to-muted/20">
        <div className="text-center">
          <BookOpen className="h-20 w-20 text-muted-foreground/30 mx-auto mb-4" />
          <h2 className="text-2xl font-bold mb-2">No Project Selected</h2>
          <p className="text-muted-foreground">Select a project to start writing</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex h-full flex-col">
      <div className="flex-1 flex h-full">
        {/* Chapters Sidebar */}
        <AnimatePresence>
          {chaptersSidebarOpen && (
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 260, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="border-r border-border/50 flex flex-col bg-muted/20 overflow-hidden"
            >
              <div className="p-3 border-b border-border/50 flex items-center justify-between">
                <h3 className="font-semibold text-sm flex items-center gap-2">
                  <FileText className="h-4 w-4 text-violet-500" />
                  Chapters
                  <Badge variant="secondary" className="text-xs">{chapters.length}</Badge>
                </h3>
                <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setNewChapterDialogOpen(true)}>
                  <Plus className="h-4 w-4" />
                </Button>
              </div>
              <ScrollArea className="flex-1">
                <div className="p-2 space-y-1">
                  {chapters.map((chapter, index) => (
                    <motion.div
                      key={chapter.id}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className={cn(
                        "group flex items-center gap-2 p-2.5 rounded-lg cursor-pointer transition-all",
                        currentChapter?.id === chapter.id 
                          ? "bg-gradient-to-r from-blue-500/20 via-sky-500/15 to-blue-500/10 text-blue-700 dark:text-blue-300 border border-blue-500/30 shadow-sm shadow-blue-500/10" 
                          : "hover:bg-muted/60 border border-transparent"
                      )}
                      onClick={() => editingChapterId !== chapter.id && selectChapter(chapter)}
                    >
                      <FileText className={cn(
                        "h-4 w-4 shrink-0 transition-colors",
                        currentChapter?.id === chapter.id 
                          ? "text-blue-500" 
                          : "text-muted-foreground"
                      )} />
                      
                      {editingChapterId === chapter.id ? (
                        <div className="flex-1 flex items-center gap-1">
                          <Input
                            value={editingChapterTitle}
                            onChange={(e) => setEditingChapterTitle(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') saveEditedChapterName()
                              if (e.key === 'Escape') cancelEditingChapter()
                            }}
                            className="h-6 text-sm px-1.5 py-0"
                            autoFocus
                            onClick={(e) => e.stopPropagation()}
                          />
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 shrink-0"
                            onClick={(e) => { e.stopPropagation(); saveEditedChapterName() }}
                          >
                            <Check className="h-3 w-3 text-emerald-500" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 shrink-0"
                            onClick={(e) => { e.stopPropagation(); cancelEditingChapter() }}
                          >
                            <X className="h-3 w-3 text-destructive" />
                          </Button>
                        </div>
                      ) : (
                        <>
                          <span className="text-sm truncate flex-1 font-medium">
                            {chapter.title}
                          </span>
                          <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 shrink-0"
                              onClick={(e) => startEditingChapter(chapter, e)}
                            >
                              <Edit className="h-3 w-3 text-muted-foreground" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 shrink-0 hover:bg-destructive/10"
                              onClick={(e) => confirmDeleteChapter(chapter, e)}
                            >
                              <Trash2 className="h-3 w-3 text-muted-foreground hover:text-destructive" />
                            </Button>
                          </div>
                        </>
                      )}
                    </motion.div>
                  ))}
                </div>
              </ScrollArea>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main Editor */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Header */}
          <header className="h-14 border-b border-border/50 flex items-center justify-between px-4 shrink-0 bg-background/80 backdrop-blur-sm">
            <div className="flex items-center gap-3">
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-8 w-8"
                onClick={() => setChaptersSidebarOpen(!chaptersSidebarOpen)}
              >
                {chaptersSidebarOpen ? <PanelRightClose className="h-4 w-4" /> : <PanelRightOpen className="h-4 w-4" />}
              </Button>
              <div className="flex items-center gap-1">
                <span className="text-lg font-semibold text-muted-foreground">
                  Chapter {currentChapter ? currentChapter.orderIndex + 1 : 1}:
                </span>
                <Input
                  value={chapterTitle}
                  onChange={(e) => handleTitleChange(e.target.value)}
                  className="text-lg font-semibold border-none bg-transparent h-8 w-auto min-w-0 focus-visible:ring-0 px-1"
                  placeholder="Chapter Title"
                />
              </div>
            </div>
            <div className="flex items-center gap-3">
              {/* Manual Save Button */}
              <Button
                variant="outline"
                size="sm"
                className="h-8 gap-1.5"
                onClick={saveChapter}
                disabled={isSaving || !currentChapter}
              >
                {isSaving ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    <span>Saving...</span>
                  </>
                ) : (
                  <>
                    <Save className="h-3.5 w-3.5" />
                    <span>Save</span>
                  </>
                )}
              </Button>
              
              {/* Word Count */}
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <AlignLeft className="h-4 w-4" />
                <span className="font-medium">{wordCount.toLocaleString()}</span>
                <span>words</span>
              </div>
              
              {/* Save Status Indicator */}
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className={cn(
                  "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-all",
                  isSaving 
                    ? "bg-amber-500/15 text-amber-600 dark:text-amber-400" 
                    : lastSaved 
                      ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                      : "bg-muted/50 text-muted-foreground"
                )}
              >
                {isSaving ? (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin" />
                    <span>Saving...</span>
                  </>
                ) : lastSaved ? (
                  <>
                    <CheckCircle className="h-3 w-3" />
                    <span>Saved {formatTimeSince(lastSaved)}</span>
                  </>
                ) : (
                  <>
                    <FileEdit className="h-3 w-3" />
                    <span>Not saved</span>
                  </>
                )}
              </motion.div>
              
              {isGenerating && (
                <Button variant="destructive" size="sm" onClick={() => {
                  setIsGenerating(false)
                  if (isTauri()) stopGeneration()
                }}>
                  <X className="h-4 w-4 mr-1" /> Stop
                </Button>
              )}
              
              {/* Language Mode Dropdown (input -> output) */}
              <div className="flex items-center gap-1">
                <Globe className="h-3.5 w-3.5 text-muted-foreground" />
                <select
                  value={languageMode}
                  onChange={(e) => setLanguageMode(e.target.value as 'fr-fr' | 'fr-en' | 'en-fr' | 'en-en')}
                  className="h-7 text-xs bg-transparent border border-border/50 rounded px-1.5 cursor-pointer focus:outline-none focus:ring-1 focus:ring-primary/50 font-medium"
                >
                  <option value="fr-fr">FR → FR</option>
                  <option value="fr-en">FR → EN</option>
                  <option value="en-fr">EN → FR</option>
                  <option value="en-en">EN → EN</option>
                </select>
              </div>
              
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-8 w-8"
                onClick={() => setRightPanelOpen(!rightPanelOpen)}
              >
                {rightPanelOpen ? <PanelRightClose className="h-4 w-4" /> : <PanelRightOpen className="h-4 w-4" />}
              </Button>
            </div>
          </header>

          {/* Editor Area with AI Guided Section */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Resizable Textarea Container */}
            <div 
              ref={textareaContainerRef}
              className="relative bg-gradient-to-b from-background to-muted/10"
              style={{ height: textareaHeight }}
            >
              <Textarea
                ref={textareaRef}
                value={editorContent}
                onChange={(e) => setEditorContent(e.target.value)}
                onScroll={handleTextareaScroll}
                className={cn(
                  "w-full h-full border-none rounded-none resize-none focus-visible:ring-0 p-8 pr-16 leading-relaxed overflow-y-auto",
                  fontSizeClass
                )}
                style={{ scrollbarGutter: 'stable' }}
                placeholder="Start writing your story..."
              />
              
              {/* Floating AI Tools - Fixed position on right side */}
              <div className="absolute right-3 top-1/2 -translate-y-1/2 z-10">
                <FloatingAITools
                  isModelLoaded={isModelLoaded}
                  editorContent={editorContent}
                  lastGeneratedContent={lastGeneratedContent}
                  lastGeneratedIndex={lastGeneratedIndex}
                  onApplySuggestion={(content) => {
                    const startIndex = editorContentRef.current.length
                    setEditorContent(prev => prev + content)
                    setLastGeneratedContent(content)
                    setLastGeneratedIndex(startIndex)
                    setHasGeneratedContent(true)
                  }}
                  onStreamChunk={(chunk) => {
                    setEditorContent(prev => prev + chunk)
                    setHasGeneratedContent(true)
                  }}
                  onGenerationStart={() => {
                    setIsGeneratingSuggestion(true)
                  }}
                  onGenerationEnd={() => {
                    setIsGeneratingSuggestion(false)
                  }}
                  onClearLastGeneration={deleteLastGeneration}
                  onRegenerate={regenerateLastInstruction}
                  isGenerating={isGeneratingSuggestion}
                  temperature={temperature}
                  maxTokens={maxTokens}
                  languageMode={languageMode}
                  // Sampling parameters
                  topP={topP}
                  topK={topK}
                  minP={minP}
                  repeatPenalty={repeatPenalty}
                  frequencyPenalty={frequencyPenalty}
                  presencePenalty={presencePenalty}
                  // Creative Settings (from Wizard) - memoized values
                  creativeCategory={creativeSettings.category}
                  creativeCategoryInstruction={creativeSettings.categoryInstruction}
                  creativeTone={creativeSettings.tone}
                  creativeToneInstruction={creativeSettings.toneInstruction}
                  creativeStyle={creativeSettings.style}
                  creativeStyleInstruction={creativeSettings.styleInstruction}
                  creativeThemeName={creativeSettings.themeName}
                  creativeThemeInstruction={creativeSettings.themeInstruction}
                />
              </div>
              
              {/* Resize Handle */}
              <div 
                className={cn(
                  "absolute bottom-0 left-0 right-0 h-6 flex items-center justify-center cursor-ns-resize group",
                  "hover:bg-violet-500/10 transition-colors",
                  isResizingTextarea && "bg-violet-500/20"
                )}
                onMouseDown={handleResizeStart}
              >
                <div className={cn(
                  "flex items-center gap-0.5 px-3 py-1 rounded-full transition-all",
                  "opacity-0 group-hover:opacity-100",
                  isResizingTextarea && "opacity-100 bg-violet-500/20"
                )}>
                  <GripHorizontal className="h-3 w-3 text-muted-foreground" />
                  <span className="text-[9px] text-muted-foreground font-medium">
                    {isResizingTextarea ? `${textareaHeight}px` : 'Drag to resize'}
                  </span>
                  <GripHorizontal className="h-3 w-3 text-muted-foreground" />
                </div>
              </div>
            </div>

            {/* AI Assistant Section - Fixed height panel */}
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              transition={{ duration: 0.3 }}
              className="border-t border-border/50 bg-gradient-to-b from-muted/30 to-muted/10 overflow-hidden shrink-0"
            >
              <div className="p-3">
                <AIAssistant
                  editorContent={editorContent}
                  onStreamChunk={handleStreamChunk}
                  isGenerating={isGeneratingSuggestion}
                  onGenerationStart={() => setIsGeneratingSuggestion(true)}
                  onGenerationEnd={() => setIsGeneratingSuggestion(false)}
                  creativeCategory={creativeSettings.category}
                  creativeCategoryInstruction={creativeSettings.categoryInstruction}
                  creativeTone={creativeSettings.tone}
                  creativeToneInstruction={creativeSettings.toneInstruction}
                  creativeStyle={creativeSettings.style}
                  creativeStyleInstruction={creativeSettings.styleInstruction}
                  creativeThemeName={creativeSettings.themeName}
                  creativeThemeInstruction={creativeSettings.themeInstruction}
                  projectCharacters={sessionCharacters.map(c => ({
                    id: c.id,
                    name: c.name,
                    description: c.description
                  }))}
                />
              </div>
            </motion.div>
          </div>
        </div>

        {/* Right Panel - Customizable Sections */}
        <AnimatePresence>
          {rightPanelOpen && (
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 360, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="border-l border-border/50 flex flex-col bg-muted/20 overflow-hidden"
            >
              <div className="p-3 border-b border-border/50 flex items-center justify-between">
                <h3 className="font-semibold text-sm flex items-center gap-2">
                  <Wand2 className="h-4 w-4 text-violet-500" />
                  AI Settings
                </h3>
                <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setRightPanelOpen(false)}>
                  <X className="h-4 w-4" />
                </Button>
              </div>

              <ScrollArea className="flex-1">
                <div className="p-3 space-y-3">
                  {/* Model Status */}
                  <div className={cn(
                    "p-3 rounded-xl transition-all",
                    isModelLoaded
                      ? "bg-gradient-to-br from-emerald-500/10 to-teal-500/5 border border-emerald-500/20"
                      : "bg-gradient-to-br from-amber-500/10 to-orange-500/5 border border-amber-500/20"
                  )}>
                    <div className="flex items-center gap-3">
                      {isModelLoaded ? (
                        <div className="p-1.5 rounded-lg bg-emerald-500/20">
                          <CheckCircle className="h-4 w-4 text-emerald-500" />
                        </div>
                      ) : (
                        <div className="p-1.5 rounded-lg bg-amber-500/20">
                          <AlertCircle className="h-4 w-4 text-amber-500" />
                        </div>
                      )}
                      <div className="flex-1">
                        <p className="text-sm font-medium">{isModelLoaded ? 'Model Ready' : 'No Model Loaded'}</p>
                        <p className="text-xs text-muted-foreground">
                          {isModelLoaded
                            ? models.find(m => m.id === selectedModelId)?.name || 'Model loaded'
                            : 'Load model in Models tab'
                          }
                        </p>
                      </div>
                      {isModelLoaded && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          onClick={() => setCurrentView('models')}
                        >
                          <Settings className="h-4 w-4" />
                        </Button>
                      )}
                    </div>
                    
                    {/* Performance Indicators */}
                    {isModelLoaded && (
                      <div className="grid grid-cols-3 gap-2 mt-3 pt-3 border-t border-border/30">
                        {/* RAM Indicator */}
                        <div className="flex flex-col items-center p-2 rounded-lg bg-gradient-to-b from-violet-500/10 to-violet-500/5 border border-violet-500/20">
                          <div className="flex items-center gap-1 mb-1">
                            <Monitor className="h-3 w-3 text-violet-500" />
                            <span className="text-[10px] font-medium text-violet-600 dark:text-violet-400">RAM</span>
                          </div>
                          <div className="w-full h-1.5 bg-violet-500/20 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-gradient-to-r from-violet-500 to-purple-500 rounded-full transition-all duration-300" 
                              style={{ width: `${Math.min(ramUsage.percent, 100)}%` }}
                            />
                          </div>
                          <span className="text-[10px] text-muted-foreground mt-1">
                            {ramUsage.used.toFixed(1)} GB
                          </span>
                        </div>
                        
                        {/* CPU Indicator */}
                        <div className="flex flex-col items-center p-2 rounded-lg bg-gradient-to-b from-sky-500/10 to-sky-500/5 border border-sky-500/20">
                          <div className="flex items-center gap-1 mb-1">
                            <Cpu className="h-3 w-3 text-sky-500" />
                            <span className="text-[10px] font-medium text-sky-600 dark:text-sky-400">CPU</span>
                          </div>
                          <div className="w-full h-1.5 bg-sky-500/20 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-gradient-to-r from-sky-500 to-cyan-500 rounded-full transition-all duration-300" 
                              style={{ width: `${Math.min(isGenerating ? 85 : 25, 100)}%` }}
                            />
                          </div>
                          <span className="text-[10px] text-muted-foreground mt-1">
                            {isGenerating ? '85%' : '25%'}
                          </span>
                        </div>
                        
                        {/* Tokens/sec Indicator */}
                        <div className="flex flex-col items-center p-2 rounded-lg bg-gradient-to-b from-emerald-500/10 to-emerald-500/5 border border-emerald-500/20">
                          <div className="flex items-center gap-1 mb-1">
                            <Zap className="h-3 w-3 text-emerald-500" />
                            <span className="text-[10px] font-medium text-emerald-600 dark:text-emerald-400">TOK/s</span>
                          </div>
                          <div className="w-full h-1.5 bg-emerald-500/20 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full transition-all duration-300" 
                              style={{ width: `${Math.min((tokensPerSecond / 50) * 100, 100)}%` }}
                            />
                          </div>
                          <span className="text-[10px] text-muted-foreground mt-1">
                            {tokensPerSecond > 0 ? tokensPerSecond.toFixed(1) : '--'}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Category Section */}
                  <Collapsible open={categorySectionOpen} onOpenChange={setCategorySectionOpen}>
                    <div className="rounded-xl border border-border/50 bg-background/50 overflow-hidden">
                      <CollapsibleTrigger className="w-full">
                        <div className="flex items-center justify-between p-3 hover:bg-muted/30 transition-colors">
                          <div className="flex items-center gap-2">
                            <Tag className="h-4 w-4 text-rose-500" />
                            <span className="font-medium text-sm">Category</span>
                            {selectedCategoryId && (
                              <Badge variant="secondary" className="text-xs">
                                {customCategories.find(c => c.id === selectedCategoryId)?.name}
                              </Badge>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                                <Button variant="ghost" size="icon" className="h-7 w-7">
                                  <FolderOpen className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end" className="w-48">
                                {customCategories.length === 0 ? (
                                  <div className="px-2 py-1.5 text-xs text-muted-foreground">
                                    No saved categories
                                  </div>
                                ) : (
                                  customCategories.map(category => (
                                    <DropdownMenuItem
                                      key={category.id}
                                      onClick={() => setSelectedCategoryId(category.id)}
                                      className="flex items-center justify-between"
                                    >
                                      <span>{category.name}</span>
                                      <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-5 w-5"
                                        onClick={(e) => {
                                          e.stopPropagation()
                                          removeCustomCategory(category.id)
                                          if (selectedCategoryId === category.id) setSelectedCategoryId(null)
                                          toast.success('Category deleted')
                                        }}
                                      >
                                        <Trash2 className="h-3 w-3" />
                                      </Button>
                                    </DropdownMenuItem>
                                  ))
                                )}
                              </DropdownMenuContent>
                            </DropdownMenu>
                            {categorySectionOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                          </div>
                        </div>
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        <div className="p-3 pt-0 space-y-3 border-t border-border/30">
                          <div className="space-y-2 pt-3">
                            <Label className="text-xs">Category name</Label>
                            <Input
                              value={categoryName}
                              onChange={(e) => setCategoryName(e.target.value)}
                              placeholder="Ex: Fantasy, Science-fiction..."
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label className="text-xs">Description</Label>
                            <Textarea
                              value={categoryDescription}
                              onChange={(e) => setCategoryDescription(e.target.value)}
                              placeholder="Describe the category..."
                              className="min-h-[60px] max-h-[100px] resize-none text-sm"
                            />
                          </div>
                          <div className="flex gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              className="flex-1 h-8 text-xs"
                              onClick={clearCategoryForm}
                            >
                              <X className="h-3 w-3 mr-1" />
                              Clear
                            </Button>
                            <Button
                              size="sm"
                              className="flex-1 h-8 text-xs bg-gradient-to-r from-rose-600 to-pink-600 border-0"
                              onClick={saveCategory}
                              disabled={!categoryName.trim()}
                            >
                              <Save className="h-3 w-3 mr-1" />
                              Save
                            </Button>
                          </div>
                        </div>
                      </CollapsibleContent>
                    </div>
                  </Collapsible>

                  {/* Tone Section */}
                  <Collapsible open={toneSectionOpen} onOpenChange={setToneSectionOpen}>
                    <div className="rounded-xl border border-border/50 bg-background/50 overflow-hidden">
                      <CollapsibleTrigger className="w-full">
                        <div className="flex items-center justify-between p-3 hover:bg-muted/30 transition-colors">
                          <div className="flex items-center gap-2">
                            <Music className="h-4 w-4 text-sky-500" />
                            <span className="font-medium text-sm">Tone</span>
                            {selectedToneId && (
                              <Badge variant="secondary" className="text-xs">
                                {customTones.find(t => t.id === selectedToneId)?.name}
                              </Badge>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                                <Button variant="ghost" size="icon" className="h-7 w-7">
                                  <FolderOpen className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end" className="w-48">
                                {customTones.length === 0 ? (
                                  <div className="px-2 py-1.5 text-xs text-muted-foreground">
                                    No saved tones
                                  </div>
                                ) : (
                                  customTones.map(tone => (
                                    <DropdownMenuItem
                                      key={tone.id}
                                      onClick={() => setSelectedToneId(tone.id)}
                                      className="flex items-center justify-between"
                                    >
                                      <span>{tone.name}</span>
                                      <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-5 w-5"
                                        onClick={(e) => {
                                          e.stopPropagation()
                                          removeCustomTone(tone.id)
                                          if (selectedToneId === tone.id) setSelectedToneId(null)
                                          toast.success('Tone deleted')
                                        }}
                                      >
                                        <Trash2 className="h-3 w-3" />
                                      </Button>
                                    </DropdownMenuItem>
                                  ))
                                )}
                              </DropdownMenuContent>
                            </DropdownMenu>
                            {toneSectionOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                          </div>
                        </div>
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        <div className="p-3 pt-0 space-y-3 border-t border-border/30">
                          <div className="space-y-2 pt-3">
                            <Label className="text-xs">Tone name</Label>
                            <Input
                              value={toneName}
                              onChange={(e) => setToneName(e.target.value)}
                              placeholder="Ex: Dark, Joyful, Mysterious..."
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label className="text-xs">Description</Label>
                            <Textarea
                              value={toneDescription}
                              onChange={(e) => setToneDescription(e.target.value)}
                              placeholder="Describe the story tone..."
                              className="min-h-[60px] max-h-[100px] resize-none text-sm"
                            />
                          </div>
                          <div className="flex gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              className="flex-1 h-8 text-xs"
                              onClick={clearToneForm}
                            >
                              <X className="h-3 w-3 mr-1" />
                              Clear
                            </Button>
                            <Button
                              size="sm"
                              className="flex-1 h-8 text-xs bg-gradient-to-r from-sky-600 to-cyan-600 border-0"
                              onClick={saveTone}
                              disabled={!toneName.trim()}
                            >
                              <Save className="h-3 w-3 mr-1" />
                              Save
                            </Button>
                          </div>
                        </div>
                      </CollapsibleContent>
                    </div>
                  </Collapsible>

                  {/* Writing Style Section */}
                  <Collapsible open={styleSectionOpen} onOpenChange={setStyleSectionOpen}>
                    <div className="rounded-xl border border-border/50 bg-background/50 overflow-hidden">
                      <CollapsibleTrigger className="w-full">
                        <div className="flex items-center justify-between p-3 hover:bg-muted/30 transition-colors">
                          <div className="flex items-center gap-2">
                            <PenTool className="h-4 w-4 text-violet-500" />
                            <span className="font-medium text-sm">Writing Style</span>
                            {selectedStyleId && (
                              <Badge variant="secondary" className="text-xs">
                                {customStyles.find(s => s.id === selectedStyleId)?.name}
                              </Badge>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                                <Button variant="ghost" size="icon" className="h-7 w-7">
                                  <FolderOpen className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end" className="w-48">
                                {customStyles.length === 0 ? (
                                  <div className="px-2 py-1.5 text-xs text-muted-foreground">
                                    No saved styles
                                  </div>
                                ) : (
                                  customStyles.map(style => (
                                    <DropdownMenuItem
                                      key={style.id}
                                      onClick={() => setSelectedStyleId(style.id)}
                                      className="flex items-center justify-between"
                                    >
                                      <span>{style.name}</span>
                                      <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-5 w-5"
                                        onClick={(e) => {
                                          e.stopPropagation()
                                          removeCustomStyle(style.id)
                                          if (selectedStyleId === style.id) setSelectedStyleId(null)
                                          toast.success('Style deleted')
                                        }}
                                      >
                                        <Trash2 className="h-3 w-3" />
                                      </Button>
                                    </DropdownMenuItem>
                                  ))
                                )}
                              </DropdownMenuContent>
                            </DropdownMenu>
                            {styleSectionOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                          </div>
                        </div>
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        <div className="p-3 pt-0 space-y-3 border-t border-border/30">
                          <div className="space-y-2 pt-3">
                            <Label className="text-xs">Style name</Label>
                            <Input
                              value={styleName}
                              onChange={(e) => setStyleName(e.target.value)}
                              placeholder="Ex: Descriptive, Action..."
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label className="text-xs">Description</Label>
                            <Textarea
                              value={styleDescription}
                              onChange={(e) => setStyleDescription(e.target.value)}
                              placeholder="Describe the writing style..."
                              className="min-h-[60px] max-h-[100px] resize-none text-sm"
                            />
                          </div>
                          <div className="flex gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              className="flex-1 h-8 text-xs"
                              onClick={clearStyleForm}
                            >
                              <X className="h-3 w-3 mr-1" />
                              Clear
                            </Button>
                            <Button
                              size="sm"
                              className="flex-1 h-8 text-xs bg-gradient-to-r from-violet-600 to-fuchsia-600 border-0"
                              onClick={saveStyle}
                              disabled={!styleName.trim()}
                            >
                              <Save className="h-3 w-3 mr-1" />
                              Save
                            </Button>
                          </div>
                        </div>
                      </CollapsibleContent>
                    </div>
                  </Collapsible>

                  {/* Prompts Section */}
                  <Collapsible open={promptSectionOpen} onOpenChange={setPromptSectionOpen}>
                    <div className="rounded-xl border border-border/50 bg-background/50 overflow-hidden">
                      <CollapsibleTrigger className="w-full">
                        <div className="flex items-center justify-between p-3 hover:bg-muted/30 transition-colors">
                          <div className="flex items-center gap-2">
                            <MessageSquare className="h-4 w-4 text-emerald-500" />
                            <span className="font-medium text-sm">Prompts</span>
                            {selectedPromptId && (
                              <Badge variant="secondary" className="text-xs">
                                {customPrompts.find(p => p.id === selectedPromptId)?.name}
                              </Badge>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                                <Button variant="ghost" size="icon" className="h-7 w-7">
                                  <FolderOpen className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end" className="w-48">
                                {customPrompts.length === 0 ? (
                                  <div className="px-2 py-1.5 text-xs text-muted-foreground">
                                    No saved prompts
                                  </div>
                                ) : (
                                  customPrompts.map(prompt => (
                                    <DropdownMenuItem
                                      key={prompt.id}
                                      onClick={() => setSelectedPromptId(prompt.id)}
                                      className="flex items-center justify-between"
                                    >
                                      <span>{prompt.name}</span>
                                      <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-5 w-5"
                                        onClick={(e) => {
                                          e.stopPropagation()
                                          removeCustomPrompt(prompt.id)
                                          if (selectedPromptId === prompt.id) setSelectedPromptId(null)
                                          toast.success('Prompt deleted')
                                        }}
                                      >
                                        <Trash2 className="h-3 w-3" />
                                      </Button>
                                    </DropdownMenuItem>
                                  ))
                                )}
                              </DropdownMenuContent>
                            </DropdownMenu>
                            {promptSectionOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                          </div>
                        </div>
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        <div className="p-3 pt-0 space-y-3 border-t border-border/30">
                          <div className="space-y-2 pt-3">
                            <Label className="text-xs">Prompt name</Label>
                            <Input
                              value={promptName}
                              onChange={(e) => setPromptName(e.target.value)}
                              placeholder="Ex: Horror, Romance..."
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label className="text-xs flex items-center gap-1">
                              <span className="w-2 h-2 rounded-full bg-emerald-500" />
                              Positive prompts
                            </Label>
                            <Textarea
                              value={positivePrompt}
                              onChange={(e) => setPositivePrompt(e.target.value)}
                              placeholder="What the AI should include..."
                              className="min-h-[60px] max-h-[100px] resize-none text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label className="text-xs flex items-center gap-1">
                              <span className="w-2 h-2 rounded-full bg-rose-500" />
                              Negative prompts
                            </Label>
                            <Textarea
                              value={negativePrompt}
                              onChange={(e) => setNegativePrompt(e.target.value)}
                              placeholder="What the AI should avoid (priority)..."
                              className="min-h-[60px] max-h-[100px] resize-none text-sm"
                            />
                          </div>
                          <div className="flex gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              className="flex-1 h-8 text-xs"
                              onClick={clearPromptForm}
                            >
                              <X className="h-3 w-3 mr-1" />
                              Clear
                            </Button>
                            <Button
                              size="sm"
                              className="flex-1 h-8 text-xs bg-gradient-to-r from-emerald-600 to-teal-600 border-0"
                              onClick={savePrompt}
                              disabled={!promptName.trim()}
                            >
                              <Save className="h-3 w-3 mr-1" />
                              Save
                            </Button>
                          </div>
                        </div>
                      </CollapsibleContent>
                    </div>
                  </Collapsible>

                  {/* Theme Section */}
                  <Collapsible open={themeSectionOpen} onOpenChange={setThemeSectionOpen}>
                    <div className="rounded-xl border border-border/50 bg-background/50 overflow-hidden">
                      <CollapsibleTrigger className="w-full">
                        <div className="flex items-center justify-between p-3 hover:bg-muted/30 transition-colors">
                          <div className="flex items-center gap-2">
                            <Palette className="h-4 w-4 text-amber-500" />
                            <span className="font-medium text-sm">Theme</span>
                            {selectedThemeId && (
                              <Badge variant="secondary" className="text-xs">
                                {customThemes.find(t => t.id === selectedThemeId)?.name}
                              </Badge>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                                <Button variant="ghost" size="icon" className="h-7 w-7">
                                  <FolderOpen className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end" className="w-48">
                                {customThemes.length === 0 ? (
                                  <div className="px-2 py-1.5 text-xs text-muted-foreground">
                                    No saved themes
                                  </div>
                                ) : (
                                  customThemes.map(theme => (
                                    <DropdownMenuItem
                                      key={theme.id}
                                      onClick={() => setSelectedThemeId(theme.id)}
                                      className="flex items-center justify-between"
                                    >
                                      <span>{theme.name}</span>
                                      <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-5 w-5"
                                        onClick={(e) => {
                                          e.stopPropagation()
                                          removeCustomTheme(theme.id)
                                          if (selectedThemeId === theme.id) setSelectedThemeId(null)
                                          toast.success('Theme deleted')
                                        }}
                                      >
                                        <Trash2 className="h-3 w-3" />
                                      </Button>
                                    </DropdownMenuItem>
                                  ))
                                )}
                              </DropdownMenuContent>
                            </DropdownMenu>
                            {themeSectionOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                          </div>
                        </div>
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        <div className="p-3 pt-0 space-y-3 border-t border-border/30">
                          <div className="space-y-2 pt-3">
                            <Label className="text-xs">Theme name</Label>
                            <Input
                              value={themeName}
                              onChange={(e) => setThemeName(e.target.value)}
                              placeholder="Ex: Medieval, Futuristic..."
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label className="text-xs">Description</Label>
                            <Textarea
                              value={themeDescription}
                              onChange={(e) => setThemeDescription(e.target.value)}
                              placeholder="Describe the story theme..."
                              className="min-h-[60px] max-h-[100px] resize-none text-sm"
                            />
                          </div>
                          <div className="flex gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              className="flex-1 h-8 text-xs"
                              onClick={clearThemeForm}
                            >
                              <X className="h-3 w-3 mr-1" />
                              Clear
                            </Button>
                            <Button
                              size="sm"
                              className="flex-1 h-8 text-xs bg-gradient-to-r from-amber-600 to-orange-600 border-0"
                              onClick={saveTheme}
                              disabled={!themeName.trim()}
                            >
                              <Save className="h-3 w-3 mr-1" />
                              Save
                            </Button>
                          </div>
                        </div>
                      </CollapsibleContent>
                    </div>
                  </Collapsible>

                  {/* Generation Parameters */}
                  <Collapsible open={paramsSectionOpen} onOpenChange={setParamsSectionOpen}>
                    <div className="rounded-xl border border-border/50 bg-background/50 overflow-hidden">
                      <CollapsibleTrigger className="w-full">
                        <div className="flex items-center justify-between p-3 hover:bg-muted/30 transition-colors">
                          <div className="flex items-center gap-2">
                            <Settings2 className="h-4 w-4 text-slate-500" />
                            <span className="font-medium text-sm">Advanced Settings</span>
                          </div>
                          {paramsSectionOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                        </div>
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        <div className="p-3 pt-0 space-y-4 border-t border-border/30">
                          <div className="space-y-3 pt-3">
                            <div className="flex justify-between">
                              <Label className="text-xs flex items-center gap-1">
                                <Gauge className="h-3 w-3" />
                                Temperature
                              </Label>
                              <span className="text-xs text-muted-foreground font-mono">{temperature.toFixed(2)}</span>
                            </div>
                            <input
                              type="range"
                              min="0"
                              max="2"
                              step="0.05"
                              value={temperature}
                              onChange={(e) => setTemperature(parseFloat(e.target.value))}
                              className="w-full h-1.5 rounded-full appearance-none bg-muted cursor-pointer"
                            />
                          </div>
                          
                          <div className="space-y-3">
                            <div className="flex justify-between">
                              <Label className="text-xs flex items-center gap-1">
                                <Target className="h-3 w-3" />
                                Max Tokens
                              </Label>
                              <span className="text-xs text-muted-foreground font-mono">{maxTokens}</span>
                            </div>
                            <input
                              type="range"
                              min="50"
                              max="2000"
                              step="50"
                              value={maxTokens}
                              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                              className="w-full h-1.5 rounded-full appearance-none bg-muted cursor-pointer"
                            />
                          </div>
                        </div>
                      </CollapsibleContent>
                    </div>
                  </Collapsible>

                  {/* Generate Button */}
                  <Button 
                    className="w-full bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-700 hover:to-fuchsia-700 border-0 h-11" 
                    onClick={handleGenerateText}
                    disabled={isGenerating || !isModelLoaded}
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        {lastGeneratedContent || hasGeneratedContent ? (
                          <>
                            <ArrowRight className="h-4 w-4 mr-2" />
                            Continue
                          </>
                        ) : (
                          <>
                            <Sparkles className="h-4 w-4 mr-2" />
                            Generate
                          </>
                        )}
                      </>
                    )}
                  </Button>
                </div>
              </ScrollArea>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* New Chapter Dialog */}
      <Dialog open={newChapterDialogOpen} onOpenChange={setNewChapterDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-violet-500" />
              New Chapter
            </DialogTitle>
            <DialogDescription>Create a new chapter for your story.</DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Label>Chapter Title</Label>
            <Input
              value={newChapterTitle}
              onChange={(e) => setNewChapterTitle(e.target.value)}
              placeholder="Enter chapter title..."
              className="mt-2"
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setNewChapterDialogOpen(false)}>Cancel</Button>
            <Button onClick={handleCreateChapter} className="bg-gradient-to-r from-violet-600 to-fuchsia-600 border-0">Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Chapter Dialog */}
      <AlertDialog open={deleteChapterDialogOpen} onOpenChange={setDeleteChapterDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Chapter</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{chapterToDelete?.title}"? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteChapter} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Add Character Dialog */}
      <Dialog open={addCharacterDialogOpen} onOpenChange={setAddCharacterDialogOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <User className="h-5 w-5 text-violet-500" />
              Add a Character
            </DialogTitle>
            <DialogDescription>
              Create a new character for your writing session.
            </DialogDescription>
          </DialogHeader>
          
          <div className="flex gap-2 mt-4">
            <Button
              variant={addCharacterTab === 'manual' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setAddCharacterTab('manual')}
              className="flex-1"
            >
              <User className="h-4 w-4 mr-1" />
              Manual
            </Button>
            <Button
              variant={addCharacterTab === 'ai' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setAddCharacterTab('ai')}
              className="flex-1"
            >
              <Brain className="h-4 w-4 mr-1" />
              AI Generate
            </Button>
          </div>
          
          {addCharacterTab === 'manual' ? (
            <div className="space-y-4 mt-4">
              <div className="space-y-2">
                <Label>Character name *</Label>
                <Input
                  value={characterName}
                  onChange={(e) => setCharacterName(e.target.value)}
                  placeholder="Ex: Elena, Marcus..."
                />
              </div>
              <div className="space-y-2">
                <Label>Description</Label>
                <Textarea
                  value={characterDescription}
                  onChange={(e) => setCharacterDescription(e.target.value)}
                  placeholder="Describe the character..."
                  className="min-h-[100px] max-h-[150px] resize-none"
                />
              </div>
            </div>
          ) : (
            <div className="space-y-4 mt-4">
              <div className="p-3 rounded-lg bg-violet-500/10 border border-violet-500/20">
                <p className="text-xs text-muted-foreground">
                  Describe the character you want to create. The AI will generate a name and description.
                </p>
              </div>
              <div className="space-y-2">
                <Label>Instructions for AI</Label>
                <Textarea
                  value={aiInstructions}
                  onChange={(e) => setAiInstructions(e.target.value)}
                  placeholder="Ex: A mysterious warrior..."
                  className="min-h-[100px] max-h-[150px] resize-none"
                />
              </div>
              {isGeneratingCharacter && (
                <div className="flex items-center justify-center gap-2 py-2">
                  <Loader2 className="h-4 w-4 animate-spin text-violet-500" />
                  <span className="text-sm text-muted-foreground">Generating...</span>
                </div>
              )}
            </div>
          )}
          
          <DialogFooter className="mt-4">
            <Button variant="outline" onClick={() => {
              setAddCharacterDialogOpen(false)
              setCharacterName('')
              setCharacterDescription('')
              setAiInstructions('')
            }}>
              Cancel
            </Button>
            {addCharacterTab === 'manual' ? (
              <Button 
                onClick={addCharacterManually}
                disabled={!characterName.trim()}
                className="bg-gradient-to-r from-violet-600 to-fuchsia-600 border-0"
              >
                <Check className="h-4 w-4 mr-1.5" />
                Add
              </Button>
            ) : (
              <Button 
                onClick={generateCharacterWithAI}
                disabled={!aiInstructions.trim() || isGeneratingCharacter}
                className="bg-gradient-to-r from-violet-600 to-fuchsia-600 border-0"
              >
                {isGeneratingCharacter ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4 mr-1.5" />
                    Generate
                  </>
                )}
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
