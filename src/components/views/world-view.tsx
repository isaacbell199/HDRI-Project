'use client'

import { useState, useEffect, useRef } from 'react'
import { useStore } from '@/lib/store'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { toast } from 'sonner'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Save,
  Sparkles,
  Palette,
  Pen,
  Wand2,
  Users,
  MapPin,
  BookOpen,
  Plus,
  X,
  Check,
  ChevronDown,
  ChevronUp,
  Send,
  Clapperboard,
  Film,
  Settings2,
  Gauge
} from 'lucide-react'
import { GenerationPresetSelector } from '@/components/generation-preset-selector'
import {
  createCharacter as tauriCreateCharacter,
  updateCharacter as tauriUpdateCharacter,
  deleteCharacter as tauriDeleteCharacter,
  getCharacters as tauriGetCharacters,
  createLocation as tauriCreateLocation,
  updateLocation as tauriUpdateLocation,
  deleteLocation as tauriDeleteLocation,
  getLocations as tauriGetLocations,
  createLoreNote as tauriCreateLoreNote,
  updateLoreNote as tauriUpdateLoreNote,
  deleteLoreNote as tauriDeleteLoreNote,
  getLoreNotes as tauriGetLoreNotes,
  isTauri,
  type Character as DBCharacter,
  type Location as DBLocation,
  type LoreNote as DBLoreNote,
} from '@/lib/tauri-api'

// ============================================================================
// Predefined Options
// ============================================================================

const CATEGORIES = [
  { id: 'fantasy', name: 'Fantasy', description: 'Magical worlds with supernatural elements' },
  { id: 'scifi', name: 'Science Fiction', description: 'Future technology and space exploration' },
  { id: 'romance', name: 'Romance', description: 'Love stories and emotional relationships' },
  { id: 'mystery', name: 'Mystery', description: 'Suspense, detective work, and puzzles' },
  { id: 'horror', name: 'Horror', description: 'Fear, suspense, and the supernatural' },
  { id: 'thriller', name: 'Thriller', description: 'Tension, excitement, and suspense' },
  { id: 'adventure', name: 'Adventure', description: 'Exciting journeys and exploration' },
  { id: 'historical', name: 'Historical', description: 'Set in a specific historical period' },
  { id: 'literary', name: 'Literary Fiction', description: 'Character-driven, artistic prose' },
  { id: 'custom', name: 'Custom...', description: 'Create your own category' },
]

const TONES = [
  { id: 'dramatic', name: 'Dramatic', description: 'Intense, emotional, and serious' },
  { id: 'humorous', name: 'Humorous', description: 'Light-hearted, funny, and witty' },
  { id: 'dark', name: 'Dark', description: 'Grim, serious, and ominous' },
  { id: 'whimsical', name: 'Whimsical', description: 'Playful, fanciful, and imaginative' },
  { id: 'romantic', name: 'Romantic', description: 'Passionate, tender, and emotional' },
  { id: 'suspenseful', name: 'Suspenseful', description: 'Tense, anxious, and thrilling' },
  { id: 'melancholic', name: 'Melancholic', description: 'Sad, reflective, and pensive' },
  { id: 'inspiring', name: 'Inspiring', description: 'Uplifting, motivational, and hopeful' },
  { id: 'satirical', name: 'Satirical', description: 'Ironic, critical, and mocking' },
  { id: 'custom', name: 'Custom...', description: 'Define your own tone' },
]

const WRITING_STYLES = [
  { id: 'descriptive', name: 'Descriptive', description: 'Rich sensory details and imagery' },
  { id: 'minimalist', name: 'Minimalist', description: 'Clean, simple, and direct prose' },
  { id: 'poetic', name: 'Poetic', description: 'Lyrical, rhythmic, and artistic language' },
  { id: 'conversational', name: 'Conversational', description: 'Casual, friendly, and approachable' },
  { id: 'formal', name: 'Formal', description: 'Professional, structured, and precise' },
  { id: 'stream', name: 'Stream of Consciousness', description: 'Inner thoughts and raw mental flow' },
  { id: 'cinematic', name: 'Cinematic', description: 'Visual, scene-focused storytelling' },
  { id: 'epistolary', name: 'Epistolary', description: 'Letters, diaries, and documents' },
  { id: 'custom', name: 'Custom...', description: 'Define your own style' },
]

const THEMES = [
  { id: 'love', name: 'Love & Relationships', description: 'Romantic connections and bonds' },
  { id: 'power', name: 'Power & Corruption', description: 'Authority, influence, and their abuse' },
  { id: 'identity', name: 'Identity & Self-Discovery', description: 'Finding who you truly are' },
  { id: 'redemption', name: 'Redemption', description: 'Making amends and finding forgiveness' },
  { id: 'survival', name: 'Survival', description: 'Overcoming impossible odds' },
  { id: 'betrayal', name: 'Betrayal', description: 'Trust broken and its consequences' },
  { id: 'coming-of-age', name: 'Coming of Age', description: 'Growing up and maturing' },
  { id: 'justice', name: 'Justice & Morality', description: 'Right vs wrong, fairness' },
  { id: 'freedom', name: 'Freedom & Oppression', description: 'Liberty against control' },
  { id: 'custom', name: 'Custom...', description: 'Define your own theme' },
]

// ============================================================================
// Types
// ============================================================================

interface WorldStudioData {
  category: { id: string; name: string; description: string }
  categoryCustom: { title: string; description: string }
  tone: { id: string; name: string; description: string }
  toneCustom: { title: string; description: string }
  writingStyle: { id: string; name: string; description: string }
  writingStyleCustom: { title: string; description: string }
  theme: { id: string; name: string; description: string }
  themeCustom: { title: string; description: string }
  prompts: { positive: string; negative: string }
  characters: Array<{ id: string; title: string; description: string }>
  locations: Array<{ id: string; description: string }>
  lore: { storyBeginning: string }
  preset: string // Generation preset ID
}

const defaultWorldData: WorldStudioData = {
  category: { id: '', name: '', description: '' },
  categoryCustom: { title: '', description: '' },
  tone: { id: '', name: '', description: '' },
  toneCustom: { title: '', description: '' },
  writingStyle: { id: '', name: '', description: '' },
  writingStyleCustom: { title: '', description: '' },
  theme: { id: '', name: '', description: '' },
  themeCustom: { title: '', description: '' },
  prompts: { positive: '', negative: '' },
  characters: [],
  locations: [],
  lore: { storyBeginning: '' },
  preset: 'classic-story',
}

// ============================================================================
// Component
// ============================================================================

export function WorldView() {
  const { 
    currentProject, 
    setCurrentView, 
    selectedPresetId, 
    restorePresetFromId,
    // Store setters for syncing
    setProjectCategory,
    setProjectTone,
    setProjectWritingStyle,
    setProjectTheme,
    setProjectPreset,
    setProjectCustomCategory,
    setProjectCustomTone,
    setProjectCustomStyle,
    setProjectCustomTheme,
    // C3 FIX: Add prompt sync functions
    addCustomPrompt,
    setSelectedPromptId,
  } = useStore()
  
  const [worldData, setWorldData] = useState<WorldStudioData>(defaultWorldData)
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    category: true,
    tone: true,
    writingStyle: true,
    theme: true,
    prompts: true,
    characters: true,
    locations: true,
    lore: true,
    generationPresets: true,
  })
  const [isSaving, setIsSaving] = useState(false)
  
  // Load saved world data on mount
  // L6 fix: Store currentProjectId for validation during async saves
  const currentProjectIdRef = useRef<string | null>(null)
  useEffect(() => {
    currentProjectIdRef.current = currentProject?.id || null
  }, [currentProject?.id])

  async function loadWorldData() {
    if (!currentProject?.id) return

    // Step 1: Load base data from localStorage first
    let baseData = { ...defaultWorldData }
    let loadedFromCache = false

    const saved = localStorage.getItem(`world_studio_${currentProject.id}`)
    if (saved) {
      try {
        const parsed = JSON.parse(saved)
        baseData = { ...defaultWorldData, ...parsed }
        loadedFromCache = true
      } catch (e) {
        console.error('Failed to load world data from localStorage:', e)
      }
    }

    if (!loadedFromCache) {
      const currentSaved = localStorage.getItem('world_studio_current')
      if (currentSaved) {
        try {
          const parsed = JSON.parse(currentSaved)
          baseData = { ...defaultWorldData, ...parsed }
          loadedFromCache = true
        } catch (e) {
          console.error('Failed to load current world data:', e)
        }
      }
    }

    // Restore preset from cached data
    if (loadedFromCache && baseData.preset && baseData.preset !== 'none') {
      restorePresetFromId(baseData.preset)
    }

    // Step 2: If in Tauri, also load from database and merge DB data over localStorage
    if (isTauri()) {
      try {
        const [dbCharacters, dbLocations, dbLoreNotes] = await Promise.all([
          tauriGetCharacters(currentProject.id),
          tauriGetLocations(currentProject.id),
          tauriGetLoreNotes(currentProject.id),
        ])

        // Convert DB characters to local format
        const characters = dbCharacters.map((c: DBCharacter) => ({
          id: c.id,
          title: c.name,
          description: c.background || '',
        }))

        // Convert DB locations to local format
        // DB has name + description; local format stores "Name: description" in a single string
        const locations = dbLocations.map((l: DBLocation) => ({
          id: l.id,
          description: l.description
            ? `${l.name}: ${l.description}`
            : l.name,
        }))

        // Convert DB lore notes to local format
        // Find the "Story Beginning" lore note specifically
        const storyBeginningNote = dbLoreNotes.find((n: DBLoreNote) => n.title === 'Story Beginning')
        const storyBeginning = storyBeginningNote?.content || baseData.lore.storyBeginning

        // If DB has data, prefer it over localStorage for these sections
        const mergedData: WorldStudioData = {
          ...baseData,
          characters: characters.length > 0 ? characters : baseData.characters,
          locations: locations.length > 0 ? locations : baseData.locations,
          lore: { storyBeginning },
        }

        setWorldData(mergedData)
        return
      } catch (e) {
        console.error('Failed to load world data from database:', e)
        // Fall through to use localStorage data
      }
    }

    // Fallback: use localStorage data (or defaults)
    setWorldData(baseData)
  }

  useEffect(() => {
    if (currentProject?.id) {
      loadWorldData()
    }
  }, [currentProject?.id])
  
  async function handleSave() {
    if (!currentProject) return
    // Bug 4 fix: Check project ID hasn't changed AND not already saving
    // Project ID check must come before isSaving check to prevent race condition
    // where isSaving state hasn't committed yet but project has already changed
    if (currentProjectIdRef.current !== currentProject.id) return
    if (isSaving) return  // M9 fix: prevent duplicate creation on double-click
    setIsSaving(true)
    
    try {
      // Include current preset from store in worldData
      const dataToSave = { ...worldData, preset: selectedPresetId }
      
      // Save to localStorage for quick access
      localStorage.setItem(`world_studio_${currentProject.id}`, JSON.stringify(dataToSave))
      localStorage.setItem('world_studio_current', JSON.stringify(dataToSave))
      
      // Persist to Tauri database (SQLite) using UPSERT logic
      if (isTauri()) {
        // Fetch existing data from database to determine what to create vs update vs delete
        const [existingCharacters, existingLocations, existingLoreNotes] = await Promise.all([
          tauriGetCharacters(currentProject.id),
          tauriGetLocations(currentProject.id),
          tauriGetLoreNotes(currentProject.id),
        ])

        // --- Characters UPSERT ---
        // Build a set of names from current worldData for deletion detection
        const localCharacterNames = new Set(
          worldData.characters
            .filter(c => c.title.trim())
            .map(c => c.title.trim())
        )

        for (const char of worldData.characters) {
          if (!char.title.trim()) continue

          // Find existing character by name
          const existing = existingCharacters.find(
            (ec: DBCharacter) => ec.name.trim() === char.title.trim()
          )

          if (existing) {
            // Update existing character
            try {
              await tauriUpdateCharacter(existing.id, {
                ...existing,
                name: char.title,
                role: existing.role || 'Character',
                background: char.description || existing.background,
              })
            } catch (e) {
              console.error('Failed to update character:', char.title, e)
            }
          } else {
            // Create new character
            try {
              await tauriCreateCharacter({
                projectId: currentProject.id,
                name: char.title,
                role: 'Character',
                background: char.description || undefined,
              })
            } catch (e) {
              console.error('Failed to create character:', char.title, e)
            }
          }
        }

        // Delete characters from DB that are no longer in the local worldData
        for (const ec of existingCharacters) {
          if (!localCharacterNames.has(ec.name.trim())) {
            try {
              await tauriDeleteCharacter(ec.id)
            } catch (e) {
              console.error('Failed to delete character:', ec.name, e)
            }
          }
        }

        // --- Locations UPSERT ---
        // Parse local location names for matching
        const localLocationNames = new Set<string>()
        const localLocationData = worldData.locations
          .filter(l => l.description.trim())
          .map(l => {
            const parts = l.description.split(':')
            const name = parts.length > 1 ? parts[0].trim() : l.description.slice(0, 50)
            const description = parts.length > 1 ? parts.slice(1).join(':').trim() : l.description
            localLocationNames.add(name || 'Unnamed Location')
            return { name: name || 'Unnamed Location', description }
          })

        for (const loc of localLocationData) {
          // Find existing location by name
          const existing = existingLocations.find(
            (el: DBLocation) => el.name.trim() === loc.name
          )

          if (existing) {
            // Update existing location
            try {
              await tauriUpdateLocation(existing.id, {
                ...existing,
                name: loc.name,
                description: loc.description,
              })
            } catch (e) {
              console.error('Failed to update location:', loc.name, e)
            }
          } else {
            // Create new location
            try {
              await tauriCreateLocation({
                projectId: currentProject.id,
                name: loc.name,
                description: loc.description,
              })
            } catch (e) {
              console.error('Failed to create location:', loc.name, e)
            }
          }
        }

        // Delete locations from DB that are no longer in the local worldData
        for (const el of existingLocations) {
          if (!localLocationNames.has(el.name.trim())) {
            try {
              await tauriDeleteLocation(el.id)
            } catch (e) {
              console.error('Failed to delete location:', el.name, e)
            }
          }
        }

        // --- Lore Notes UPSERT ---
        const LORE_TITLE = 'Story Beginning'
        const existingStoryNote = existingLoreNotes.find(
          (n: DBLoreNote) => n.title === LORE_TITLE
        )

        if (worldData.lore.storyBeginning.trim()) {
          if (existingStoryNote) {
            // Update existing lore note
            try {
              await tauriUpdateLoreNote(existingStoryNote.id, {
                ...existingStoryNote,
                title: LORE_TITLE,
                content: worldData.lore.storyBeginning,
              })
            } catch (e) {
              console.error('Failed to update lore note:', LORE_TITLE, e)
            }
          } else {
            // Create new lore note
            try {
              await tauriCreateLoreNote({
                projectId: currentProject.id,
                title: LORE_TITLE,
                content: worldData.lore.storyBeginning,
              })
            } catch (e) {
              console.error('Failed to create lore note:', LORE_TITLE, e)
            }
          }
        } else {
          // Story beginning was cleared — delete the lore note if it exists
          if (existingStoryNote) {
            try {
              await tauriDeleteLoreNote(existingStoryNote.id)
            } catch (e) {
              console.error('Failed to delete lore note:', LORE_TITLE, e)
            }
          }
        }
      }
      
      toast.success('World studio saved!', {
        description: 'Your production settings are ready for the editor.'
      })
    } catch (error) {
      toast.error('Failed to save')
      console.error(error)
    } finally {
      setIsSaving(false)
    }
  }
  
  async function handleImportToEditor() {
    // Sync all settings to global store before importing
    syncSettingsToStore()
    
    // M3 FIX: Await the save to ensure data is persisted before navigating
    await handleSave()
    
    // Navigate to editor
    setCurrentView('editor')
    toast.success('Settings imported to editor!', {
      description: 'Your world studio configuration is now active.'
    })
  }
  
  // Sync all World Studio settings to the global store
  function syncSettingsToStore() {
    // Sync category
    if (worldData.category.id === 'custom' && worldData.categoryCustom.title) {
      setProjectCategory('custom')
      setProjectCustomCategory({
        id: `custom-category-${Date.now()}`,
        name: worldData.categoryCustom.title,
        description: worldData.categoryCustom.description,
        isCustom: true,
      })
    } else if (worldData.category.id) {
      setProjectCategory(worldData.category.id)
      setProjectCustomCategory(null)
    }
    
    // Sync tone
    if (worldData.tone.id === 'custom' && worldData.toneCustom.title) {
      setProjectTone('custom')
      setProjectCustomTone({
        id: `custom-tone-${Date.now()}`,
        name: worldData.toneCustom.title,
        description: worldData.toneCustom.description,
        isCustom: true,
      })
    } else if (worldData.tone.id) {
      setProjectTone(worldData.tone.id)
      setProjectCustomTone(null)
    }
    
    // Sync writing style
    if (worldData.writingStyle.id === 'custom' && worldData.writingStyleCustom.title) {
      setProjectWritingStyle('custom')
      setProjectCustomStyle({
        id: `custom-style-${Date.now()}`,
        name: worldData.writingStyleCustom.title,
        description: worldData.writingStyleCustom.description,
        isCustom: true,
      })
    } else if (worldData.writingStyle.id) {
      setProjectWritingStyle(worldData.writingStyle.id)
      setProjectCustomStyle(null)
    }
    
    // Sync theme
    if (worldData.theme.id === 'custom' && worldData.themeCustom.title) {
      setProjectTheme('custom')
      setProjectCustomTheme({
        id: `custom-theme-${Date.now()}`,
        name: worldData.themeCustom.title,
        description: worldData.themeCustom.description,
        isCustom: true,
      })
    } else if (worldData.theme.id) {
      setProjectTheme(worldData.theme.id)
      setProjectCustomTheme(null)
    }
    
    // C3 FIX: Sync prompts (positive/negative) to the global store
    // Previously World View prompts were stored in localStorage but never synced to customPrompts,
    // causing them to be lost when transitioning to the Editor.
    if (worldData.prompts.positive.trim() || worldData.prompts.negative.trim()) {
      const promptId = `world-prompt-${currentProject?.id || Date.now()}`
      addCustomPrompt({
        id: promptId,
        name: 'World Studio Prompt',
        positivePrompt: worldData.prompts.positive,
        negativePrompt: worldData.prompts.negative,
        createdAt: new Date().toISOString(),
      })
      setSelectedPromptId(promptId)
    }
    
    // Sync preset
    setProjectPreset(selectedPresetId)
  }
  
  function toggleSection(section: string) {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }))
  }
  
  function updateCategory(id: string) {
    const selected = CATEGORIES.find(c => c.id === id)
    if (selected) {
      setWorldData(prev => ({
        ...prev,
        category: { id: selected.id, name: selected.name, description: selected.description }
      }))
    }
  }
  
  function updateTone(id: string) {
    const selected = TONES.find(t => t.id === id)
    if (selected) {
      setWorldData(prev => ({
        ...prev,
        tone: { id: selected.id, name: selected.name, description: selected.description }
      }))
    }
  }
  
  function updateWritingStyle(id: string) {
    const selected = WRITING_STYLES.find(s => s.id === id)
    if (selected) {
      setWorldData(prev => ({
        ...prev,
        writingStyle: { id: selected.id, name: selected.name, description: selected.description }
      }))
    }
  }
  
  function updateTheme(id: string) {
    const selected = THEMES.find(t => t.id === id)
    if (selected) {
      setWorldData(prev => ({
        ...prev,
        theme: { id: selected.id, name: selected.name, description: selected.description }
      }))
    }
  }
  
  function addCharacter() {
    setWorldData(prev => ({
      ...prev,
      characters: [...prev.characters, { id: Date.now().toString(), title: '', description: '' }]
    }))
  }
  
  function updateCharacter(id: string, field: 'title' | 'description', value: string) {
    setWorldData(prev => ({
      ...prev,
      characters: prev.characters.map(c => c.id === id ? { ...c, [field]: value } : c)
    }))
  }
  
  function removeCharacter(id: string) {
    setWorldData(prev => ({
      ...prev,
      characters: prev.characters.filter(c => c.id !== id)
    }))
  }
  
  function addLocation() {
    setWorldData(prev => ({
      ...prev,
      locations: [...prev.locations, { id: Date.now().toString(), description: '' }]
    }))
  }
  
  function updateLocation(id: string, description: string) {
    setWorldData(prev => ({
      ...prev,
      locations: prev.locations.map(l => l.id === id ? { ...l, description } : l)
    }))
  }
  
  function removeLocation(id: string) {
    setWorldData(prev => ({
      ...prev,
      locations: prev.locations.filter(l => l.id !== id)
    }))
  }

  if (!currentProject) {
    return (
      <div className="flex-1 flex items-center justify-center bg-gradient-to-br from-background via-background to-muted/20">
        <div className="text-center">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="relative"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-violet-500/20 to-fuchsia-500/20 blur-3xl rounded-full" />
            <Clapperboard className="h-24 w-24 text-muted-foreground/30 mx-auto mb-6 relative" />
          </motion.div>
          <h2 className="text-2xl font-bold mb-2">No Project Selected</h2>
          <p className="text-muted-foreground">Select a project to access the World Studio</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col bg-gradient-to-br from-background via-background to-muted/10">
      {/* Header */}
      <header className="h-16 border-b border-border/40 flex items-center justify-between px-6 bg-gradient-to-r from-background/80 to-background/40 backdrop-blur-sm">
        <div className="flex items-center gap-4">
          <motion.div 
            initial={{ rotate: -10 }}
            animate={{ rotate: 0 }}
            className="p-2 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500"
          >
            <Clapperboard className="h-5 w-5 text-white" />
          </motion.div>
          <div>
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text">
              World Studio
            </h1>
            <p className="text-xs text-muted-foreground">Production settings for your story</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="gap-1.5 py-1">
            <Film className="h-3.5 w-3.5" />
            {currentProject.name}
          </Badge>
        </div>
      </header>

      {/* Content */}
      <ScrollArea className="flex-1">
        <div className="p-6 max-w-5xl mx-auto space-y-4">
          
          {/* Category Section */}
          <CollapsibleSection
            title="Category"
            icon={<Palette className="h-4 w-4" />}
            color="violet"
            expanded={expandedSections.category}
            onToggle={() => toggleSection('category')}
          >
            <div className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {CATEGORIES.map((cat) => (
                  <motion.button
                    key={cat.id}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => updateCategory(cat.id)}
                    className={cn(
                      "px-4 py-2 rounded-xl text-sm font-medium transition-all",
                      "border-2 hover:border-violet-400/50",
                      worldData.category.id === cat.id
                        ? "border-violet-500 bg-violet-500/10 text-violet-600 dark:text-violet-400"
                        : "border-border bg-background/50 hover:bg-muted/50"
                    )}
                  >
                    {cat.name}
                  </motion.button>
                ))}
              </div>
              
              {worldData.category.id && worldData.category.id !== 'custom' && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-3 rounded-xl bg-violet-500/5 border border-violet-500/20"
                >
                  <p className="text-sm text-muted-foreground">{worldData.category.description}</p>
                </motion.div>
              )}
              
              {worldData.category.id === 'custom' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="space-y-3"
                >
                  <Input
                    placeholder="Custom category name..."
                    value={worldData.categoryCustom.title}
                    onChange={(e) => setWorldData(prev => ({
                      ...prev,
                      categoryCustom: { ...prev.categoryCustom, title: e.target.value }
                    }))}
                    className="border-violet-500/30 focus:border-violet-500"
                  />
                  <ScrollArea className="h-24 rounded-lg border border-violet-500/20 bg-background">
                    <Textarea
                      placeholder="Describe your custom category..."
                      value={worldData.categoryCustom.description}
                      onChange={(e) => setWorldData(prev => ({
                        ...prev,
                        categoryCustom: { ...prev.categoryCustom, description: e.target.value }
                      }))}
                      className="min-h-[90px] border-0 resize-none focus-visible:ring-0"
                    />
                  </ScrollArea>
                </motion.div>
              )}
            </div>
          </CollapsibleSection>

          {/* Tone Section */}
          <CollapsibleSection
            title="Tone"
            icon={<Wand2 className="h-4 w-4" />}
            color="amber"
            expanded={expandedSections.tone}
            onToggle={() => toggleSection('tone')}
          >
            <div className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {TONES.map((tone) => (
                  <motion.button
                    key={tone.id}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => updateTone(tone.id)}
                    className={cn(
                      "px-4 py-2 rounded-xl text-sm font-medium transition-all",
                      "border-2 hover:border-amber-400/50",
                      worldData.tone.id === tone.id
                        ? "border-amber-500 bg-amber-500/10 text-amber-600 dark:text-amber-400"
                        : "border-border bg-background/50 hover:bg-muted/50"
                    )}
                  >
                    {tone.name}
                  </motion.button>
                ))}
              </div>
              
              {worldData.tone.id && worldData.tone.id !== 'custom' && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-3 rounded-xl bg-amber-500/5 border border-amber-500/20"
                >
                  <p className="text-sm text-muted-foreground">{worldData.tone.description}</p>
                </motion.div>
              )}
              
              {worldData.tone.id === 'custom' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="space-y-3"
                >
                  <Input
                    placeholder="Custom tone name..."
                    value={worldData.toneCustom.title}
                    onChange={(e) => setWorldData(prev => ({
                      ...prev,
                      toneCustom: { ...prev.toneCustom, title: e.target.value }
                    }))}
                    className="border-amber-500/30 focus:border-amber-500"
                  />
                  <ScrollArea className="h-24 rounded-lg border border-amber-500/20 bg-background">
                    <Textarea
                      placeholder="Describe your custom tone..."
                      value={worldData.toneCustom.description}
                      onChange={(e) => setWorldData(prev => ({
                        ...prev,
                        toneCustom: { ...prev.toneCustom, description: e.target.value }
                      }))}
                      className="min-h-[90px] border-0 resize-none focus-visible:ring-0"
                    />
                  </ScrollArea>
                </motion.div>
              )}
            </div>
          </CollapsibleSection>

          {/* Writing Style Section */}
          <CollapsibleSection
            title="Writing Style"
            icon={<Pen className="h-4 w-4" />}
            color="emerald"
            expanded={expandedSections.writingStyle}
            onToggle={() => toggleSection('writingStyle')}
          >
            <div className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {WRITING_STYLES.map((style) => (
                  <motion.button
                    key={style.id}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => updateWritingStyle(style.id)}
                    className={cn(
                      "px-4 py-2 rounded-xl text-sm font-medium transition-all",
                      "border-2 hover:border-emerald-400/50",
                      worldData.writingStyle.id === style.id
                        ? "border-emerald-500 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                        : "border-border bg-background/50 hover:bg-muted/50"
                    )}
                  >
                    {style.name}
                  </motion.button>
                ))}
              </div>
              
              {worldData.writingStyle.id && worldData.writingStyle.id !== 'custom' && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-3 rounded-xl bg-emerald-500/5 border border-emerald-500/20"
                >
                  <p className="text-sm text-muted-foreground">{worldData.writingStyle.description}</p>
                </motion.div>
              )}
              
              {worldData.writingStyle.id === 'custom' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="space-y-3"
                >
                  <Input
                    placeholder="Custom writing style name..."
                    value={worldData.writingStyleCustom.title}
                    onChange={(e) => setWorldData(prev => ({
                      ...prev,
                      writingStyleCustom: { ...prev.writingStyleCustom, title: e.target.value }
                    }))}
                    className="border-emerald-500/30 focus:border-emerald-500"
                  />
                  <ScrollArea className="h-24 rounded-lg border border-emerald-500/20 bg-background">
                    <Textarea
                      placeholder="Describe your custom writing style..."
                      value={worldData.writingStyleCustom.description}
                      onChange={(e) => setWorldData(prev => ({
                        ...prev,
                        writingStyleCustom: { ...prev.writingStyleCustom, description: e.target.value }
                      }))}
                      className="min-h-[90px] border-0 resize-none focus-visible:ring-0"
                    />
                  </ScrollArea>
                </motion.div>
              )}
            </div>
          </CollapsibleSection>

          {/* Theme Section */}
          <CollapsibleSection
            title="Theme"
            icon={<Sparkles className="h-4 w-4" />}
            color="rose"
            expanded={expandedSections.theme}
            onToggle={() => toggleSection('theme')}
          >
            <div className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {THEMES.map((theme) => (
                  <motion.button
                    key={theme.id}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => updateTheme(theme.id)}
                    className={cn(
                      "px-4 py-2 rounded-xl text-sm font-medium transition-all",
                      "border-2 hover:border-rose-400/50",
                      worldData.theme.id === theme.id
                        ? "border-rose-500 bg-rose-500/10 text-rose-600 dark:text-rose-400"
                        : "border-border bg-background/50 hover:bg-muted/50"
                    )}
                  >
                    {theme.name}
                  </motion.button>
                ))}
              </div>
              
              {worldData.theme.id && worldData.theme.id !== 'custom' && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-3 rounded-xl bg-rose-500/5 border border-rose-500/20"
                >
                  <p className="text-sm text-muted-foreground">{worldData.theme.description}</p>
                </motion.div>
              )}
              
              {worldData.theme.id === 'custom' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="space-y-3"
                >
                  <Input
                    placeholder="Custom theme name..."
                    value={worldData.themeCustom.title}
                    onChange={(e) => setWorldData(prev => ({
                      ...prev,
                      themeCustom: { ...prev.themeCustom, title: e.target.value }
                    }))}
                    className="border-rose-500/30 focus:border-rose-500"
                  />
                  <ScrollArea className="h-24 rounded-lg border border-rose-500/20 bg-background">
                    <Textarea
                      placeholder="Describe your custom theme..."
                      value={worldData.themeCustom.description}
                      onChange={(e) => setWorldData(prev => ({
                        ...prev,
                        themeCustom: { ...prev.themeCustom, description: e.target.value }
                      }))}
                      className="min-h-[90px] border-0 resize-none focus-visible:ring-0"
                    />
                  </ScrollArea>
                </motion.div>
              )}
            </div>
          </CollapsibleSection>

          {/* Prompts Section */}
          <CollapsibleSection
            title="Prompts"
            icon={<Settings2 className="h-4 w-4" />}
            color="sky"
            expanded={expandedSections.prompts}
            onToggle={() => toggleSection('prompts')}
          >
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-sm font-medium flex items-center gap-2">
                  <Badge className="bg-emerald-500 text-white border-0">+</Badge>
                  Positive Prompts
                </Label>
                <ScrollArea className="h-32 rounded-lg border border-emerald-500/20 bg-background">
                  <Textarea
                    placeholder="Elements to include in your story..."
                    value={worldData.prompts.positive}
                    onChange={(e) => setWorldData(prev => ({
                      ...prev,
                      prompts: { ...prev.prompts, positive: e.target.value }
                    }))}
                    className="min-h-[120px] border-0 resize-none focus-visible:ring-0"
                  />
                </ScrollArea>
              </div>
              
              <div className="space-y-2">
                <Label className="text-sm font-medium flex items-center gap-2">
                  <Badge className="bg-rose-500 text-white border-0">−</Badge>
                  Negative Prompts
                </Label>
                <ScrollArea className="h-32 rounded-lg border border-rose-500/20 bg-background">
                  <Textarea
                    placeholder="Elements to avoid in your story..."
                    value={worldData.prompts.negative}
                    onChange={(e) => setWorldData(prev => ({
                      ...prev,
                      prompts: { ...prev.prompts, negative: e.target.value }
                    }))}
                    className="min-h-[120px] border-0 resize-none focus-visible:ring-0"
                  />
                </ScrollArea>
              </div>
            </div>
          </CollapsibleSection>

          {/* Characters Section */}
          <CollapsibleSection
            title="Characters"
            icon={<Users className="h-4 w-4" />}
            color="fuchsia"
            expanded={expandedSections.characters}
            onToggle={() => toggleSection('characters')}
            action={
              <Button size="sm" variant="ghost" onClick={addCharacter} className="h-7 gap-1">
                <Plus className="h-3.5 w-3.5" />
                Add
              </Button>
            }
          >
            <AnimatePresence mode="popLayout">
              {worldData.characters.length === 0 ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center py-8"
                >
                  <Users className="h-10 w-10 text-muted-foreground/30 mx-auto mb-3" />
                  <p className="text-sm text-muted-foreground mb-3">No characters defined yet</p>
                  <Button size="sm" variant="outline" onClick={addCharacter}>
                    <Plus className="h-4 w-4 mr-2" />
                    Add Character
                  </Button>
                </motion.div>
              ) : (
                <div className="space-y-3">
                  {worldData.characters.map((char, index) => (
                    <motion.div
                      key={char.id}
                      initial={{ opacity: 0, y: 20, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -20, scale: 0.95 }}
                      transition={{ delay: index * 0.05 }}
                      className="p-4 rounded-xl bg-gradient-to-r from-fuchsia-500/5 to-violet-500/5 border border-fuchsia-500/20"
                    >
                      <div className="flex items-start gap-3">
                        <div className="flex-1 space-y-3">
                          <Input
                            placeholder="Character name..."
                            value={char.title}
                            onChange={(e) => updateCharacter(char.id, 'title', e.target.value)}
                            className="border-fuchsia-500/30 focus:border-fuchsia-500"
                          />
                          <ScrollArea className="h-28 rounded-lg border border-fuchsia-500/20 bg-background">
                            <Textarea
                              placeholder="Character description: role, personality, appearance, background..."
                              value={char.description}
                              onChange={(e) => updateCharacter(char.id, 'description', e.target.value)}
                              className="min-h-[100px] border-0 resize-none focus-visible:ring-0"
                            />
                          </ScrollArea>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 hover:bg-rose-500/10 hover:text-rose-500"
                          onClick={() => removeCharacter(char.id)}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </AnimatePresence>
          </CollapsibleSection>

          {/* Locations Section */}
          <CollapsibleSection
            title="Locations"
            icon={<MapPin className="h-4 w-4" />}
            color="teal"
            expanded={expandedSections.locations}
            onToggle={() => toggleSection('locations')}
            action={
              <Button size="sm" variant="ghost" onClick={addLocation} className="h-7 gap-1">
                <Plus className="h-3.5 w-3.5" />
                Add
              </Button>
            }
          >
            <AnimatePresence mode="popLayout">
              {worldData.locations.length === 0 ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center py-8"
                >
                  <MapPin className="h-10 w-10 text-muted-foreground/30 mx-auto mb-3" />
                  <p className="text-sm text-muted-foreground mb-3">No locations defined yet</p>
                  <Button size="sm" variant="outline" onClick={addLocation}>
                    <Plus className="h-4 w-4 mr-2" />
                    Add Location
                  </Button>
                </motion.div>
              ) : (
                <div className="space-y-3">
                  {worldData.locations.map((loc, index) => (
                    <motion.div
                      key={loc.id}
                      initial={{ opacity: 0, y: 20, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -20, scale: 0.95 }}
                      transition={{ delay: index * 0.05 }}
                      className="p-4 rounded-xl bg-gradient-to-r from-teal-500/5 to-emerald-500/5 border border-teal-500/20"
                    >
                      <div className="flex items-start gap-3">
                        <div className="flex-1">
                          <ScrollArea className="h-28 rounded-lg border border-teal-500/20 bg-background">
                            <Textarea
                              placeholder="Location description: name, type, atmosphere, features..."
                              value={loc.description}
                              onChange={(e) => updateLocation(loc.id, e.target.value)}
                              className="min-h-[100px] border-0 resize-none focus-visible:ring-0"
                            />
                          </ScrollArea>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 hover:bg-rose-500/10 hover:text-rose-500"
                          onClick={() => removeLocation(loc.id)}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </AnimatePresence>
          </CollapsibleSection>

          {/* Lore Section */}
          <CollapsibleSection
            title="Lore - Story Beginning"
            icon={<BookOpen className="h-4 w-4" />}
            color="orange"
            expanded={expandedSections.lore}
            onToggle={() => toggleSection('lore')}
          >
            <ScrollArea className="h-48 rounded-lg border border-orange-500/20 bg-background">
              <Textarea
                placeholder="Write the beginning of your story, describe the world's history, or add important lore elements..."
                value={worldData.lore.storyBeginning}
                onChange={(e) => setWorldData(prev => ({
                  ...prev,
                  lore: { storyBeginning: e.target.value }
                }))}
                className="min-h-[180px] border-0 resize-none focus-visible:ring-0"
              />
            </ScrollArea>
          </CollapsibleSection>

          {/* Generation Presets Section */}
          <CollapsibleSection
            title="Generation Presets"
            icon={<Gauge className="h-4 w-4" />}
            color="violet"
            expanded={expandedSections.generationPresets}
            onToggle={() => toggleSection('generationPresets')}
          >
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground mb-4">
                Select or create custom generation presets to control the AI's writing style. 
                These settings affect creativity, repetition, and output length.
              </p>
              <GenerationPresetSelector showCreateButton={true} />
            </div>
          </CollapsibleSection>

          {/* Save Button */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="sticky bottom-4 pt-4"
          >
            <div className="flex items-center justify-center gap-3">
              <Button
                variant="outline"
                size="lg"
                onClick={handleSave}
                disabled={isSaving}
                className="gap-2 min-w-[160px]"
              >
                {isSaving ? (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Save className="h-4 w-4" />
                  </motion.div>
                ) : (
                  <Save className="h-4 w-4" />
                )}
                Save Studio
              </Button>
              
              <Button
                size="lg"
                onClick={handleImportToEditor}
                disabled={isSaving}
                className="gap-2 min-w-[200px] bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-700 hover:to-fuchsia-700 border-0"
              >
                <Send className="h-4 w-4" />
                Import to Editor
              </Button>
            </div>
          </motion.div>
          
          {/* Spacer for sticky button */}
          <div className="h-16" />
        </div>
      </ScrollArea>
    </div>
  )
}

// ============================================================================
// Collapsible Section Component
// ============================================================================

interface CollapsibleSectionProps {
  title: string
  icon: React.ReactNode
  color: 'violet' | 'amber' | 'emerald' | 'rose' | 'sky' | 'fuchsia' | 'teal' | 'orange'
  expanded: boolean
  onToggle: () => void
  action?: React.ReactNode
  children: React.ReactNode
}

function CollapsibleSection({ title, icon, color, expanded, onToggle, action, children }: CollapsibleSectionProps) {
  const colorClasses = {
    violet: 'from-violet-500/10 to-fuchsia-500/5 border-violet-500/30 hover:border-violet-500/50',
    amber: 'from-amber-500/10 to-orange-500/5 border-amber-500/30 hover:border-amber-500/50',
    emerald: 'from-emerald-500/10 to-teal-500/5 border-emerald-500/30 hover:border-emerald-500/50',
    rose: 'from-rose-500/10 to-pink-500/5 border-rose-500/30 hover:border-rose-500/50',
    sky: 'from-sky-500/10 to-blue-500/5 border-sky-500/30 hover:border-sky-500/50',
    fuchsia: 'from-fuchsia-500/10 to-violet-500/5 border-fuchsia-500/30 hover:border-fuchsia-500/50',
    teal: 'from-teal-500/10 to-emerald-500/5 border-teal-500/30 hover:border-teal-500/50',
    orange: 'from-orange-500/10 to-amber-500/5 border-orange-500/30 hover:border-orange-500/50',
  }
  
  const iconBgClasses = {
    violet: 'bg-violet-500/20 text-violet-500',
    amber: 'bg-amber-500/20 text-amber-500',
    emerald: 'bg-emerald-500/20 text-emerald-500',
    rose: 'bg-rose-500/20 text-rose-500',
    sky: 'bg-sky-500/20 text-sky-500',
    fuchsia: 'bg-fuchsia-500/20 text-fuchsia-500',
    teal: 'bg-teal-500/20 text-teal-500',
    orange: 'bg-orange-500/20 text-orange-500',
  }
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "rounded-2xl border-2 transition-all overflow-hidden",
        "bg-gradient-to-br",
        colorClasses[color]
      )}
    >
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-4 hover:bg-white/5 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className={cn("p-2 rounded-xl", iconBgClasses[color])}>
            {icon}
          </div>
          <span className="font-semibold">{title}</span>
        </div>
        <div className="flex items-center gap-2">
          {action && expanded && (
            <div onClick={(e) => e.stopPropagation()}>
              {action}
            </div>
          )}
          {expanded ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </div>
      </button>
      
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
