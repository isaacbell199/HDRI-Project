'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  ArrowRight,
  Swords,
  Zap,
  Clapperboard,
  Undo2,
  RefreshCw,
  Square,
  Loader2,
  Sparkles,
  Play,
  Check,
  X,
  Plus,
  MessageCircle
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'sonner'
import {
  generateText,
  stopGeneration,
  onGenerationChunk,
  isTauri,
  LanguageMode
} from '@/lib/tauri-api'
import { StoryChatPanel } from '@/components/story-chat-panel'

interface FloatingAIToolsProps {
  isModelLoaded: boolean
  editorContent: string
  lastGeneratedContent: string | null
  lastGeneratedIndex: number | null
  onApplySuggestion: (content: string) => void
  onStreamChunk: (chunk: string) => void
  onGenerationStart: () => void
  onGenerationEnd: () => void
  onClearLastGeneration: () => void
  onRegenerate: () => void
  isGenerating: boolean
  temperature: number
  maxTokens: number
  languageMode: LanguageMode
  // Sampling parameters
  topP: number
  topK: number
  minP: number
  repeatPenalty: number
  frequencyPenalty: number
  presencePenalty: number
  // Creative Settings (from Wizard or Editor)
  creativeCategory?: string
  creativeCategoryInstruction?: string
  creativeTone?: string
  creativeToneInstruction?: string
  creativeStyle?: string
  creativeStyleInstruction?: string
  creativeThemeName?: string
  creativeThemeInstruction?: string
}

interface Suggestion {
  id: string
  text: string
}

// Prompts to generate dynamic suggestions based on story context
const SUGGESTION_GENERATION_PROMPTS: Record<string, string> = {
  next: `You are a creative writing assistant analyzing the current story. Read the story context carefully and generate 5 creative, intelligent suggestions for what could happen next.

Consider:
- Current plot trajectory and pacing
- Character motivations and relationships
- Unresolved tension or mysteries
- Natural story progression

Each suggestion should be brief (3-6 words), compelling, and feel organic to the story. Avoid generic ideas - make them specific to the current context.

Format: exactly 5 numbered suggestions, one per line.
Example format:
1. [specific suggestion based on story context]
2. [another contextual suggestion]
...`,

  conflict: `You are a creative writing assistant analyzing the current story. Generate 5 compelling conflict ideas that could emerge naturally from the current situation.

Consider:
- Character flaws and opposing goals
- External threats or obstacles
- Internal struggles and moral dilemmas
- Power dynamics and betrayals

Each suggestion should be brief (3-6 words), dramatic, and arise naturally from the established story. Make the conflict feel inevitable yet surprising.

Format: exactly 5 numbered suggestions, one per line.`,

  action: `You are a creative writing assistant analyzing the current story. Generate 5 dynamic action scene ideas that would feel thrilling and appropriate.

Consider:
- Physical stakes and dangers
- Chase, combat, or escape scenarios
- Time pressure and urgency
- Environmental hazards or advantages

Each suggestion should be brief (3-6 words), exciting, and fit naturally within the story's tone and setting.

Format: exactly 5 numbered suggestions, one per line.`,

  scenario: `You are a creative writing assistant analyzing the current story. Generate 5 interesting scenario developments that could take the story in compelling new directions.

Consider:
- Unexpected twists that recontextualize events
- New locations or world-building opportunities
- Surprising character revelations
- Thematic resonances and symbolism

Each suggestion should be brief (3-6 words), intriguing, and open new narrative possibilities while staying true to the story's spirit.

Format: exactly 5 numbered suggestions, one per line.`
}

// Prompts to generate a paragraph based on chosen suggestion
const PARAGRAPH_GENERATION_PROMPTS: Record<string, (suggestion: string) => string> = {
  next: (suggestion) => `Continue this story by incorporating: "${suggestion}".

Write ONE engaging paragraph (3-5 sentences) that:
- Flows naturally from the current text
- Maintains the story's established tone and voice
- Advances the plot meaningfully
- Shows rather than tells
- Uses specific, vivid details

Do not repeat or summarize previous content. Continue forward.`,

  conflict: (suggestion) => `Introduce this conflict into the story: "${suggestion}".

Write ONE dramatic paragraph (3-5 sentences) that:
- Creates genuine tension and stakes
- Reveals character through reaction
- Advances the plot organically
- Uses sensory details and strong verbs
- Makes the conflict feel inevitable yet surprising

Do not repeat previous content.`,

  action: (suggestion) => `Write an action scene based on: "${suggestion}".

Write ONE dynamic paragraph (3-5 sentences) that:
- Uses short, punchy sentences for impact
- Incorporates strong verbs and sensory details
- Creates urgency and momentum
- Shows physical movement and reaction
- Maintains clarity amidst chaos

Focus on the moment-to-moment action.`,

  scenario: (suggestion) => `Develop this scenario: "${suggestion}".

Write ONE atmospheric paragraph (3-5 sentences) that:
- Establishes the new direction smoothly
- Uses vivid, evocative language
- Creates intrigue and curiosity
- Maintains narrative coherence
- Grounds the reader in the new situation

Transition naturally from what came before.`
}

// Generate demo suggestions for browser mode
function generateDemoSuggestions(toolId: string): Suggestion[] {
  const baseSuggestions: Record<string, string[][]> = {
    next: [
      ['A mysterious figure emerges', 'Hidden truth comes to light', 'Unexpected ally arrives', 'Time runs short', 'Secrets unravel'],
    ],
    conflict: [
      ['Loyalty is tested', 'Hidden betrayal revealed', 'Rivalry ignites', 'Trust crumbles away', 'Old wounds reopen'],
    ],
    action: [
      ['Sudden chase begins', 'Fight for survival', 'Daring escape attempt', 'Race against time', 'Ambush from shadows'],
    ],
    scenario: [
      ['Journey to unknown', 'Ancient secret discovered', 'Unexpected reunion', 'Mysterious stranger', 'Portal opens'],
    ]
  }

  const options = baseSuggestions[toolId] || baseSuggestions.next
  const randomIndex = Math.floor(Math.random() * options.length)
  return options[randomIndex].map((text, i) => ({ id: `suggestion-${i}`, text }))
}

export function FloatingAITools({
  isModelLoaded,
  editorContent,
  lastGeneratedContent,
  onStreamChunk,
  onGenerationStart,
  onGenerationEnd,
  onClearLastGeneration,
  onRegenerate,
  isGenerating,
  temperature,
  maxTokens,
  languageMode,
  topP,
  topK,
  minP,
  repeatPenalty,
  frequencyPenalty,
  presencePenalty,
  creativeCategory,
  creativeCategoryInstruction,
  creativeTone,
  creativeToneInstruction,
  creativeStyle,
  creativeStyleInstruction,
  creativeThemeName,
  creativeThemeInstruction,
}: FloatingAIToolsProps) {
  const [activeTool, setActiveTool] = useState<string | null>(null)
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [isGeneratingSuggestions, setIsGeneratingSuggestions] = useState(false)
  
  // Preview system
  const [selectedSuggestion, setSelectedSuggestion] = useState<Suggestion | null>(null)
  const [previewContent, setPreviewContent] = useState<string>('')
  const [isGeneratingPreview, setIsGeneratingPreview] = useState(false)
  
  // Chat panel
  const [isChatOpen, setIsChatOpen] = useState(false)

  // Ref to store unsubscribe functions
  const unsubscribeRef = useRef<{ unsubscribe: (() => void) | null; timeoutId: NodeJS.Timeout | null }>({
    unsubscribe: null,
    timeoutId: null
  })

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (unsubscribeRef.current.timeoutId) {
        clearTimeout(unsubscribeRef.current.timeoutId)
      }
      if (unsubscribeRef.current.unsubscribe) {
        unsubscribeRef.current.unsubscribe()
      }
    }
  }, [])

  // Generate suggestions
  const generateSuggestions = useCallback(async (toolId: string) => {
    setIsGeneratingSuggestions(true)
    setSuggestions([])
    setSelectedSuggestion(null)
    setPreviewContent('')

    try {
      if (isTauri()) {
        let generatedText = ''
        
        const unsubscribe = onGenerationChunk((chunk) => {
          if (chunk.done) {
            const lines = generatedText.split('\n').filter(line => line.trim())
            const parsedSuggestions: Suggestion[] = []
            
            lines.forEach((line, index) => {
              const cleaned = line.replace(/^\d+[\.\)]\s*/, '').trim()
              if (cleaned && parsedSuggestions.length < 5) {
                parsedSuggestions.push({
                  id: `suggestion-${index}`,
                  text: cleaned
                })
              }
            })

            while (parsedSuggestions.length < 5) {
              parsedSuggestions.push({
                id: `suggestion-${parsedSuggestions.length}`,
                text: `Idea ${parsedSuggestions.length + 1}`
              })
            }

            setSuggestions(parsedSuggestions.slice(0, 5))
            setIsGeneratingSuggestions(false)
          } else {
            generatedText += chunk.content
          }
        })

        const storyContext = editorContent.slice(-3000)
        const prompt = SUGGESTION_GENERATION_PROMPTS[toolId]

        await generateText({
          mode: 'suggestion',
          text: storyContext,
          context: prompt,
          temperature: 0.9,
          maxTokens: 150,
          topP,
          topK,
          minP,
          repeatPenalty,
          frequencyPenalty,
          presencePenalty,
          languageMode,
          selectedTone: creativeTone,
          customToneInstruction: creativeToneInstruction,
          customStyleName: creativeStyle,
          customStyleInstruction: creativeStyleInstruction,
          customGenreName: creativeCategory,
          customGenreInstruction: creativeCategoryInstruction,
          themeName: creativeThemeName,
          themeInstruction: creativeThemeInstruction,
        })

        unsubscribeRef.current.unsubscribe = unsubscribe
        unsubscribeRef.current.timeoutId = setTimeout(() => {
          if (unsubscribe) unsubscribe()
        }, 30000)
      } else {
        await new Promise(r => setTimeout(r, 1000))
        const demoSuggestions = generateDemoSuggestions(toolId)
        setSuggestions(demoSuggestions)
        setIsGeneratingSuggestions(false)
      }
    } catch (error) {
      console.error('Failed to generate suggestions:', error)
      toast.error('Failed to generate suggestions')
      setIsGeneratingSuggestions(false)
      setSuggestions([
        { id: '1', text: 'An unexpected event occurs' },
        { id: '2', text: 'A new character appears' },
        { id: '3', text: 'A secret is revealed' },
        { id: '4', text: 'The situation intensifies' },
        { id: '5', text: 'A choice must be made' }
      ])
    }
  }, [editorContent, languageMode, creativeCategory, creativeCategoryInstruction, creativeTone, creativeToneInstruction, creativeStyle, creativeStyleInstruction, creativeThemeName, creativeThemeInstruction, topP, topK, minP, repeatPenalty, frequencyPenalty, presencePenalty])

  // Toggle tool and generate suggestions
  const handleToolClick = useCallback(async (toolId: string) => {
    if (activeTool === toolId) {
      // Close the bubble
      setActiveTool(null)
      setSuggestions([])
      setSelectedSuggestion(null)
      setPreviewContent('')
      return
    }

    if (!isModelLoaded) {
      toast.error('Please load an AI model first')
      return
    }

    setActiveTool(toolId)
    generateSuggestions(toolId)
  }, [activeTool, isModelLoaded, generateSuggestions])

  // Regenerate all suggestions
  const handleRegenerateAll = useCallback(() => {
    if (activeTool) {
      setSelectedSuggestion(null)
      setPreviewContent('')
      generateSuggestions(activeTool)
      toast.success('New suggestions generated!')
    }
  }, [activeTool, generateSuggestions])

  // Select a suggestion and generate preview
  const handleSelectSuggestion = useCallback(async (suggestion: Suggestion) => {
    if (!isModelLoaded) {
      toast.error('Please load an AI model first')
      return
    }

    const toolId = activeTool
    if (!toolId) return

    setSelectedSuggestion(suggestion)
    setPreviewContent('')
    setIsGeneratingPreview(true)

    try {
      if (isTauri()) {
        let generatedText = ''
        
        const unsubscribe = onGenerationChunk((chunk) => {
          if (chunk.done) {
            setPreviewContent(generatedText.trim())
            setIsGeneratingPreview(false)
          } else {
            generatedText += chunk.content
          }
        })

        const prompt = PARAGRAPH_GENERATION_PROMPTS[toolId](suggestion.text)

        await generateText({
          mode: 'suggestion',
          text: editorContent.slice(-2000),
          context: prompt,
          temperature,
          maxTokens: 500,
          topP,
          topK,
          minP,
          repeatPenalty,
          frequencyPenalty,
          presencePenalty,
          languageMode,
          selectedTone: creativeTone,
          customToneInstruction: creativeToneInstruction,
          customStyleName: creativeStyle,
          customStyleInstruction: creativeStyleInstruction,
          customGenreName: creativeCategory,
          customGenreInstruction: creativeCategoryInstruction,
          themeName: creativeThemeName,
          themeInstruction: creativeThemeInstruction,
        })

        unsubscribeRef.current.unsubscribe = unsubscribe
        unsubscribeRef.current.timeoutId = setTimeout(() => {
          if (unsubscribe) unsubscribe()
        }, 60000)
      } else {
        await new Promise(r => setTimeout(r, 800))
        const demoPreview = `${suggestion.text}. The moment unfolded with unexpected intensity, each detail sharpening into focus as the situation evolved. What had seemed like a simple path forward now revealed layers of complexity that would change everything.`
        setPreviewContent(demoPreview)
        setIsGeneratingPreview(false)
      }
    } catch (error) {
      console.error('Preview generation error:', error)
      toast.error('Failed to generate preview')
      setIsGeneratingPreview(false)
    }
  }, [activeTool, isModelLoaded, editorContent, temperature, languageMode, creativeCategory, creativeCategoryInstruction, creativeTone, creativeToneInstruction, creativeStyle, creativeStyleInstruction, creativeThemeName, creativeThemeInstruction, topP, topK, minP, repeatPenalty, frequencyPenalty, presencePenalty])

  // Insert the preview content into editor
  const handleInsertPreview = useCallback(() => {
    if (previewContent) {
      onStreamChunk(' ' + previewContent)
      toast.success('Content inserted!')
      // Keep bubble open but clear selection
      setSelectedSuggestion(null)
      setPreviewContent('')
    }
  }, [previewContent, onStreamChunk])

  // Cancel preview
  const handleCancelPreview = useCallback(() => {
    setSelectedSuggestion(null)
    setPreviewContent('')
  }, [])

  // Continue generation - generates directly without suggestions
  const handleContinue = useCallback(async () => {
    if (!isModelLoaded) {
      toast.error('Please load an AI model first')
      return
    }

    onGenerationStart()

    try {
      if (isTauri()) {
        const unsubscribe = onGenerationChunk((chunk) => {
          if (chunk.done) {
            onGenerationEnd()
            toast.success('Content generated!')
          } else {
            onStreamChunk(chunk.content)
          }
        })

        const prompt = `Continue this story naturally.

Write ONE engaging paragraph (3-5 sentences) that:
- Flows naturally from the current text
- Maintains the story's established tone and voice
- Advances the plot meaningfully
- Shows rather than tells
- Uses specific, vivid details

Do not repeat or summarize previous content. Continue forward.

STORY CONTEXT:
${editorContent.slice(-2000)}

Write the continuation now:`

        await generateText({
          mode: 'continue',
          text: editorContent.slice(-2000),
          context: prompt,
          temperature,
          maxTokens: 500,
          topP,
          topK,
          minP,
          repeatPenalty,
          frequencyPenalty,
          presencePenalty,
          languageMode,
          selectedTone: creativeTone,
          customToneInstruction: creativeToneInstruction,
          customStyleName: creativeStyle,
          customStyleInstruction: creativeStyleInstruction,
          customGenreName: creativeCategory,
          customGenreInstruction: creativeCategoryInstruction,
          themeName: creativeThemeName,
          themeInstruction: creativeThemeInstruction,
        })

        unsubscribeRef.current.unsubscribe = unsubscribe
        unsubscribeRef.current.timeoutId = setTimeout(() => {
          if (unsubscribe) unsubscribe()
        }, 60000)
      } else {
        const demoContent = ` The story continued to unfold with unexpected twists, each moment bringing new revelations that would change everything. The characters found themselves drawn deeper into a web of mystery and intrigue, their fates intertwined in ways they had never imagined.`
        
        for (let i = 0; i < demoContent.length; i += 3) {
          await new Promise(r => setTimeout(r, 20))
          onStreamChunk(demoContent.slice(i, i + 3))
        }
        onGenerationEnd()
        toast.success('Content generated!')
      }
    } catch (error) {
      console.error('Continue generation error:', error)
      onGenerationEnd()
      toast.error('Failed to generate content')
    }
  }, [isModelLoaded, editorContent, temperature, languageMode, onStreamChunk, onGenerationStart, onGenerationEnd, creativeCategory, creativeCategoryInstruction, creativeTone, creativeToneInstruction, creativeStyle, creativeStyleInstruction, creativeThemeName, creativeThemeInstruction, topP, topK, minP, repeatPenalty, frequencyPenalty, presencePenalty])

  // Stop generation
  const handleStopGeneration = useCallback(() => {
    stopGeneration()
    onGenerationEnd()
    setIsGeneratingPreview(false)
    toast.info('Generation stopped')
  }, [onGenerationEnd])

  // Close suggestions when clicking outside
  const handleCloseSuggestions = useCallback(() => {
    setActiveTool(null)
    setSuggestions([])
    setSelectedSuggestion(null)
    setPreviewContent('')
  }, [])

  const tools = [
    { 
      id: 'next', 
      icon: ArrowRight, 
      label: 'Next',
      color: 'text-sky-500',
      bg: 'hover:bg-sky-500/10',
      activeBg: 'bg-sky-500/20 border-sky-500/30'
    },
    { 
      id: 'conflict', 
      icon: Swords, 
      label: 'Conflict',
      color: 'text-rose-500',
      bg: 'hover:bg-rose-500/10',
      activeBg: 'bg-rose-500/20 border-rose-500/30'
    },
    { 
      id: 'action', 
      icon: Zap, 
      label: 'Action',
      color: 'text-amber-500',
      bg: 'hover:bg-amber-500/10',
      activeBg: 'bg-amber-500/20 border-amber-500/30'
    },
    { 
      id: 'scenario', 
      icon: Clapperboard, 
      label: 'Scenario',
      color: 'text-violet-500',
      bg: 'hover:bg-violet-500/10',
      activeBg: 'bg-violet-500/20 border-violet-500/30'
    },
  ]

  const actionTools = [
    { 
      id: 'clear', 
      icon: Undo2, 
      label: 'Clear',
      color: 'text-muted-foreground',
      bg: 'hover:bg-rose-500/10 hover:text-rose-500',
      disabled: !lastGeneratedContent,
      onClick: onClearLastGeneration
    },
    { 
      id: 'regenerate', 
      icon: RefreshCw, 
      label: 'Regenerate',
      color: 'text-muted-foreground',
      bg: 'hover:bg-emerald-500/10 hover:text-emerald-500',
      disabled: !lastGeneratedContent || isGenerating,
      onClick: onRegenerate
    },
  ]

  return (
    <div className="relative">
      {/* Fixed Buttons Column */}
      <div className="flex flex-col gap-1.5 items-center">
        {/* Suggestion Tools */}
        {tools.map((tool) => (
          <div key={tool.id} className="relative">
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                "h-9 w-9 rounded-lg transition-all relative group border",
                tool.color,
                tool.bg,
                activeTool === tool.id ? tool.activeBg : "border-transparent"
              )}
              disabled={!isModelLoaded || isGenerating || isGeneratingSuggestions}
              onClick={() => handleToolClick(tool.id)}
            >
              <tool.icon className="h-4 w-4" />
            </Button>
            
            {/* Tooltip */}
            <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 px-2 py-1 rounded-md bg-popover border border-border/50 text-[10px] font-medium whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none shadow-md z-50">
              {tool.label}
            </div>
          </div>
        ))}

        {/* Divider */}
        <div className="w-5 h-px bg-border/50 my-0.5" />

        {/* Action Tools */}
        {actionTools.map((tool) => (
          <div key={tool.id} className="relative">
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                "h-9 w-9 rounded-lg transition-all relative group border border-transparent",
                tool.color,
                tool.bg,
                tool.disabled && "opacity-40 cursor-not-allowed"
              )}
              disabled={tool.disabled}
              onClick={tool.onClick}
            >
              <tool.icon className={cn("h-4 w-4", isGenerating && tool.id === 'regenerate' && "animate-spin")} />
            </Button>
            
            {/* Tooltip */}
            <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 px-2 py-1 rounded-md bg-popover border border-border/50 text-[10px] font-medium whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none shadow-md z-50">
              {tool.label}
            </div>
          </div>
        ))}

        {/* Divider */}
        <div className="w-5 h-px bg-border/50 my-0.5" />

        {/* Continue Button */}
        <div className="relative">
          <Button
            variant="ghost"
            size="icon"
              className={cn(
              "h-9 w-9 rounded-lg transition-all relative group border border-transparent text-emerald-500 hover:bg-emerald-500/10",
              isGenerating && "bg-red-500/20 border-red-500/30 text-red-500 hover:bg-red-500/30"
            )}
            disabled={!isModelLoaded || isGenerating || isGeneratingSuggestions}
            onClick={isGenerating ? handleStopGeneration : handleContinue}
          >
            {isGenerating ? (
              <Square className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
          </Button>
          
          {/* Tooltip */}
          <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 px-2 py-1 rounded-md bg-popover border border-border/50 text-[10px] font-medium whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none shadow-md z-50">
            {isGenerating ? 'Stop' : 'Continue'}
          </div>
        </div>

        {/* Divider */}
        <div className="w-5 h-px bg-border/50 my-0.5" />

        {/* Chat Button */}
        <div className="relative">
          <Button
            variant="ghost"
            size="icon"
            className={cn(
              "h-9 w-9 rounded-lg transition-all relative group border",
              "text-violet-500 hover:bg-violet-500/10",
              isChatOpen ? "bg-violet-500/20 border-violet-500/30" : "border-transparent"
            )}
            onClick={() => setIsChatOpen(true)}
          >
            <MessageCircle className="h-4 w-4" />
          </Button>
          
          {/* Tooltip */}
          <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 px-2 py-1 rounded-md bg-popover border border-border/50 text-[10px] font-medium whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none shadow-md z-50">
            Chat
          </div>
        </div>
      </div>

      {/* Floating Suggestion Bubble */}
      <AnimatePresence>
        {activeTool && (
          <>
            {/* Backdrop to close on click outside */}
            <div 
              className="fixed inset-0 z-40" 
              onClick={handleCloseSuggestions}
            />
            
            {/* Floating Bubble */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9, x: -10 }}
              animate={{ opacity: 1, scale: 1, x: 0 }}
              exit={{ opacity: 0, scale: 0.9, x: -10 }}
              transition={{ duration: 0.15 }}
              className="absolute left-full ml-3 top-0 z-50 w-[280px] rounded-xl bg-popover/95 backdrop-blur-sm border border-border/50 shadow-xl overflow-hidden"
            >
              {/* Header */}
              <div className="flex items-center justify-between px-3 py-2 border-b border-border/30 bg-muted/10">
                <div className="flex items-center gap-1.5">
                  <Sparkles className="h-3 w-3 text-violet-500" />
                  <span className="text-[10px] font-semibold text-foreground">
                    {tools.find(t => t.id === activeTool)?.label} Ideas
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  {/* Regenerate Button */}
                  <button
                    onClick={handleRegenerateAll}
                    disabled={isGeneratingSuggestions || isGeneratingPreview}
                    className="w-5 h-5 rounded flex items-center justify-center hover:bg-muted/50 text-muted-foreground hover:text-violet-500 transition-colors disabled:opacity-50"
                    title="Regenerate all suggestions"
                  >
                    <RefreshCw className={cn("h-3 w-3", isGeneratingSuggestions && "animate-spin")} />
                  </button>
                  <button
                    onClick={handleCloseSuggestions}
                    className="w-5 h-5 rounded flex items-center justify-center hover:bg-muted/50 text-muted-foreground transition-colors"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              </div>

              {/* Content */}
              <div className="p-2 max-h-[300px] overflow-y-auto">
                {isGeneratingSuggestions ? (
                  <div className="flex items-center justify-center py-6 gap-2">
                    <Loader2 className="h-4 w-4 animate-spin text-violet-500" />
                    <span className="text-xs text-muted-foreground">Analyzing story...</span>
                  </div>
                ) : (
                  <>
                    {/* Suggestions List */}
                    <div className="space-y-1 mb-2">
                      {suggestions.map((suggestion, index) => (
                        <motion.button
                          key={suggestion.id}
                          initial={{ opacity: 0, y: 5 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.05 }}
                          onClick={() => handleSelectSuggestion(suggestion)}
                          disabled={isGeneratingPreview}
                          className={cn(
                            "w-full px-3 py-2 rounded-lg text-xs text-left transition-all flex items-center justify-between gap-2",
                            selectedSuggestion?.id === suggestion.id
                              ? "bg-violet-500/20 border border-violet-500/40 text-violet-700 dark:text-violet-300"
                              : "hover:bg-muted/50 border border-transparent hover:border-border/50",
                            "disabled:opacity-50 disabled:cursor-not-allowed"
                          )}
                        >
                          <span className="flex-1">{suggestion.text}</span>
                          {selectedSuggestion?.id === suggestion.id && (
                            <Check className="h-3 w-3 text-violet-500 shrink-0" />
                          )}
                        </motion.button>
                      ))}
                    </div>

                    {/* Preview Section */}
                    <AnimatePresence>
                      {(selectedSuggestion || isGeneratingPreview) && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          transition={{ duration: 0.2 }}
                          className="overflow-hidden"
                        >
                          <div className="border-t border-border/30 pt-2 mt-2">
                            <div className="flex items-center gap-1.5 mb-2">
                              <div className="w-1.5 h-1.5 rounded-full bg-violet-500" />
                              <span className="text-[10px] font-medium text-muted-foreground">Preview</span>
                            </div>
                            
                            {isGeneratingPreview ? (
                              <div className="flex items-center justify-center py-4 gap-2 bg-muted/30 rounded-lg">
                                <Loader2 className="h-3 w-3 animate-spin text-violet-500" />
                                <span className="text-[10px] text-muted-foreground">Generating...</span>
                              </div>
                            ) : previewContent ? (
                              <div className="space-y-2">
                                <div className="p-2 rounded-lg bg-muted/30 border border-border/30 text-xs text-foreground/90 leading-relaxed max-h-[100px] overflow-y-auto">
                                  {previewContent}
                                </div>
                                <div className="flex gap-2">
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={handleCancelPreview}
                                    className="flex-1 h-7 text-[10px] text-muted-foreground hover:text-foreground"
                                  >
                                    <X className="h-3 w-3 mr-1" />
                                    Cancel
                                  </Button>
                                  <Button
                                    size="sm"
                                    onClick={handleInsertPreview}
                                    className="flex-1 h-7 text-[10px] bg-gradient-to-r from-violet-500 to-fuchsia-500 border-0"
                                  >
                                    <Plus className="h-3 w-3 mr-1" />
                                    Insert
                                  </Button>
                                </div>
                              </div>
                            ) : null}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </>
                )}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Chat Panel */}
      <StoryChatPanel
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
        isModelLoaded={isModelLoaded}
        editorContent={editorContent}
        temperature={temperature}
        languageMode={languageMode}
        topP={topP}
        topK={topK}
        minP={minP}
        repeatPenalty={repeatPenalty}
        frequencyPenalty={frequencyPenalty}
        presencePenalty={presencePenalty}
        creativeCategory={creativeCategory}
        creativeTone={creativeTone}
        creativeStyle={creativeStyle}
        creativeThemeName={creativeThemeName}
      />
    </div>
  )
}
