'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import {
  MessageCircle,
  X,
  Send,
  User,
  Bot,
  Sparkles,
  Swords,
  MapPin,
  Lightbulb,
  Heart,
  Zap,
  RefreshCw,
  Copy,
  Check,
  Loader2
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

interface StoryChatPanelProps {
  isOpen: boolean
  onClose: () => void
  isModelLoaded: boolean
  editorContent: string
  temperature: number
  languageMode: LanguageMode
  topP: number
  topK: number
  minP: number
  repeatPenalty: number
  frequencyPenalty: number
  presencePenalty: number
  creativeCategory?: string
  creativeTone?: string
  creativeStyle?: string
  creativeThemeName?: string
  projectCharacters?: { id: string; name: string; description: string }[]
  projectLore?: string
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

interface QuickPrompt {
  id: string
  icon: typeof User
  label: string
  prompt: string
  color: string
}

const QUICK_PROMPTS: QuickPrompt[] = [
  {
    id: 'character',
    icon: User,
    label: 'Personnage',
    prompt: "J'ai besoin d'idées pour créer un nouveau personnage. Peux-tu me proposer des archétypes qui correspondraient bien à mon histoire ?",
    color: 'text-sky-500 hover:bg-sky-500/10'
  },
  {
    id: 'conflict',
    icon: Swords,
    label: 'Conflit',
    prompt: "Quels types de conflits pourraient intensifier mon histoire ? Propose-moi des idées dramatiques.",
    color: 'text-rose-500 hover:bg-rose-500/10'
  },
  {
    id: 'place',
    icon: MapPin,
    label: 'Lieu',
    prompt: "Aide-moi à développer un lieu important pour mon histoire. Quels détails atmosphériques devrais-je inclure ?",
    color: 'text-emerald-500 hover:bg-emerald-500/10'
  },
  {
    id: 'idea',
    icon: Lightbulb,
    label: 'Idée',
    prompt: "Je suis bloqué dans mon histoire. Peux-tu m'aider à trouver des idées pour la suite ?",
    color: 'text-amber-500 hover:bg-amber-500/10'
  },
  {
    id: 'emotion',
    icon: Heart,
    label: 'Émotion',
    prompt: "Comment puis-je rendre cette scène plus émouvante ? Donne-moi des conseils pour toucher le lecteur.",
    color: 'text-pink-500 hover:bg-pink-500/10'
  },
  {
    id: 'twist',
    icon: Zap,
    label: 'Twist',
    prompt: "Propose-moi des twists surprenants qui pourraient transformer mon histoire de façon inattendue.",
    color: 'text-violet-500 hover:bg-violet-500/10'
  }
]

const SYSTEM_PROMPT = `Tu es un assistant narratif expert en écriture créative. Tu aides l'auteur à développer son histoire avec des conseils pertinents et créatifs.

Tu peux aider avec:
- Création et développement de personnages (personnalité, motivation, arc narratif)
- Structure de l'intrigue et conflits narratifs
- Atmosphère et descriptions de lieux
- Dialogues et voix des personnages
- Twists et rebondissements
- Émotion et engagement du lecteur

Réponds de façon claire, structurée et actionable. Utilise des exemples concrets quand c'est pertinent.
Si l'auteur te donne du contexte sur son histoire, utilise-le pour personnaliser tes conseils.
Sois créatif mais reste cohérent avec l'univers établi.`

export function StoryChatPanel({
  isOpen,
  onClose,
  isModelLoaded,
  editorContent,
  temperature,
  languageMode,
  topP,
  topK,
  minP,
  repeatPenalty,
  frequencyPenalty,
  presencePenalty,
  creativeCategory,
  creativeTone,
  creativeStyle,
  creativeThemeName,
  projectCharacters,
  projectLore,
}: StoryChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Bug 1 fix: Track timeout IDs for cleanup on unmount
  const timeoutRefs = useRef<ReturnType<typeof setTimeout>[]>([])

  // Bug 3 fix: Ref for isGenerating to avoid stale closures in async callbacks
  const isGeneratingRef = useRef(isGenerating)
  useEffect(() => {
    isGeneratingRef.current = isGenerating
  }, [isGenerating])

  // Bug 1 fix: Cleanup all tracked timeouts on unmount
  useEffect(() => {
    return () => {
      timeoutRefs.current.forEach(id => clearTimeout(id))
      timeoutRefs.current = []
    }
  }, [])

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Focus input when panel opens
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [isOpen])

  // Build context for AI
  const buildContext = useCallback(() => {
    let context = ''
    
    if (creativeCategory) context += `Genre: ${creativeCategory}\n`
    if (creativeTone) context += `Tone: ${creativeTone}\n`
    if (creativeStyle) context += `Style: ${creativeStyle}\n`
    if (creativeThemeName) context += `Theme: ${creativeThemeName}\n`
    
    if (projectCharacters && projectCharacters.length > 0) {
      context += `\nPersonnages:\n${projectCharacters.map(c => `- ${c.name}: ${c.description}`).join('\n')}\n`
    }
    
    if (projectLore) {
      context += `\nLore du monde: ${projectLore}\n`
    }
    
    if (editorContent.trim()) {
      context += `\nContenu actuel de l'histoire (extrait):\n${editorContent.slice(-2000)}\n`
    }
    
    return context
  }, [creativeCategory, creativeTone, creativeStyle, creativeThemeName, projectCharacters, projectLore, editorContent])

  // Send message
  const handleSend = useCallback(async (messageText?: string) => {
    const text = messageText || input.trim()
    if (!text || isGenerating) return

    if (!isModelLoaded) {
      toast.error('Please load an AI model first')
      return
    }

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsGenerating(true)

    // Add placeholder for AI response
    const assistantMessageId = `assistant-${Date.now()}`
    setMessages(prev => [...prev, {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date()
    }])

    try {
      if (isTauri()) {
        let generatedContent = ''
        
        const unsubscribe = onGenerationChunk((chunk) => {
          if (chunk.done) {
            setMessages(prev => prev.map(m => 
              m.id === assistantMessageId 
                ? { ...m, content: generatedContent }
                : m
            ))
            setIsGenerating(false)
          } else {
            generatedContent += chunk.content
            setMessages(prev => prev.map(m => 
              m.id === assistantMessageId 
                ? { ...m, content: generatedContent }
                : m
            ))
          }
        })

        const context = buildContext()
        const conversationHistory = messages.map(m => 
          `${m.role === 'user' ? 'Auteur' : 'Assistant'}: ${m.content}`
        ).join('\n\n')

        const prompt = `${SYSTEM_PROMPT}

${context ? `CONTEXTE DE L'HISTOIRE:\n${context}\n` : ''}

${conversationHistory ? `CONVERSATION PRÉCÉDENTE:\n${conversationHistory}\n` : ''}

Auteur: ${text}

Assistant:`

        await generateText({
          mode: 'chat',
          text: text,
          context: prompt,
          temperature: 0.85,
          maxTokens: 800,
          topP,
          topK,
          minP,
          repeatPenalty,
          frequencyPenalty,
          presencePenalty,
          languageMode,
          selectedTone: creativeTone,
          customStyleName: creativeStyle,
          customGenreName: creativeCategory,
          themeName: creativeThemeName,
        })

        const timeoutId = setTimeout(() => unsubscribe(), 60000)
        timeoutRefs.current.push(timeoutId)
      } else {
        // Demo mode
        const demoResponses = [
          "C'est une excellente question ! Pour développer ton idée, je te suggère de considérer plusieurs aspects...\n\n**1. Motivation du personnage**\nChaque personnage a besoin d'une motivation claire qui guide ses actions.\n\n**2. Conflit interne**\nUn conflit interne rend le personnage plus profond et attachant.\n\n**3. Arc narratif**\nPense à comment le personnage va évoluer au fil de l'histoire.",
          "Voici mes suggestions créatives:\n\n• **Option A**: Un retournement de situation inattendu\n• **Option B**: Une révélation sur le passé d'un personnage\n• **Option C**: Un événement externe qui change tout\n\nLaquelle résonne le plus avec ta vision ?",
          "Pour intensifier cette scène, je recommande:\n\n1. Ajouter des détails sensoriels (sons, odeurs, textures)\n2. Ralentir le rythme pour créer de la tension\n3. Montrer les réactions émotionnelles des personnages\n4. Utiliser des dialogues percutants"
        ]
        
        const demoContent = demoResponses[Math.floor(Math.random() * demoResponses.length)]
        
        for (let i = 0; i < demoContent.length; i += 3) {
          await new Promise(r => setTimeout(r, 15))
          const partial = demoContent.slice(0, i + 3)
          setMessages(prev => prev.map(m => 
            m.id === assistantMessageId 
              ? { ...m, content: partial }
              : m
          ))
        }
        
        setIsGenerating(false)
      }
    } catch (error) {
      console.error('Chat generation error:', error)
      setMessages(prev => prev.map(m => 
        m.id === assistantMessageId 
          ? { ...m, content: "Désolé, une erreur s'est produite. Veuillez réessayer." }
          : m
      ))
      setIsGenerating(false)
      toast.error('Erreur de génération')
    }
  }, [input, isGenerating, isModelLoaded, messages, buildContext, creativeCategory, creativeTone, creativeStyle, creativeThemeName, topP, topK, minP, repeatPenalty, frequencyPenalty, presencePenalty, languageMode])

  // Copy message to clipboard
  const handleCopy = useCallback((content: string, id: string) => {
    navigator.clipboard.writeText(content)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
    toast.success('Copié !')
  }, [])

  // Clear conversation
  const handleClearConversation = useCallback(() => {
    setMessages([])
    toast.success('Conversation effacée')
  }, [])

  // Handle quick prompt click
  const handleQuickPrompt = useCallback((prompt: string) => {
    handleSend(prompt)
  }, [handleSend])

  // Handle Enter key
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }, [handleSend])

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
            onClick={onClose}
          />

          {/* Chat Panel - Centered */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            transition={{ type: 'spring', duration: 0.4 }}
            className="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"
          >
            <div className="w-[700px] h-[750px] bg-gradient-to-b from-background to-muted/30 rounded-2xl border border-border/50 shadow-2xl overflow-hidden flex flex-col">
              
              {/* Header */}
              <div className="flex items-center justify-between px-6 py-4 border-b border-border/30 bg-muted/10">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
                    <MessageCircle className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-base font-semibold">Story Chat</h2>
                    <p className="text-xs text-muted-foreground">Discute de ton histoire avec l'IA</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {messages.length > 0 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleClearConversation}
                      className="text-xs text-muted-foreground hover:text-foreground"
                    >
                      <RefreshCw className="h-3.5 w-3.5 mr-1" />
                      Effacer
                    </Button>
                  )}
                  <button
                    onClick={onClose}
                    className="w-8 h-8 rounded-lg flex items-center justify-center hover:bg-muted/50 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>
              </div>

              {/* Messages Area */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-center px-8">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 flex items-center justify-center mb-4">
                      <Sparkles className="h-8 w-8 text-violet-500" />
                    </div>
                    <h3 className="text-lg font-semibold mb-2">Bienvenue !</h3>
                    <p className="text-sm text-muted-foreground max-w-md mb-6">
                      Je suis ton assistant narratif. Pose-moi des questions sur tes personnages, 
                      ton intrigue, ou demande des conseils créatifs pour enrichir ton histoire.
                    </p>
                  </div>
                ) : (
                  messages.map((message) => (
                    <motion.div
                      key={message.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={cn(
                        "flex gap-3",
                        message.role === 'user' ? "flex-row-reverse" : ""
                      )}
                    >
                      <div className={cn(
                        "w-8 h-8 rounded-lg flex items-center justify-center shrink-0",
                        message.role === 'user' 
                          ? "bg-sky-500/20 text-sky-500"
                          : "bg-violet-500/20 text-violet-500"
                      )}>
                        {message.role === 'user' ? (
                          <User className="h-4 w-4" />
                        ) : (
                          <Bot className="h-4 w-4" />
                        )}
                      </div>
                      <div className={cn(
                        "flex-1 max-w-[85%]",
                        message.role === 'user' ? "text-right" : ""
                      )}>
                        <div className={cn(
                          "rounded-2xl px-4 py-3 inline-block text-left",
                          message.role === 'user'
                            ? "bg-sky-500/10 border border-sky-500/20"
                            : "bg-muted/50 border border-border/30"
                        )}>
                          <p className="text-sm whitespace-pre-wrap leading-relaxed">
                            {message.content || (
                              <span className="flex items-center gap-2 text-muted-foreground">
                                <Loader2 className="h-4 w-4 animate-spin" />
                                Réflexion en cours...
                              </span>
                            )}
                          </p>
                        </div>
                        {message.role === 'assistant' && message.content && (
                          <div className="mt-1 flex gap-1">
                            <button
                              onClick={() => handleCopy(message.content, message.id)}
                              className="p-1.5 rounded hover:bg-muted/50 text-muted-foreground hover:text-foreground transition-colors"
                              title="Copier"
                            >
                              {copiedId === message.id ? (
                                <Check className="h-3.5 w-3.5 text-emerald-500" />
                              ) : (
                                <Copy className="h-3.5 w-3.5" />
                              )}
                            </button>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  ))
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Quick Prompts */}
              <div className="px-4 py-3 border-t border-border/20 bg-muted/5">
                <div className="flex flex-wrap gap-2 justify-center">
                  {QUICK_PROMPTS.map((qp) => (
                    <button
                      key={qp.id}
                      onClick={() => handleQuickPrompt(qp.prompt)}
                      disabled={isGenerating}
                      className={cn(
                        "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-border/30 transition-all",
                        qp.color,
                        "disabled:opacity-50 disabled:cursor-not-allowed"
                      )}
                    >
                      <qp.icon className="h-3.5 w-3.5" />
                      {qp.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Input Area */}
              <div className="p-4 border-t border-border/30 bg-muted/10">
                <div className="flex gap-3">
                  <Textarea
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Pose ta question sur ton histoire..."
                    disabled={isGenerating}
                    className="min-h-[60px] max-h-[120px] resize-none flex-1"
                  />
                  <Button
                    onClick={() => handleSend()}
                    disabled={!input.trim() || isGenerating}
                    className="h-[60px] px-5 bg-gradient-to-r from-violet-500 to-fuchsia-500 border-0"
                  >
                    {isGenerating ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      <Send className="h-5 w-5" />
                    )}
                  </Button>
                </div>
                <p className="text-[10px] text-muted-foreground mt-2 text-center">
                  Entrée pour envoyer • Shift+Entrée pour nouvelle ligne
                </p>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
