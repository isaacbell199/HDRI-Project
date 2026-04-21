'use client'

import { useState } from 'react'
import { useStore, GENERATION_PRESETS, GenerationPreset } from '@/lib/store'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Slider } from '@/components/ui/slider'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import {
  BookOpen,
  MessageCircle,
  Zap,
  PenTool,
  Target,
  Sparkles,
  Check,
  Plus,
  Trash2,
  Copy,
  Edit,
  Star,
  Settings
} from 'lucide-react'
import { toast } from 'sonner'
import { motion, AnimatePresence } from 'framer-motion'

// Icons for each preset type
const PRESET_ICONS: Record<string, React.ReactNode> = {
  'classic-story': <BookOpen className="h-4 w-4" />,
  'dialogue': <MessageCircle className="h-4 w-4" />,
  'action': <Zap className="h-4 w-4" />,
  'poetic': <PenTool className="h-4 w-4" />,
  'precise': <Target className="h-4 w-4" />,
  'custom': <Star className="h-4 w-4" />,
}

// Color schemes for presets
const PRESET_COLORS: Record<string, string> = {
  'classic-story': 'bg-amber-500/10 border-amber-500/30 text-amber-600 dark:text-amber-400',
  'dialogue': 'bg-blue-500/10 border-blue-500/30 text-blue-600 dark:text-blue-400',
  'action': 'bg-red-500/10 border-red-500/30 text-red-600 dark:text-red-400',
  'poetic': 'bg-purple-500/10 border-purple-500/30 text-purple-600 dark:text-purple-400',
  'precise': 'bg-green-500/10 border-green-500/30 text-green-600 dark:text-green-400',
  'custom': 'bg-pink-500/10 border-pink-500/30 text-pink-600 dark:text-pink-400',
}

interface GenerationPresetSelectorProps {
  compact?: boolean
  className?: string
  showCreateButton?: boolean
}

// Default preset for creating new ones
const DEFAULT_PRESET: GenerationPreset = {
  id: '',
  name: '',
  description: '',
  temperature: 0.75,
  topP: 0.92,
  topK: 50,
  minP: 0.03,
  repeatPenalty: 1.12,
  frequencyPenalty: 0.4,
  presencePenalty: 0.3,
  maxTokens: 400,
  isCustom: true,
}

export function GenerationPresetSelector({ 
  compact = false, 
  className,
  showCreateButton = true
}: GenerationPresetSelectorProps) {
  const { 
    selectedPresetId, 
    setSelectedPresetId, 
    applyPreset,
    customPresets,
    addCustomPreset,
    removeCustomPreset,
    duplicatePreset
  } = useStore()
  
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [editingPreset, setEditingPreset] = useState<GenerationPreset | null>(null)
  const [formData, setFormData] = useState<GenerationPreset>(DEFAULT_PRESET)
  
  // Combine built-in and custom presets
  const allPresets = [...GENERATION_PRESETS, ...customPresets]
  
  const handleSelectPreset = (preset: GenerationPreset) => {
    setSelectedPresetId(preset.id)
    applyPreset(preset)
  }
  
  const handleCreateNew = () => {
    setFormData({
      ...DEFAULT_PRESET,
      id: `custom-${Date.now()}`,
      name: '',
      description: '',
      createdAt: new Date().toISOString(),
    })
    setEditingPreset(null)
    setIsCreateOpen(true)
  }
  
  const handleEdit = (preset: GenerationPreset) => {
    setFormData(preset)
    setEditingPreset(preset)
    setIsCreateOpen(true)
  }
  
  const handleDuplicate = (preset: GenerationPreset) => {
    duplicatePreset(preset)
    toast.success(`Preset "${preset.name}" duplicated!`)
  }
  
  const handleDelete = (preset: GenerationPreset) => {
    removeCustomPreset(preset.id)
    if (selectedPresetId === preset.id) {
      setSelectedPresetId('classic-story')
      applyPreset(GENERATION_PRESETS[0])
    }
    toast.success(`Preset "${preset.name}" deleted`)
  }
  
  const handleSave = () => {
    if (!formData.name.trim()) {
      toast.error('Please enter a preset name')
      return
    }
    
    if (editingPreset) {
      // Update existing
      const { updateCustomPreset } = useStore.getState()
      updateCustomPreset(editingPreset.id, formData)
      toast.success(`Preset "${formData.name}" updated!`)
    } else {
      // Create new
      addCustomPreset({
        ...formData,
        id: `custom-${Date.now()}`,
        createdAt: new Date().toISOString(),
      })
      toast.success(`Preset "${formData.name}" created!`)
    }
    
    setIsCreateOpen(false)
    setEditingPreset(null)
  }
  
  const handleCancel = () => {
    setIsCreateOpen(false)
    setEditingPreset(null)
    setFormData(DEFAULT_PRESET)
  }
  
  if (compact) {
    // Compact dropdown style for toolbar
    return (
      <div className={cn("flex items-center gap-1 flex-wrap", className)}>
        {allPresets.map((preset) => (
          <Button
            key={preset.id}
            variant="ghost"
            size="sm"
            onClick={() => handleSelectPreset(preset)}
            className={cn(
              "h-8 px-2 gap-1.5 text-xs",
              selectedPresetId === preset.id && "bg-primary/10 text-primary"
            )}
            title={preset.description}
          >
            {PRESET_ICONS[preset.isCustom ? 'custom' : preset.id] || <Star className="h-4 w-4" />}
            <span className="hidden sm:inline">{preset.name}</span>
          </Button>
        ))}
      </div>
    )
  }
  
  // Full card style for settings panel
  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-violet-500" />
          <h3 className="text-sm font-semibold">Generation Style</h3>
        </div>
        
        {showCreateButton && (
          <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
            <DialogTrigger asChild>
              <Button size="sm" variant="outline" className="h-7 gap-1 text-xs" onClick={handleCreateNew}>
                <Plus className="h-3.5 w-3.5" />
                Create Custom
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-md max-h-[90vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  {editingPreset ? 'Edit Preset' : 'Create Custom Preset'}
                </DialogTitle>
                <DialogDescription>
                  Configure your own generation parameters for specific writing styles.
                </DialogDescription>
              </DialogHeader>
              
              <div className="space-y-4 py-4">
                {/* Name */}
                <div className="space-y-2">
                  <Label htmlFor="name">Preset Name</Label>
                  <Input
                    id="name"
                    placeholder="e.g., My Thriller Style"
                    value={formData.name}
                    onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                  />
                </div>
                
                {/* Description */}
                <div className="space-y-2">
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    placeholder="What is this preset best for?"
                    value={formData.description}
                    onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                    className="h-20"
                  />
                </div>
                
                {/* Temperature */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Temperature</Label>
                    <Badge variant="outline" className="text-xs">{formData.temperature.toFixed(2)}</Badge>
                  </div>
                  <Slider
                    value={[formData.temperature]}
                    onValueChange={([value]) => setFormData(prev => ({ ...prev, temperature: value }))}
                    min={0.1}
                    max={2}
                    step={0.01}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">Lower = more focused, Higher = more creative</p>
                </div>
                
                {/* Top P */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Top P (Nucleus Sampling)</Label>
                    <Badge variant="outline" className="text-xs">{formData.topP.toFixed(2)}</Badge>
                  </div>
                  <Slider
                    value={[formData.topP]}
                    onValueChange={([value]) => setFormData(prev => ({ ...prev, topP: value }))}
                    min={0.5}
                    max={1}
                    step={0.01}
                    className="w-full"
                  />
                </div>
                
                {/* Top K */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Top K</Label>
                    <Badge variant="outline" className="text-xs">{formData.topK}</Badge>
                  </div>
                  <Slider
                    value={[formData.topK]}
                    onValueChange={([value]) => setFormData(prev => ({ ...prev, topK: value }))}
                    min={1}
                    max={100}
                    step={1}
                    className="w-full"
                  />
                </div>
                
                {/* Min P */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Min P</Label>
                    <Badge variant="outline" className="text-xs">{formData.minP.toFixed(2)}</Badge>
                  </div>
                  <Slider
                    value={[formData.minP]}
                    onValueChange={([value]) => setFormData(prev => ({ ...prev, minP: value }))}
                    min={0}
                    max={0.5}
                    step={0.01}
                    className="w-full"
                  />
                </div>
                
                {/* Repeat Penalty */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Repeat Penalty</Label>
                    <Badge variant="outline" className="text-xs">{formData.repeatPenalty.toFixed(2)}</Badge>
                  </div>
                  <Slider
                    value={[formData.repeatPenalty]}
                    onValueChange={([value]) => setFormData(prev => ({ ...prev, repeatPenalty: value }))}
                    min={1}
                    max={2}
                    step={0.01}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">Higher = more repetition avoidance</p>
                </div>
                
                {/* Frequency Penalty */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Frequency Penalty</Label>
                    <Badge variant="outline" className="text-xs">{formData.frequencyPenalty.toFixed(2)}</Badge>
                  </div>
                  <Slider
                    value={[formData.frequencyPenalty]}
                    onValueChange={([value]) => setFormData(prev => ({ ...prev, frequencyPenalty: value }))}
                    min={0}
                    max={2}
                    step={0.05}
                    className="w-full"
                  />
                </div>
                
                {/* Presence Penalty */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Presence Penalty</Label>
                    <Badge variant="outline" className="text-xs">{formData.presencePenalty.toFixed(2)}</Badge>
                  </div>
                  <Slider
                    value={[formData.presencePenalty]}
                    onValueChange={([value]) => setFormData(prev => ({ ...prev, presencePenalty: value }))}
                    min={0}
                    max={2}
                    step={0.05}
                    className="w-full"
                  />
                </div>
                
                {/* Max Tokens */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Max Tokens</Label>
                    <Badge variant="outline" className="text-xs">{formData.maxTokens}</Badge>
                  </div>
                  <Slider
                    value={[formData.maxTokens]}
                    onValueChange={([value]) => setFormData(prev => ({ ...prev, maxTokens: value }))}
                    min={100}
                    max={1000}
                    step={50}
                    className="w-full"
                  />
                </div>
              </div>
              
              <DialogFooter>
                <Button variant="outline" onClick={handleCancel}>
                  Cancel
                </Button>
                <Button onClick={handleSave}>
                  {editingPreset ? 'Update Preset' : 'Create Preset'}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        )}
      </div>
      
      <ScrollArea className="w-full">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pr-4">
          <AnimatePresence mode="popLayout">
            {allPresets.map((preset) => {
              const isCustom = preset.isCustom
              const isSelected = selectedPresetId === preset.id
              const colorKey = isCustom ? 'custom' : preset.id
              
              return (
                <motion.div
                  key={preset.id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  layout
                  className={cn(
                    "relative flex flex-col items-start p-3 rounded-xl border transition-all text-left group",
                    "hover:bg-muted/50",
                    isSelected 
                      ? cn(PRESET_COLORS[colorKey], "border-2")
                      : "bg-muted/20 border-border/50 hover:border-border"
                  )}
                >
                  <button
                    onClick={() => handleSelectPreset(preset)}
                    className="w-full text-left"
                  >
                    <div className="flex items-center gap-2 w-full">
                      <span className={cn(
                        "p-1.5 rounded-lg",
                        isSelected 
                          ? PRESET_COLORS[colorKey].split(' ')[0]
                          : "bg-muted"
                      )}>
                        {PRESET_ICONS[colorKey] || <Star className="h-4 w-4" />}
                      </span>
                      <span className="font-medium text-sm flex-1 truncate">{preset.name}</span>
                      {isSelected && (
                        <Check className="h-4 w-4 flex-shrink-0" />
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground mt-1.5 line-clamp-2">
                      {preset.description}
                    </p>
                    <div className="flex items-center gap-1.5 mt-2 flex-wrap">
                      <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4">
                        T: {preset.temperature}
                      </Badge>
                      <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4">
                        RP: {preset.repeatPenalty}
                      </Badge>
                      {isCustom && (
                        <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 text-pink-500 border-pink-300">
                          Custom
                        </Badge>
                      )}
                    </div>
                  </button>
                  
                  {/* Action buttons for custom presets */}
                  {isCustom && (
                    <div className="absolute top-2 right-2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleEdit(preset)
                        }}
                        title="Edit preset"
                      >
                        <Edit className="h-3 w-3" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleDuplicate(preset)
                        }}
                        title="Duplicate preset"
                      >
                        <Copy className="h-3 w-3" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 hover:bg-destructive/10 hover:text-destructive"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleDelete(preset)
                        }}
                        title="Delete preset"
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  )}
                  
                  {/* Duplicate button for built-in presets */}
                  {!isCustom && (
                    <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleDuplicate(preset)
                        }}
                        title="Duplicate as custom"
                      >
                        <Copy className="h-3 w-3" />
                      </Button>
                    </div>
                  )}
                </motion.div>
              )
            })}
          </AnimatePresence>
        </div>
      </ScrollArea>
    </div>
  )
}

export default GenerationPresetSelector
