'use client'

import { useState } from 'react'
import { useStore } from '@/lib/store'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { FileText, Loader2 } from 'lucide-react'
import { toast } from 'sonner'
import { createProject, isTauri } from '@/lib/tauri-api'

interface NewProjectDialogProps {
  open: boolean
  onClose: () => void
  onComplete: (projectId: string) => void
}

export function NewProjectDialog({ open, onClose, onComplete }: NewProjectDialogProps) {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async () => {
    if (!title.trim()) {
      toast.error('Project title is required')
      return
    }

    if (title.trim().length < 2) {
      toast.error('Title must be at least 2 characters')
      return
    }

    setIsSubmitting(true)

    try {
      if (isTauri()) {
        const project = await createProject({
          name: title.trim(),
          description: description.trim() || undefined,
        })

        toast.success('Project created! Configure it in World Studio.')
        onComplete(project.id)
      } else {
        // Browser demo mode
        toast.success('Project created (demo mode)')
        onComplete('demo-project')
      }

      // Reset form
      setTitle('')
      setDescription('')
    } catch (error) {
      console.error('Failed to create project:', error)
      toast.error('Failed to create project')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleClose = () => {
    setTitle('')
    setDescription('')
    onClose()
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5 text-violet-500" />
            New Project
          </DialogTitle>
          <DialogDescription>
            Create a new writing project. You'll configure all settings in World Studio.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="title">Project Title *</Label>
            <Input
              id="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Enter your story title..."
              maxLength={100}
            />
            <p className="text-[10px] text-muted-foreground">
              {title.length}/100 characters
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Briefly describe your story idea..."
              rows={3}
              maxLength={500}
            />
            <p className="text-[10px] text-muted-foreground">
              {description.length}/500 characters
            </p>
          </div>

          <div className="p-3 rounded-lg bg-violet-500/10 border border-violet-500/20">
            <p className="text-xs text-violet-400">
              💡 After creating the project, you'll be redirected to World Studio to configure characters, locations, tone, style, and generation presets.
            </p>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={isSubmitting || !title.trim() || title.trim().length < 2}
            className="bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-700 hover:to-fuchsia-700 border-0"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Creating...
              </>
            ) : (
              'Create & Go to Studio'
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default NewProjectDialog
