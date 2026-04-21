import { useState, useEffect } from 'react'
import { useStore, Project } from '@/lib/store'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Progress } from '@/components/ui/progress'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Plus, MoreVertical, Trash2, Edit, BookOpen, FileText,
  User, Sparkles, LayoutGrid, List, Search,
  Clock, Globe2, Clapperboard
} from 'lucide-react'
import { toast } from 'sonner'
import { motion } from 'framer-motion'
import {
  getProjects,
  getProject,
  deleteProject,
  updateProject,
  getLoreNotes,
  type ProjectWithCounts
} from '@/lib/tauri-api'
import { NewProjectDialog } from '@/components/new-project-dialog'

// Genre options with icons and colors
const genreOptions = [
  { id: 'fantasy', name: 'Fantasy', color: 'bg-violet-500', textColor: 'text-violet-500' },
  { id: 'sci-fi', name: 'Science Fiction', color: 'bg-cyan-500', textColor: 'text-cyan-500' },
  { id: 'romance', name: 'Romance', color: 'bg-pink-500', textColor: 'text-pink-500' },
  { id: 'mystery', name: 'Mystery', color: 'bg-amber-500', textColor: 'text-amber-500' },
  { id: 'thriller', name: 'Thriller', color: 'bg-red-500', textColor: 'text-red-500' },
  { id: 'horror', name: 'Horror', color: 'bg-slate-700', textColor: 'text-slate-500' },
  { id: 'adventure', name: 'Adventure', color: 'bg-emerald-500', textColor: 'text-emerald-500' },
  { id: 'historical', name: 'Historical', color: 'bg-orange-500', textColor: 'text-orange-500' },
  { id: 'literary', name: 'Literary Fiction', color: 'bg-indigo-500', textColor: 'text-indigo-500' },
  { id: 'dystopian', name: 'Dystopian', color: 'bg-stone-600', textColor: 'text-stone-500' },
  { id: 'young-adult', name: 'Young Adult', color: 'bg-teal-500', textColor: 'text-teal-500' },
  { id: 'crime', name: 'Crime', color: 'bg-rose-600', textColor: 'text-rose-500' },
]

export function ProjectsView() {
  const { setCurrentView, setCurrentProject } = useStore()
  const [projects, setProjects] = useState<ProjectWithCounts[]>([])
  const [loading, setLoading] = useState(true)
  const [wizardOpen, setWizardOpen] = useState(false)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [editDialogOpen, setEditDialogOpen] = useState(false)
  const [selectedProject, setSelectedProject] = useState<ProjectWithCounts | null>(null)
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [searchQuery, setSearchQuery] = useState('')
  
  // Edit form state
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [editGenre, setEditGenre] = useState<string | null>(null)
  const [isSaving, setIsSaving] = useState(false)

  useEffect(() => {
    loadProjects()
  }, [])

  async function loadProjects() {
    setLoading(true)
    try {
      // 100% Desktop - Use Tauri backend only
      const data = await getProjects()
      setProjects(data)
    } catch (error) {
      console.error('Failed to load projects:', error)
      setProjects([])
      toast.error('Failed to load projects from database')
    } finally {
      setLoading(false)
    }
  }

  async function handleProjectCreated(projectId: string) {
    setWizardOpen(false)

    try {
      // Fetch the actual project data from backend
      const projectData = await getProject(projectId)

      if (projectData) {
        // Load lore notes from database
        let loreNotes: { id: string; projectId: string; title: string; category: string | null; content: string | null; tags: string | null; createdAt: string; updatedAt: string }[] = []
        try {
          loreNotes = await getLoreNotes(projectId)
        } catch (e) {
          console.warn('Could not load lore notes:', e)
        }

        // Use the actual project data
        const fullProject: Project = {
          ...projectData,
          chapters: [],
          characters: [],
          locations: [],
          loreNotes: loreNotes,
          settings: null
        }

        setCurrentProject(fullProject)
        localStorage.setItem('nexastory_current_project', JSON.stringify(fullProject))

        // Refresh the projects list
        loadProjects()

        // Redirect to World Studio for configuration
        setCurrentView('world')
      } else {
        toast.error('Failed to load created project')
        loadProjects()
      }
    } catch (error) {
      console.error('Failed to fetch created project:', error)
      toast.error('Failed to load created project')
      loadProjects()
    }
  }

  async function openProject(project: ProjectWithCounts) {
    // Load lore notes from database
    let loreNotes: { id: string; projectId: string; title: string; category: string | null; content: string | null; tags: string | null; createdAt: string; updatedAt: string }[] = []
    try {
      loreNotes = await getLoreNotes(project.id)
    } catch (e) {
      console.warn('Could not load lore notes:', e)
    }

    const fullProject: Project = {
      ...project,
      chapters: [],
      characters: [],
      locations: [],
      loreNotes: loreNotes,
      settings: null
    }
    setCurrentProject(fullProject)

    // Save to localStorage for AI components
    localStorage.setItem('nexastory_current_project', JSON.stringify(fullProject))

    setCurrentView('editor')
    toast.success(`Opened "${project.name}"`)
  }

  async function openWorldStudio(project: ProjectWithCounts) {
    // Load lore notes from database
    let loreNotes: { id: string; projectId: string; title: string; category: string | null; content: string | null; tags: string | null; createdAt: string; updatedAt: string }[] = []
    try {
      loreNotes = await getLoreNotes(project.id)
    } catch (e) {
      console.warn('Could not load lore notes:', e)
    }

    const fullProject: Project = {
      ...project,
      chapters: [],
      characters: [],
      locations: [],
      loreNotes: loreNotes,
      settings: null
    }
    setCurrentProject(fullProject)
    
    // Save to localStorage for AI components
    localStorage.setItem('nexastory_current_project', JSON.stringify(fullProject))
    
    setCurrentView('world')
    toast.success(`Configuring "${project.name}" in World Studio`)
  }

  async function handleDeleteProject() {
    if (!selectedProject) return
    
    try {
      // 100% Desktop - Use Tauri backend only
      await deleteProject(selectedProject.id)
      setProjects(projects.filter(p => p.id !== selectedProject.id))
      toast.success('Project deleted')
    } catch (error) {
      console.error('Failed to delete project:', error)
      toast.error('Failed to delete project')
    } finally {
      setDeleteDialogOpen(false)
      setSelectedProject(null)
    }
  }

  // Open edit dialog with project data
  function openEditDialog(project: ProjectWithCounts) {
    setSelectedProject(project)
    setEditName(project.name)
    setEditDescription(project.description || '')
    setEditGenre(project.genre)
    setEditDialogOpen(true)
  }

  // Handle edit project save
  async function handleEditProject() {
    if (!selectedProject) return
    if (!editName.trim()) {
      toast.error('Project name is required')
      return
    }

    setIsSaving(true)
    try {
      const updatedProject = await updateProject(selectedProject.id, {
        ...selectedProject,
        name: editName.trim(),
        description: editDescription.trim() || null,
        genre: editGenre,
      })
      
      // Update local state
      setProjects(projects.map(p => 
        p.id === selectedProject.id 
          ? { ...p, name: editName.trim(), description: editDescription.trim() || null, genre: editGenre }
          : p
      ))
      
      toast.success('Project updated')
      setEditDialogOpen(false)
      setSelectedProject(null)
    } catch (error) {
      console.error('Failed to update project:', error)
      toast.error('Failed to update project')
    } finally {
      setIsSaving(false)
    }
  }

  const filteredProjects = projects.filter(p => 
    p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    p.genre?.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const getGenreInfo = (genreId: string | null) => {
    return genreOptions.find(g => g.id === genreId) || null
  }

  return (
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <header className="h-14 border-b border-border/50 flex items-center justify-between px-6 bg-gradient-to-r from-background to-muted/20 shrink-0">
        <div className="flex items-center gap-4">
          <div>
            <h1 className="text-xl font-bold tracking-tight">Projects</h1>
            <p className="text-xs text-muted-foreground">{projects.length} stories</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Search */}
          <div className="relative">
            <Search className="h-4 w-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search projects..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 w-48 h-8 bg-muted/50 border-0 focus-visible:ring-1 text-sm"
            />
          </div>
          
          {/* View Mode Toggle */}
          <div className="flex items-center border rounded-lg p-0.5 bg-muted/30">
            <Button
              variant={viewMode === 'grid' ? 'secondary' : 'ghost'}
              size="icon"
              className="h-7 w-7"
              onClick={() => setViewMode('grid')}
            >
              <LayoutGrid className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant={viewMode === 'list' ? 'secondary' : 'ghost'}
              size="icon"
              className="h-7 w-7"
              onClick={() => setViewMode('list')}
            >
              <List className="h-3.5 w-3.5" />
            </Button>
          </div>
          
          {/* New Project Button */}
          <Button 
            onClick={() => setWizardOpen(true)} 
            className="gap-2 h-8 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-700 hover:to-fuchsia-700 border-0"
          >
            <Plus className="h-4 w-4" />
            New Project
          </Button>
        </div>
      </header>

      {/* Content */}
      <ScrollArea className="flex-1">
        <div className="p-6">
          {loading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[1, 2, 3].map((i) => (
                <Card key={i} className="overflow-hidden">
                  <CardHeader className="pb-3">
                    <Skeleton className="h-6 w-3/4" />
                    <Skeleton className="h-4 w-1/2" />
                  </CardHeader>
                  <CardContent>
                    <Skeleton className="h-16 w-full" />
                    <div className="flex gap-2 mt-4">
                      <Skeleton className="h-6 w-16" />
                      <Skeleton className="h-6 w-16" />
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : filteredProjects.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-[60vh] text-center">
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.4 }}
              >
                <div className="relative">
                  <BookOpen className="h-24 w-24 text-muted-foreground/30" />
                  <Sparkles className="h-8 w-8 text-violet-500 absolute -top-2 -right-2" />
                </div>
              </motion.div>
              <h2 className="text-2xl font-bold mt-6 mb-2">No Projects Yet</h2>
              <p className="text-muted-foreground mb-6 max-w-md">
                {searchQuery ? 'No projects match your search.' : 'Create your first writing project to start using AI-powered writing tools.'}
              </p>
              <Button onClick={() => setWizardOpen(true)} className="gap-2 bg-gradient-to-r from-violet-600 to-fuchsia-600 border-0">
                <Plus className="h-4 w-4" />
                Create Your First Project
              </Button>
            </div>
          ) : viewMode === 'grid' ? (
            <motion.div 
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
              initial="hidden"
              animate="visible"
              variants={{
                visible: { transition: { staggerChildren: 0.05 } }
              }}
            >
              {filteredProjects.map((project) => {
                const genreInfo = getGenreInfo(project.genre)
                return (
                  <motion.div
                    key={project.id}
                    variants={{
                      hidden: { opacity: 0, y: 20 },
                      visible: { opacity: 1, y: 0 }
                    }}
                  >
                    <Card 
                      className="cursor-pointer group hover:shadow-lg hover:shadow-primary/5 transition-all duration-300 overflow-hidden border-border/50 hover:border-primary/30"
                      onClick={() => openProject(project)}
                    >
                      {/* Genre color accent */}
                      {genreInfo && (
                        <div className={cn("h-1.5", genreInfo.color)} />
                      )}
                      
                      <CardHeader className="pb-2">
                        <div className="flex items-start justify-between">
                          <div className="space-y-1 flex-1">
                            <CardTitle className="text-base line-clamp-1 group-hover:text-primary transition-colors">
                              {project.name}
                            </CardTitle>
                            {genreInfo && (
                              <Badge variant="outline" className={cn("text-[10px]", genreInfo.textColor)}>
                                {genreInfo.name}
                              </Badge>
                            )}
                          </div>
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                              <Button variant="ghost" size="icon" className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity">
                                <MoreVertical className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem onClick={(e) => { e.stopPropagation(); openWorldStudio(project) }}>
                                <Clapperboard className="h-4 w-4 mr-2" /> World Studio
                              </DropdownMenuItem>
                              <DropdownMenuItem onClick={(e) => { e.stopPropagation(); openEditDialog(project) }}>
                                <Edit className="h-4 w-4 mr-2" /> Edit Details
                              </DropdownMenuItem>
                              <DropdownMenuSeparator />
                              <DropdownMenuItem 
                                className="text-destructive"
                                onClick={(e) => { e.stopPropagation(); setSelectedProject(project); setDeleteDialogOpen(true) }}
                              >
                                <Trash2 className="h-4 w-4 mr-2" /> Delete
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </div>
                      </CardHeader>
                      <CardContent className="pb-4">
                        {project.description && (
                          <p className="text-xs text-muted-foreground line-clamp-2 mb-3">
                            {project.description}
                          </p>
                        )}
                        
                        {/* Stats */}
                        <div className="grid grid-cols-4 gap-1 mb-3">
                          <div className="text-center p-1.5 rounded-lg bg-muted/50">
                            <FileText className="h-3.5 w-3.5 mx-auto mb-0.5 text-violet-500" />
                            <span className="text-[10px] font-medium">{project.chapterCount}</span>
                          </div>
                          <div className="text-center p-1.5 rounded-lg bg-muted/50">
                            <User className="h-3.5 w-3.5 mx-auto mb-0.5 text-emerald-500" />
                            <span className="text-[10px] font-medium">{project.characterCount}</span>
                          </div>
                          <div className="text-center p-1.5 rounded-lg bg-muted/50">
                            <Globe2 className="h-3.5 w-3.5 mx-auto mb-0.5 text-amber-500" />
                            <span className="text-[10px] font-medium">{project.locationCount}</span>
                          </div>
                          <div className="text-center p-1.5 rounded-lg bg-muted/50">
                            <BookOpen className="h-3.5 w-3.5 mx-auto mb-0.5 text-rose-500" />
                            <span className="text-[10px] font-medium">{project.loreNoteCount}</span>
                          </div>
                        </div>
                        
                        {/* Word count progress */}
                        {project.wordCount && project.wordCount > 0 && (
                          <div className="space-y-1">
                            <div className="flex justify-between text-[10px]">
                              <span className="text-muted-foreground">Progress</span>
                              <span className="font-medium">{project.wordCount.toLocaleString()} words</span>
                            </div>
                            <Progress value={(project.wordCount / 50000) * 100} className="h-1" />
                          </div>
                        )}
                        
                        {/* Updated date */}
                        <p className="text-[10px] text-muted-foreground mt-2 flex items-center gap-1">
                          <Clock className="h-2.5 w-2.5" />
                          Updated {new Date(project.updatedAt).toLocaleDateString()}
                        </p>
                      </CardContent>
                    </Card>
                  </motion.div>
                )
              })}
            </motion.div>
          ) : (
            // List view
            <div className="space-y-2">
              {filteredProjects.map((project) => {
                const genreInfo = getGenreInfo(project.genre)
                return (
                  <Card 
                    key={project.id} 
                    className="cursor-pointer hover:bg-muted/30 transition-all"
                    onClick={() => openProject(project)}
                  >
                    <CardContent className="p-3 flex items-center gap-4">
                      <div className={cn("w-10 h-10 rounded-lg flex items-center justify-center shrink-0", genreInfo?.color || "bg-muted")}>
                        <BookOpen className="h-5 w-5 text-white" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <h3 className="font-medium text-sm truncate">{project.name}</h3>
                          {genreInfo && (
                            <Badge variant="outline" className={cn("text-[10px]", genreInfo.textColor)}>
                              {genreInfo.name}
                            </Badge>
                          )}
                        </div>
                        {project.description && (
                          <p className="text-xs text-muted-foreground truncate">{project.description}</p>
                        )}
                      </div>
                      <div className="hidden md:flex items-center gap-4 text-xs text-muted-foreground">
                        <span className="flex items-center gap-1">
                          <FileText className="h-3 w-3" /> {project.chapterCount}
                        </span>
                        <span className="flex items-center gap-1">
                          <User className="h-3 w-3" /> {project.characterCount}
                        </span>
                        <span className="flex items-center gap-1">
                          <Globe2 className="h-3 w-3" /> {project.locationCount}
                        </span>
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          )}
        </div>
      </ScrollArea>

      {/* New Project Dialog */}
      <NewProjectDialog
        open={wizardOpen}
        onClose={() => setWizardOpen(false)}
        onComplete={handleProjectCreated}
      />

      {/* Edit Project Dialog */}
      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Project</DialogTitle>
            <DialogDescription>
              Update the project details below.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="edit-name">Name</Label>
              <Input
                id="edit-name"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                placeholder="Project name"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="edit-description">Description</Label>
              <Textarea
                id="edit-description"
                value={editDescription}
                onChange={(e) => setEditDescription(e.target.value)}
                placeholder="Project description (optional)"
                rows={3}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="edit-genre">Genre</Label>
              <select
                id="edit-genre"
                value={editGenre || ''}
                onChange={(e) => setEditGenre(e.target.value || null)}
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                <option value="">Select a genre (optional)</option>
                {genreOptions.map((genre) => (
                  <option key={genre.id} value={genre.id}>
                    {genre.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleEditProject} disabled={isSaving || !editName.trim()}>
              {isSaving ? 'Saving...' : 'Save Changes'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Project?</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{selectedProject?.name}"? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={handleDeleteProject}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
