'use client'

import { useState, useEffect } from 'react'
import { useStore } from '@/lib/store'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { toast } from 'sonner'
import {
  Palette,
  Download,
  Moon,
  Sun,
  RotateCcw,
  Info,
  Check,
  Brain,
  Sparkles,
  Monitor,
  Type,
  Shield,
  Heart,
  Zap,
  Database,
  Archive,
  Trash2,
  Clock,
  HardDrive,
  RefreshCw,
  Loader2,
  FileText,
  Layers,
  Cpu,
  Settings2
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  createBackup,
  listBackups,
  restoreBackup,
  deleteBackup,
  cleanupBackups,
  type BackupInfo,
  cacheGetStats,
  cacheClearType,
  cacheClearAll,
  cacheCleanupExpired,
  cacheList,
  type CacheStats,
  type CacheEntryInfo,
  type CacheType
} from '@/lib/tauri-api'

export function SettingsView() {
  const { 
    isDarkMode, 
    setIsDarkMode, 
    isModelLoaded, 
    modelPath,
    fontSize,
    setFontSize
  } = useStore()
  
  // Backup state
  const [backups, setBackups] = useState<BackupInfo[]>([])
  const [isLoadingBackups, setIsLoadingBackups] = useState(false)
  const [isCreatingBackup, setIsCreatingBackup] = useState(false)
  const [restoreDialogOpen, setRestoreDialogOpen] = useState(false)
  const [selectedBackup, setSelectedBackup] = useState<BackupInfo | null>(null)
  
  // Cache state
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null)
  const [isLoadingCache, setIsLoadingCache] = useState(false)
  const [selectedCacheType, setSelectedCacheType] = useState<CacheType>('generation')
  const [cacheEntries, setCacheEntries] = useState<CacheEntryInfo[]>([])
  
  // Load backups on mount
  useEffect(() => {
    loadBackups()
    loadCacheStats()
  }, [])
  
  async function loadBackups() {
    setIsLoadingBackups(true)
    try {
      const backupList = await listBackups()
      setBackups(backupList)
    } catch (error) {
      console.error('Failed to load backups:', error)
    } finally {
      setIsLoadingBackups(false)
    }
  }
  
  async function handleCreateBackup() {
    setIsCreatingBackup(true)
    try {
      await createBackup()
      toast.success('Backup created successfully')
      await loadBackups()
    } catch (error) {
      toast.error('Failed to create backup')
      console.error(error)
    } finally {
      setIsCreatingBackup(false)
    }
  }
  
  async function handleRestoreBackup() {
    if (!selectedBackup) return
    
    try {
      await restoreBackup(selectedBackup.filename)
      toast.success('Backup restored. Please restart the application.')
      setRestoreDialogOpen(false)
      setSelectedBackup(null)
    } catch (error) {
      toast.error('Failed to restore backup')
      console.error(error)
    }
  }
  
  async function handleDeleteBackup(backup: BackupInfo) {
    try {
      await deleteBackup(backup.filename)
      toast.success('Backup deleted')
      await loadBackups()
    } catch (error) {
      toast.error('Failed to delete backup')
      console.error(error)
    }
  }
  
  async function handleCleanupBackups() {
    try {
      const deleted = await cleanupBackups(10)
      toast.success(`Cleaned up ${deleted} old backup(s)`)
      await loadBackups()
    } catch (error) {
      toast.error('Failed to cleanup backups')
      console.error(error)
    }
  }
  
  // Cache functions
  async function loadCacheStats() {
    setIsLoadingCache(true)
    try {
      const stats = await cacheGetStats()
      setCacheStats(stats)
    } catch (error) {
      console.error('Failed to load cache stats:', error)
    } finally {
      setIsLoadingCache(false)
    }
  }
  
  async function loadCacheEntries(type: CacheType) {
    try {
      const entries = await cacheList(type)
      setCacheEntries(entries)
    } catch (error) {
      console.error('Failed to load cache entries:', error)
    }
  }
  
  async function handleClearCacheType(type: CacheType) {
    try {
      const count = await cacheClearType(type)
      toast.success(`Cleared ${count} cache entries`)
      await loadCacheStats()
      if (selectedCacheType === type) {
        setCacheEntries([])
      }
    } catch (error) {
      toast.error('Failed to clear cache')
      console.error(error)
    }
  }
  
  async function handleClearAllCache() {
    try {
      const count = await cacheClearAll()
      toast.success(`Cleared all ${count} cache entries`)
      await loadCacheStats()
      setCacheEntries([])
    } catch (error) {
      toast.error('Failed to clear cache')
      console.error(error)
    }
  }
  
  async function handleCleanupExpiredCache() {
    try {
      const count = await cacheCleanupExpired()
      toast.success(`Removed ${count} expired cache entries`)
      await loadCacheStats()
    } catch (error) {
      toast.error('Failed to cleanup cache')
      console.error(error)
    }
  }
  
  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }
  
  function formatDate(dateString: string): string {
    try {
      const date = new Date(dateString)
      return date.toLocaleString()
    } catch {
      return 'Unknown'
    }
  }
  
  function exportAllData() {
    const data = {
      exportedAt: new Date().toISOString(),
      settings: {
        isDarkMode,
        fontSize
      }
    }
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'nexastory-settings.json'
    a.click()
    // L5 fix: Only revoke after the download has had time to start
    setTimeout(() => URL.revokeObjectURL(url), 1000)
    
    toast.success('Settings exported')
  }
  
  function resetSettings() {
    setIsDarkMode(false)
    setFontSize('md')
    toast.success('Settings reset to defaults')
  }

  const fontSizes: Array<{ id: 'sm' | 'md' | 'lg' | 'xl'; name: string; description: string }> = [
    { id: 'sm', name: 'Small', description: 'Compact text size' },
    { id: 'md', name: 'Medium', description: 'Default comfortable size' },
    { id: 'lg', name: 'Large', description: 'Easier to read' },
    { id: 'xl', name: 'Extra Large', description: 'Maximum readability' },
  ]

  return (
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <header className="h-16 border-b border-border/50 flex items-center justify-between px-6 bg-gradient-to-r from-background to-muted/20">
        <div className="flex items-center gap-3">
          <Settings2 className="h-5 w-5 text-slate-500" />
          <h1 className="text-xl font-semibold">Settings</h1>
        </div>
        <Badge variant="outline" className="gap-1.5">
          <HardDrive className="h-3.5 w-3.5" />
          Desktop v0.3.0
        </Badge>
      </header>

      {/* Content */}
      <ScrollArea className="flex-1">
        <div className="p-6 max-w-4xl mx-auto space-y-8">
          
          {/* Appearance */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0 }}
          >
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Palette className="h-5 w-5 text-violet-500" />
              Appearance
            </h2>
            
            <div className="space-y-4">
              {/* Theme */}
              <Card className="border-border/50 overflow-hidden">
                <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-violet-500 via-fuchsia-500 to-rose-500" />
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Display Mode</CardTitle>
                  <CardDescription>Choose between light and dark mode</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex gap-3">
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => setIsDarkMode(false)}
                      className={cn(
                        "flex-1 flex flex-col items-center gap-3 p-5 rounded-xl border-2 transition-all",
                        !isDarkMode 
                          ? "border-amber-500 bg-amber-50 dark:bg-amber-500/10 shadow-lg shadow-amber-500/10" 
                          : "border-border hover:border-amber-500/50"
                      )}
                    >
                      <div className={cn(
                        "p-3 rounded-full",
                        !isDarkMode ? "bg-amber-500 text-white" : "bg-muted"
                      )}>
                        <Sun className="h-6 w-6" />
                      </div>
                      <span className={cn(
                        "font-medium",
                        !isDarkMode && "text-amber-600 dark:text-amber-400"
                      )}>
                        Light
                      </span>
                      {!isDarkMode && <Check className="h-4 w-4 text-amber-500" />}
                    </motion.button>
                    
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => setIsDarkMode(true)}
                      className={cn(
                        "flex-1 flex flex-col items-center gap-3 p-5 rounded-xl border-2 transition-all",
                        isDarkMode 
                          ? "border-violet-500 bg-violet-50 dark:bg-violet-500/10 shadow-lg shadow-violet-500/10" 
                          : "border-border hover:border-violet-500/50"
                      )}
                    >
                      <div className={cn(
                        "p-3 rounded-full",
                        isDarkMode ? "bg-violet-500 text-white" : "bg-muted"
                      )}>
                        <Moon className="h-6 w-6" />
                      </div>
                      <span className={cn(
                        "font-medium",
                        isDarkMode && "text-violet-600 dark:text-violet-400"
                      )}>
                        Dark
                      </span>
                      {isDarkMode && <Check className="h-4 w-4 text-violet-500" />}
                    </motion.button>
                  </div>
                </CardContent>
              </Card>

              {/* Font Size */}
              <Card className="border-border/50">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Type className="h-4 w-4" />
                    Editor Font Size
                  </CardTitle>
                  <CardDescription>Adjust text size in the editor</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {fontSizes.map((size) => (
                      <motion.button
                        key={size.id}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => setFontSize(size.id)}
                        className={cn(
                          "flex flex-col items-center gap-2 p-4 rounded-xl border-2 transition-all",
                          fontSize === size.id 
                            ? "border-primary bg-primary/5" 
                            : "border-border hover:border-primary/50"
                        )}
                      >
                        <Type className={cn(
                          "transition-all",
                          size.id === 'sm' && "h-4 w-4",
                          size.id === 'md' && "h-5 w-5",
                          size.id === 'lg' && "h-6 w-6",
                          size.id === 'xl' && "h-7 w-7"
                        )} />
                        <span className="font-medium text-sm">{size.name}</span>
                      </motion.button>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Theme Preview */}
              <Card className="border-border/50">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Preview</CardTitle>
                  <CardDescription>See how your theme looks</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="rounded-xl border border-border/50 p-4 space-y-3 bg-gradient-to-br from-background to-muted/20">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-white font-semibold text-sm">
                        A
                      </div>
                      <div className="flex-1">
                        <div className="h-3 w-24 bg-foreground rounded mb-2" />
                        <div className="h-2 w-32 bg-muted-foreground/50 rounded" />
                      </div>
                      <Button size="sm" className="bg-gradient-to-r from-violet-600 to-fuchsia-600 border-0">
                        Button
                      </Button>
                    </div>
                    <div className="flex gap-2 flex-wrap">
                      <Badge className="bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white border-0">
                        Primary
                      </Badge>
                      <Badge variant="secondary">Secondary</Badge>
                      <Badge variant="outline">Outline</Badge>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <div className="h-12 bg-card border border-border/50 rounded-lg flex items-center justify-center text-xs text-muted-foreground">Card</div>
                      <div className="h-12 bg-muted rounded-lg flex items-center justify-center text-xs text-muted-foreground">Muted</div>
                      <div className="h-12 bg-emerald-500/10 border border-emerald-500/20 rounded-lg flex items-center justify-center text-xs text-emerald-600 dark:text-emerald-400">Accent</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </motion.section>

          <Separator />

          {/* AI Model */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Brain className="h-5 w-5 text-rose-500" />
              AI Model
            </h2>
            
            <Card className="border-border/50">
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Model Status</CardTitle>
                <CardDescription>Local AI model for text generation</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-4 p-4 rounded-xl bg-gradient-to-r from-emerald-500/10 to-emerald-500/5 border border-emerald-500/20">
                  <div className="p-2.5 rounded-xl bg-emerald-500/20">
                    <Cpu className="h-6 w-6 text-emerald-500" />
                  </div>
                  <div className="flex-1">
                    <p className="font-medium">
                      {isModelLoaded ? 'Model Loaded' : 'No Model Loaded'}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {isModelLoaded && modelPath 
                        ? modelPath.split('/').pop()?.replace('.gguf', '') 
                        : 'Load a model in the Models tab'}
                    </p>
                  </div>
                  <Badge variant="default" className="bg-emerald-500 text-white">
                    {isModelLoaded ? 'Ready' : 'Offline'}
                  </Badge>
                </div>

                <div className="p-4 rounded-xl bg-primary/5 border border-primary/20">
                  <p className="text-sm">
                    <strong>Local AI Benefits:</strong>
                  </p>
                  <ul className="text-sm text-muted-foreground mt-2 space-y-1">
                    <li className="flex items-center gap-2">
                      <Shield className="h-3.5 w-3.5 text-emerald-500" />
                      100% private - your data never leaves your computer
                    </li>
                    <li className="flex items-center gap-2">
                      <Zap className="h-3.5 w-3.5 text-amber-500" />
                      No internet connection required
                    </li>
                    <li className="flex items-center gap-2">
                      <Brain className="h-3.5 w-3.5 text-violet-500" />
                      Use any GGUF model you want
                    </li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </motion.section>

          <Separator />

          {/* Data Management */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Download className="h-5 w-5 text-amber-500" />
              Data Management
            </h2>
            
            <Card className="border-border/50">
              <CardContent className="pt-6 space-y-4">
                <div className="flex items-center justify-between p-4 rounded-xl bg-muted/30">
                  <div>
                    <Label className="font-medium">Export Settings</Label>
                    <p className="text-sm text-muted-foreground mt-0.5">Download your settings as a JSON file</p>
                  </div>
                  <Button variant="outline" onClick={exportAllData}>
                    <Download className="h-4 w-4 mr-2" />
                    Export
                  </Button>
                </div>
                
                <Separator />
                
                <div className="flex items-center justify-between p-4 rounded-xl bg-muted/30">
                  <div>
                    <Label className="font-medium">Reset to Defaults</Label>
                    <p className="text-sm text-muted-foreground mt-0.5">Restore all settings to default values</p>
                  </div>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button variant="outline" className="text-destructive border-destructive/50 hover:bg-destructive/10">
                        <RotateCcw className="h-4 w-4 mr-2" />
                        Reset
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Reset Settings?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This will reset all settings to their default values.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction onClick={resetSettings} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
                          Reset
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
              </CardContent>
            </Card>
          </motion.section>

          <Separator />

          {/* Backup & Restore */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 }}
          >
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Archive className="h-5 w-5 text-emerald-500" />
              Backup & Restore
            </h2>
            
            <Card className="border-border/50">
              <CardContent className="pt-6 space-y-4">
                {/* Create Backup */}
                <div className="flex items-center justify-between p-4 rounded-xl bg-gradient-to-r from-emerald-500/10 to-teal-500/5 border border-emerald-500/20">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 rounded-xl bg-emerald-500/20">
                      <Database className="h-5 w-5 text-emerald-500" />
                    </div>
                    <div>
                      <Label className="font-medium">Create Backup</Label>
                      <p className="text-sm text-muted-foreground mt-0.5">Save a snapshot of your database</p>
                    </div>
                  </div>
                  <Button 
                    onClick={handleCreateBackup}
                    disabled={isCreatingBackup}
                    className="bg-gradient-to-r from-emerald-600 to-teal-600 border-0"
                  >
                    {isCreatingBackup ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Creating...
                      </>
                    ) : (
                      <>
                        <Archive className="h-4 w-4 mr-2" />
                        Backup Now
                      </>
                    )}
                  </Button>
                </div>
                
                {/* Backup List */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="font-medium">Available Backups</Label>
                    <div className="flex items-center gap-2">
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        onClick={loadBackups}
                        disabled={isLoadingBackups}
                      >
                        {isLoadingBackups ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <RefreshCw className="h-4 w-4" />
                        )}
                      </Button>
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={handleCleanupBackups}
                      >
                        <Trash2 className="h-3.5 w-3.5 mr-1.5" />
                        Cleanup
                      </Button>
                    </div>
                  </div>
                  
                  <div className="max-h-64 overflow-y-auto rounded-xl border border-border/50 bg-muted/20">
                    {isLoadingBackups ? (
                      <div className="flex items-center justify-center py-8">
                        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                      </div>
                    ) : backups.length === 0 ? (
                      <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                        <Archive className="h-8 w-8 mb-2 opacity-50" />
                        <p className="text-sm">No backups yet</p>
                      </div>
                    ) : (
                      <AnimatePresence>
                        {backups.map((backup, index) => (
                          <motion.div
                            key={backup.id}
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            transition={{ delay: index * 0.05 }}
                            className={cn(
                              "flex items-center justify-between p-3 hover:bg-muted/30 transition-colors",
                              index !== backups.length - 1 && "border-b border-border/30"
                            )}
                          >
                            <div className="flex items-center gap-3">
                              <div className={cn(
                                "p-2 rounded-lg",
                                backup.isAuto ? "bg-sky-500/20" : "bg-violet-500/20"
                              )}>
                                <HardDrive className={cn(
                                  "h-4 w-4",
                                  backup.isAuto ? "text-sky-500" : "text-violet-500"
                                )} />
                              </div>
                              <div>
                                <p className="text-sm font-medium truncate max-w-[200px]">
                                  {backup.filename}
                                </p>
                                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                                  <span className="flex items-center gap-1">
                                    <Clock className="h-3 w-3" />
                                    {formatDate(backup.createdAt)}
                                  </span>
                                  <span>{formatBytes(backup.sizeBytes)}</span>
                                  {backup.isAuto && (
                                    <Badge variant="outline" className="text-xs py-0 px-1.5">
                                      Auto
                                    </Badge>
                                  )}
                                </div>
                              </div>
                            </div>
                            
                            <div className="flex items-center gap-1.5">
                              <AlertDialog open={restoreDialogOpen && selectedBackup?.id === backup.id} onOpenChange={(open) => {
                                setRestoreDialogOpen(open)
                                if (!open) setSelectedBackup(null)
                              }}>
                                <AlertDialogTrigger asChild>
                                  <Button 
                                    variant="ghost" 
                                    size="sm"
                                    onClick={() => {
                                      setSelectedBackup(backup)
                                      setRestoreDialogOpen(true)
                                    }}
                                  >
                                    <RefreshCw className="h-3.5 w-3.5 mr-1" />
                                    Restore
                                  </Button>
                                </AlertDialogTrigger>
                                <AlertDialogContent>
                                  <AlertDialogHeader>
                                    <AlertDialogTitle>Restore Backup?</AlertDialogTitle>
                                    <AlertDialogDescription>
                                      This will replace your current database with the backup from {formatDate(backup.createdAt)}.
                                      A backup of your current data will be created automatically before restore.
                                    </AlertDialogDescription>
                                  </AlertDialogHeader>
                                  <AlertDialogFooter>
                                    <AlertDialogCancel onClick={() => setSelectedBackup(null)}>Cancel</AlertDialogCancel>
                                    <AlertDialogAction onClick={handleRestoreBackup}>
                                      Restore
                                    </AlertDialogAction>
                                  </AlertDialogFooter>
                                </AlertDialogContent>
                              </AlertDialog>
                              
                              <Button 
                                variant="ghost" 
                                size="icon"
                                className="h-8 w-8 hover:bg-destructive/10"
                                onClick={() => handleDeleteBackup(backup)}
                              >
                                <Trash2 className="h-3.5 w-3.5 text-muted-foreground hover:text-destructive" />
                              </Button>
                            </div>
                          </motion.div>
                        ))}
                      </AnimatePresence>
                    )}
                  </div>
                </div>
                
                {/* Auto Backup Info */}
                <div className="p-3 rounded-xl bg-sky-500/10 border border-sky-500/20">
                  <p className="text-sm text-muted-foreground">
                    <strong className="text-sky-600 dark:text-sky-400">Auto Backup:</strong> Automatic backups are created when you close the application or every 30 minutes of editing.
                  </p>
                </div>
              </CardContent>
            </Card>
          </motion.section>

          <Separator />

          {/* Cache Management */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.275 }}
          >
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Layers className="h-5 w-5 text-sky-500" />
              Cache Management
            </h2>
            
            <Card className="border-border/50">
              <CardContent className="pt-6 space-y-4">
                {/* Cache Stats */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="p-4 rounded-xl bg-gradient-to-br from-violet-500/10 to-fuchsia-500/5 border border-violet-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <FileText className="h-4 w-4 text-violet-500" />
                      <span className="text-xs text-muted-foreground">Generations</span>
                    </div>
                    <div className="text-xl font-bold text-violet-600 dark:text-violet-400">
                      {cacheStats?.entriesByType?.generation || 0}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {formatBytes(cacheStats?.sizeByType?.generation || 0)}
                    </div>
                  </div>
                  
                  <div className="p-4 rounded-xl bg-gradient-to-br from-emerald-500/10 to-teal-500/5 border border-emerald-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Database className="h-4 w-4 text-emerald-500" />
                      <span className="text-xs text-muted-foreground">DB Queries</span>
                    </div>
                    <div className="text-xl font-bold text-emerald-600 dark:text-emerald-400">
                      {cacheStats?.entriesByType?.dbquery || 0}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {formatBytes(cacheStats?.sizeByType?.dbquery || 0)}
                    </div>
                  </div>
                  
                  <div className="p-4 rounded-xl bg-gradient-to-br from-amber-500/10 to-orange-500/5 border border-amber-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Cpu className="h-4 w-4 text-amber-500" />
                      <span className="text-xs text-muted-foreground">Embeddings</span>
                    </div>
                    <div className="text-xl font-bold text-amber-600 dark:text-amber-400">
                      {cacheStats?.entriesByType?.embedding || 0}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {formatBytes(cacheStats?.sizeByType?.embedding || 0)}
                    </div>
                  </div>
                  
                  <div className="p-4 rounded-xl bg-gradient-to-br from-sky-500/10 to-blue-500/5 border border-sky-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Layers className="h-4 w-4 text-sky-500" />
                      <span className="text-xs text-muted-foreground">Total</span>
                    </div>
                    <div className="text-xl font-bold text-sky-600 dark:text-sky-400">
                      {cacheStats?.totalEntries || 0}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {formatBytes(cacheStats?.totalSizeBytes || 0)}
                    </div>
                  </div>
                </div>
                
                {/* Cache Actions */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => handleClearCacheType('generation')}
                    disabled={!cacheStats?.entriesByType?.generation}
                  >
                    <Trash2 className="h-3.5 w-3.5 mr-1.5" />
                    Clear Generations
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => handleClearCacheType('dbquery')}
                    disabled={!cacheStats?.entriesByType?.dbquery}
                  >
                    <Trash2 className="h-3.5 w-3.5 mr-1.5" />
                    Clear DB Cache
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={handleCleanupExpiredCache}
                  >
                    <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
                    Cleanup Expired
                  </Button>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button 
                        variant="outline" 
                        size="sm"
                        className="text-destructive border-destructive/50 hover:bg-destructive/10"
                        disabled={!cacheStats?.totalEntries}
                      >
                        <Trash2 className="h-3.5 w-3.5 mr-1.5" />
                        Clear All
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Clear All Cache?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This will remove all cached data including generated content, DB queries, and embeddings. 
                          This action cannot be undone.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction onClick={handleClearAllCache} className="bg-destructive text-destructive-foreground">
                          Clear All
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
                
                {/* Cache Info */}
                <div className="p-3 rounded-xl bg-muted/30">
                  <div className="flex items-start gap-3">
                    <Layers className="h-5 w-5 text-muted-foreground mt-0.5" />
                    <div className="text-sm text-muted-foreground space-y-1">
                      <p><strong>📝 Generated Content:</strong> Cached LLM responses are stored for 7 days to speed up repeated requests.</p>
                      <p><strong>🗃️ DB Queries:</strong> Database query results are cached for 1 hour to improve performance.</p>
                      <p><strong>💾 Embeddings:</strong> Text embeddings are cached for 30 days for faster similarity searches.</p>
                    </div>
                  </div>
                </div>
                
                {/* Cache Directory */}
                {cacheStats?.cacheDirectory && (
                  <div className="p-3 rounded-xl bg-sky-500/10 border border-sky-500/20">
                    <p className="text-xs text-muted-foreground">
                      <strong className="text-sky-600 dark:text-sky-400">Cache Location:</strong>{' '}
                      <code className="text-xs bg-muted/50 px-1.5 py-0.5 rounded">{cacheStats.cacheDirectory}</code>
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.section>

          <Separator />

          {/* About */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Info className="h-5 w-5 text-slate-500" />
              About NexaStory
            </h2>
            
            <Card className="border-border/50">
              <CardContent className="pt-6">
                <div className="space-y-4">
                  <div className="flex items-center gap-4">
                    <div className="p-3 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500">
                      <Sparkles className="h-8 w-8 text-white" />
                    </div>
                    <div>
                      <h3 className="font-bold text-lg">NexaStory Desktop</h3>
                      <p className="text-sm text-muted-foreground">AI-Powered Creative Writing Assistant</p>
                    </div>
                    <Badge variant="secondary" className="ml-auto">v0.3.0</Badge>
                  </div>
                  
                  <p className="text-sm text-muted-foreground">
                    NexaStory helps writers create compelling stories with AI assistance. 
                    Generate text, develop characters, build worlds, and overcome writer's block
                    with 100% local AI models.
                  </p>

                  <div className="grid grid-cols-3 gap-4 pt-2">
                    <div className="text-center p-4 rounded-xl bg-muted/30">
                      <div className="text-2xl font-bold text-violet-500">16</div>
                      <div className="text-xs text-muted-foreground">Writing Modes</div>
                    </div>
                    <div className="text-center p-4 rounded-xl bg-muted/30">
                      <div className="text-2xl font-bold text-emerald-500">4</div>
                      <div className="text-xs text-muted-foreground">AI Models</div>
                    </div>
                    <div className="text-center p-4 rounded-xl bg-muted/30">
                      <div className="text-2xl font-bold text-rose-500">100%</div>
                      <div className="text-xs text-muted-foreground">Private</div>
                    </div>
                  </div>

                  <div className="pt-2 flex items-center justify-center gap-1 text-xs text-muted-foreground">
                    Made with <Heart className="h-3 w-3 text-rose-500 mx-0.5" /> for writers everywhere
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.section>

        </div>
      </ScrollArea>
    </div>
  )
}
