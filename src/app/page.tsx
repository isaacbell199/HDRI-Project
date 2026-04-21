'use client'

import { useEffect } from 'react'
import { useTheme } from 'next-themes'
import { useStore } from '@/lib/store'
import { Sidebar } from '@/components/sidebar'
import { ProjectsView } from '@/components/views/projects-view'
import { EditorView } from '@/components/views/editor-view'
import { WorldView } from '@/components/views/world-view'
import { ModelsView } from '@/components/views/models-view'
import { SettingsView } from '@/components/views/settings-view'
import { Toaster } from '@/components/ui/sonner'

export default function NexaStory() {
  const { currentView, isDarkMode, setIsDarkMode } = useStore()
  const { setTheme, resolvedTheme } = useTheme()

  // Sync store theme with next-themes on mount and when isDarkMode changes
  useEffect(() => {
    setTheme(isDarkMode ? 'dark' : 'light')
  }, [isDarkMode, setTheme])

  // Sync next-themes with store on mount (for system preference)
  // H10 fix: Added resolvedTheme as dependency so the effect actually runs
  useEffect(() => {
    if (resolvedTheme) {
      setIsDarkMode(resolvedTheme === 'dark')
    }
  }, [resolvedTheme])

  return (
    <div className="min-h-screen bg-background flex">
      <Sidebar />
      <main className="flex-1 flex flex-col min-h-screen overflow-hidden">
        {currentView === 'projects' && <ProjectsView />}
        {currentView === 'editor' && <EditorView />}
        {currentView === 'world' && <WorldView />}
        {currentView === 'models' && <ModelsView />}
        {currentView === 'settings' && <SettingsView />}
      </main>
      <Toaster />
    </div>
  )
}
