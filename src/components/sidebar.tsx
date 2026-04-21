'use client'

import { useStore } from '@/lib/store'
import { cn } from '@/lib/utils'
import { 
  BookOpen, 
  PenTool, 
  Globe2, 
  Settings, 
  ChevronRight,
  Sparkles,
  Brain,
  Zap
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip'
import { Badge } from '@/components/ui/badge'
import { motion, AnimatePresence } from 'framer-motion'

const navItems = [
  { 
    id: 'projects', 
    label: 'Projects', 
    icon: BookOpen, 
    color: 'text-violet-500',
    activeBg: 'bg-gradient-to-r from-violet-500/20 to-violet-500/5',
    activeText: 'text-violet-600 dark:text-violet-400',
    activeBorder: 'bg-violet-500',
    hoverBg: 'hover:bg-violet-500/10',
    shadowColor: 'shadow-violet-500/20'
  },
  { 
    id: 'editor', 
    label: 'Editor', 
    icon: PenTool, 
    color: 'text-emerald-500',
    activeBg: 'bg-gradient-to-r from-emerald-500/20 to-emerald-500/5',
    activeText: 'text-emerald-600 dark:text-emerald-400',
    activeBorder: 'bg-emerald-500',
    hoverBg: 'hover:bg-emerald-500/10',
    shadowColor: 'shadow-emerald-500/20'
  },
  { 
    id: 'world', 
    label: 'World', 
    icon: Globe2, 
    color: 'text-amber-500',
    activeBg: 'bg-gradient-to-r from-amber-500/20 to-amber-500/5',
    activeText: 'text-amber-600 dark:text-amber-400',
    activeBorder: 'bg-amber-500',
    hoverBg: 'hover:bg-amber-500/10',
    shadowColor: 'shadow-amber-500/20'
  },
  { 
    id: 'models', 
    label: 'Models', 
    icon: Brain, 
    color: 'text-rose-500',
    activeBg: 'bg-gradient-to-r from-rose-500/20 to-rose-500/5',
    activeText: 'text-rose-600 dark:text-rose-400',
    activeBorder: 'bg-rose-500',
    hoverBg: 'hover:bg-rose-500/10',
    shadowColor: 'shadow-rose-500/20'
  },
  { 
    id: 'settings', 
    label: 'Settings', 
    icon: Settings, 
    color: 'text-slate-500',
    activeBg: 'bg-gradient-to-r from-slate-500/20 to-slate-500/5',
    activeText: 'text-slate-600 dark:text-slate-400',
    activeBorder: 'bg-slate-500',
    hoverBg: 'hover:bg-slate-500/10',
    shadowColor: 'shadow-slate-500/20'
  },
] as const

export function Sidebar() {
  const { 
    currentView, 
    setCurrentView, 
    sidebarCollapsed, 
    setSidebarCollapsed,
    currentProject,
    isModelLoaded
  } = useStore()

  return (
    <TooltipProvider delayDuration={0}>
      <motion.aside 
        initial={false}
        animate={{ width: sidebarCollapsed ? 72 : 224 }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
        className="h-screen bg-gradient-to-b from-card to-card/95 border-r border-border/50 flex flex-col backdrop-blur-sm relative"
      >
        {/* Decorative top accent */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-violet-500 via-fuchsia-500 to-rose-500" />
        
        {/* Header */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-border/30">
          <AnimatePresence mode="wait">
            {!sidebarCollapsed && (
              <motion.div 
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
                className="flex items-center gap-2.5"
              >
                <div className="relative">
                  <Sparkles className="h-6 w-6 text-violet-500" />
                  <Zap className="h-3 w-3 text-amber-400 absolute -top-1 -right-1" />
                </div>
                <div>
                  <span className="font-bold text-xl tracking-tight bg-gradient-to-r from-violet-600 to-fuchsia-600 dark:from-violet-400 dark:to-fuchsia-400 bg-clip-text text-transparent">
                    NexaStory
                  </span>
                  <p className="text-[10px] text-muted-foreground -mt-0.5">Desktop Edition</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="h-9 w-9 rounded-full hover:bg-primary/10 hover:text-primary transition-all"
          >
            <motion.div
              animate={{ rotate: sidebarCollapsed ? 180 : 0 }}
              transition={{ duration: 0.2 }}
            >
              <ChevronRight className="h-4 w-4" />
            </motion.div>
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-4 px-2 space-y-1.5">
          {navItems.map((item, index) => {
            const Icon = item.icon
            const isActive = currentView === item.id
            
            // Disable editor and world if no project is open
            const isDisabled = (item.id === 'editor' || item.id === 'world') && !currentProject
            
            return (
              <Tooltip key={item.id}>
                <TooltipTrigger asChild>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.2, delay: index * 0.05 }}
                  >
                    <Button
                      variant="ghost"
                      className={cn(
                        "w-full justify-start gap-3 h-11 relative overflow-hidden rounded-xl transition-all duration-200",
                        sidebarCollapsed && "justify-center px-0",
                        isDisabled && "opacity-40 cursor-not-allowed",
                        isActive && !isDisabled && `${item.activeBg} ${item.activeText} shadow-sm ${item.shadowColor}`,
                        !isActive && !isDisabled && `hover:bg-muted/60 hover:shadow-sm`
                      )}
                      onClick={() => !isDisabled && setCurrentView(item.id)}
                      disabled={isDisabled}
                    >
                      {/* Active indicator */}
                      {isActive && !isDisabled && (
                        <motion.div
                          layoutId="activeIndicator"
                          className={cn(
                            "absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 rounded-r-full",
                            item.activeBorder
                          )}
                          transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                        />
                      )}
                      
                      <Icon className={cn(
                        "h-5 w-5 shrink-0 transition-all",
                        isActive ? item.color : "text-muted-foreground",
                        !sidebarCollapsed && "ml-1"
                      )} />
                      
                      <AnimatePresence mode="wait">
                        {!sidebarCollapsed && (
                          <motion.span
                            initial={{ opacity: 0, width: 0 }}
                            animate={{ opacity: 1, width: 'auto' }}
                            exit={{ opacity: 0, width: 0 }}
                            transition={{ duration: 0.2 }}
                            className="font-medium truncate"
                          >
                            {item.label}
                          </motion.span>
                        )}
                      </AnimatePresence>
                    </Button>
                  </motion.div>
                </TooltipTrigger>
                {sidebarCollapsed && (
                  <TooltipContent side="right" className="flex items-center gap-2">
                    <span>{item.label}</span>
                    {isDisabled && (
                      <Badge variant="outline" className="text-xs py-0 px-1.5 border-amber-500/50 text-amber-600">
                        Select project
                      </Badge>
                    )}
                  </TooltipContent>
                )}
              </Tooltip>
            )
          })}
        </nav>

        {/* Model Status Indicator */}
        {isModelLoaded && (
          <div className="px-3 py-2">
            <div className={cn(
              "flex items-center gap-2 p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20",
              sidebarCollapsed && "justify-center"
            )}>
              <div className="relative">
                <Brain className="h-4 w-4 text-emerald-500" />
                <span className="absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              </div>
              {!sidebarCollapsed && (
                <span className="text-xs font-medium text-emerald-600 dark:text-emerald-400">Model Ready</span>
              )}
            </div>
          </div>
        )}

        {/* Project Info */}
        {currentProject && !sidebarCollapsed && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-3 border-t border-border/30"
          >
            <div className="text-xs text-muted-foreground mb-1 font-medium">Current Project</div>
            <div className="font-semibold truncate text-sm">{currentProject.name}</div>
            {currentProject.genre && (
              <Badge variant="secondary" className="mt-1.5 text-xs">
                {currentProject.genre}
              </Badge>
            )}
          </motion.div>
        )}
      </motion.aside>
    </TooltipProvider>
  )
}
