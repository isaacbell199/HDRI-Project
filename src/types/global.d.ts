/// <reference types="react" />

declare namespace NodeJS {
  interface ProcessEnv {
    NEXT_PUBLIC_APP_VERSION: string
  }
}

declare module '*.css' {
  const content: { [className: string]: string }
  export default content
}

declare module '*.svg' {
  const content: React.FC<React.SVGProps<SVGSVGElement>>
  export default content
}

// Tauri event types
interface TauriEvents {
  'generation-chunk': {
    content: string
    done: boolean
  }
}

// Extend Window for Tauri
declare global {
  interface Window {
    __TAURI__?: {
      invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>
      event: {
        listen: <T>(event: string, handler: (event: { payload: T }) => void) => Promise<() => void>
      }
    }
  }
}

export {}
