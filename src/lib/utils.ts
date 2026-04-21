import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

/**
 * Combines class names using clsx and tailwind-merge.
 * This ensures Tailwind classes are properly merged without conflicts.
 * 
 * @param inputs - Class values to combine
 * @returns Merged class string
 * 
 * @example
 * cn('px-2 py-1', 'p-4') // Returns 'p-4' (p-4 overrides px-2 py-1)
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// ============================================================================
// Error Handling Utilities
// ============================================================================

/**
 * Error-like object with a message property
 */
export interface ErrorLike {
  message: unknown
}

/**
 * Type guard to check if a value is an Error-like object.
 * 
 * @param error - Value to check
 * @returns True if the value has a message property
 */
export function isErrorLike(error: unknown): error is ErrorLike {
  return error !== null 
    && typeof error === 'object' 
    && 'message' in error
}

/**
 * Extracts a human-readable error message from various error types.
 * Handles Error instances, strings, and error-like objects.
 * 
 * @param error - The error to extract message from
 * @returns A string representation of the error message
 * 
 * @example
 * getErrorMessage(new Error('Test')) // Returns 'Test'
 * getErrorMessage('Something failed') // Returns 'Something failed'
 * getErrorMessage({ message: 'Custom' }) // Returns 'Custom'
 */
export function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  if (typeof error === 'string') return error
  if (isErrorLike(error)) {
    return typeof error.message === 'string' 
      ? error.message 
      : String(error.message)
  }
  return 'An unknown error occurred'
}

/**
 * Extracts the stack trace from an error if available.
 * 
 * @param error - The error to extract stack from
 * @returns The stack trace string or undefined
 */
export function getErrorStack(error: unknown): string | undefined {
  if (error instanceof Error) return error.stack
  return undefined
}

/**
 * Formats an error for logging or display.
 * Includes message, stack trace, and timestamp.
 * 
 * @param error - The error to format
 * @param context - Optional context string (e.g., function name)
 * @returns Formatted error string
 */
export function formatError(error: unknown, context?: string): string {
  const message = getErrorMessage(error)
  const stack = getErrorStack(error)
  const prefix = context ? `[${context}]` : ''
  const timestamp = new Date().toISOString()
  
  let formatted = `${prefix} Error at ${timestamp}: ${message}`
  if (stack) {
    formatted += `\nStack: ${stack}`
  }
  return formatted
}
