import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import { ThemeProvider } from "next-themes";
import "./globals.css";

const inter = Inter({ 
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "NexaStory - AI Creative Writing Assistant",
  description: "A powerful creative writing assistant powered by local GGUF models. Generate text, develop characters, build worlds, and write amazing stories - 100% private and offline.",
  keywords: ["NexaStory", "creative writing", "GGUF", "AI", "writing assistant", "fiction", "storytelling", "local LLM", "Tauri"],
  authors: [{ name: "NexaStory Team" }],
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "white" },
    { media: "(prefers-color-scheme: dark)", color: "#0f0f0f" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
