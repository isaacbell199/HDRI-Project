import type { NextConfig } from "next";

// For Tauri desktop builds, use static export
// For CI/development, allow API routes to work
const shouldExport = process.env.NEXT_STATIC_EXPORT === 'true';

const nextConfig: NextConfig = {
  // Only use static export for Tauri builds
  ...(shouldExport ? { output: "export" } : {}),
  // Disable image optimization for static export
  images: {
    unoptimized: true,
  },
  // Trailing slashes for static hosting
  trailingSlash: true,
  // React strict mode
  reactStrictMode: true,
  // TypeScript strict checking enabled
  typescript: {
    ignoreBuildErrors: false,
  },
  // Transpile Tauri packages
  transpilePackages: [
    '@tauri-apps/api',
    '@tauri-apps/plugin-dialog',
    '@tauri-apps/plugin-fs',
    '@tauri-apps/plugin-shell',
  ],
  // Turbopack config (Next.js 16 default)
  turbopack: {
    resolveAlias: {
      '@tauri-apps/api/core': '@tauri-apps/api/core',
      '@tauri-apps/api/event': '@tauri-apps/api/event',
    },
  },
  // Webpack fallback for Tauri packages
  webpack: (config, { isServer }) => {
    if (isServer) {
      // Provide empty mocks for Tauri APIs on server side
      config.resolve = config.resolve || {};
      config.resolve.alias = config.resolve.alias || {};
      config.resolve.alias['@tauri-apps/api/core'] = false;
      config.resolve.alias['@tauri-apps/api/event'] = false;
    }
    return config;
  },
};

export default nextConfig;
