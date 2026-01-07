import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { readFileSync } from 'fs'
import { resolve } from 'path'
import { execSync } from 'child_process'

// Execute a shell command and return trimmed stdout, or null on failure.
function safeExecTrim(command: string): string | null {
  try {
    return execSync(command).toString().trim() || null
  } catch {
    return null
  }
}

const packageJson = JSON.parse(
  readFileSync(resolve(__dirname, 'package.json'), 'utf-8')
)

// Build-time metadata
const buildDate = process.env.BUILD_DATE || new Date().toISOString()

// Resolve a short commit SHA for build metadata.
// Precedence:
// 1) GIT_COMMIT_SHA_SHORT environment variable (CI can set this explicitly)
// 2) git rev-parse --short HEAD  -> returns "git:<sha>"
// 3) hg id -i (Mercurial)        -> returns "hg:<sha>"
// 4) 'unknown' (neither VCS available)
function resolveCommitSha(): string {
  const envSha = process.env.GIT_COMMIT_SHA_SHORT
  if (envSha) return envSha

  // Try Git first, then Mercurial.
  const candidates: Array<{ cmd: string; prefix: string }> = [
    { cmd: 'git rev-parse --short HEAD', prefix: 'git:' },
    { cmd: 'hg id -i', prefix: 'hg:' }
  ]

  for (const { cmd, prefix } of candidates) {
    const out = safeExecTrim(cmd)
    if (out) return `${prefix}${out}`
  }
  // Final fallback when no VCS is present or accessible.
  return 'unknown'
}

// Compute once at config load; injected below via define.
const gitSha = resolveCommitSha()

// Check if this is an FB (internal) build
// Set FB_BUILD=true environment variable when building for internal deployment
const isFbBuild = process.env.FB_BUILD === 'true'

// Internal wiki URL - only included in FB builds
const internalWikiUrl = 'https://www.internalfb.com/wiki/Triton_Core/Tools/TritonParse/'

export default defineConfig({
  plugins: [
    tailwindcss(),
    react(),
    {
      name: 'strip-module-attr',
      enforce: 'post',
      apply: 'build', // only apply this plugin during build
      transformIndexHtml(html) {
        return html.replace(
          /<script\s+type=["']module["']\s+([^>]*?)src=/g,
          '<script defer $1src='
        )
      }
    }
  ],

  base: './',
  build: {
    sourcemap: true,
    rollupOptions: {
      output: {
        format: 'iife',
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]'
      }
    }
  },
  define: {
    'import.meta.env.PACKAGE_VERSION': JSON.stringify(packageJson.version),
    'import.meta.env.PACKAGE_BUILD_DATE': JSON.stringify(buildDate),
    'import.meta.env.GIT_COMMIT_SHA_SHORT': JSON.stringify(gitSha),
    // FB build flag - defaults to false, set FB_BUILD=true for internal builds
    'import.meta.env.IS_FB_BUILD': JSON.stringify(isFbBuild),
    // Internal wiki URL - only meaningful when IS_FB_BUILD is true
    'import.meta.env.INTERNAL_WIKI_URL': JSON.stringify(isFbBuild ? internalWikiUrl : '')
  }
})
