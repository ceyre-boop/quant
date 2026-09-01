import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Builds into ../_site so the GitHub Pages workflow can keep treating _site as the
// artifact root. deploy.yml copies data/ in alongside; base stays './' so the same
// bundle works on Pages (project subpath), Vercel (root) and file:// preview.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: './',
  build: {
    outDir: '../_site',
    emptyOutDir: false,   // deploy.yml also cp's data/ into _site; never wipe it
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: {
      // Local dev talks to scripts/live_signals_server.py on :8765, so the app
      // uses same-origin relative paths in both dev and prod.
      // Only the dev server proxies. When live_signals_server.py serves the
      // built app it already owns these routes at the root, so lib/api.ts drops
      // the /api prefix outside dev.
      '/api':  { target: 'http://localhost:8765', changeOrigin: true, rewrite: p => p.replace(/^\/api/, '') },
      '/data': { target: 'http://localhost:8765', changeOrigin: true },
    },
  },
})
