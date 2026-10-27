import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies /api to the FastAPI backend so the browser only ever
// talks to one origin in development. In production the frontend uses
// VITE_API_BASE_URL (see src/api.js).
export default defineConfig({
  plugins: [react()],
  build: {
    // Recharts ships as one large third-party chunk (~540 kB). The application
    // code itself is ~30 kB. The limit is raised so the warning reflects the
    // real situation instead of a library we do not control.
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        // Split the heavy mapping and charting libraries out of the main
        // bundle so the initial payload stays small.
        manualChunks: {
          leaflet: ['leaflet', 'react-leaflet'],
          recharts: ['recharts'],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_PROXY_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.js',
    css: false,
  },
})
