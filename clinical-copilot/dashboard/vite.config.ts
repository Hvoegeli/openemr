import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'node:path';

// We build into `clinical-copilot/app/web/dashboard-build/` so Copilot's
// FastAPI can mount the directory as static files at `/dashboard`. The
// `base: '/dashboard/'` is critical — without it the bundle's <script src>
// references would resolve at `/` and 404 in production.
//
// The dev server proxy at `/api` lets `npm run dev` (Vite on :5173) call
// Copilot at :8000 without CORS. Production hits same-origin /api.
export default defineConfig({
  plugins: [react()],
  base: '/dashboard/',
  build: {
    outDir: resolve(__dirname, '..', 'app', 'web', 'dashboard-build'),
    emptyOutDir: true,
    sourcemap: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
});
