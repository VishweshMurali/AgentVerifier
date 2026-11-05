import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base './' keeps every asset path relative, so dist/ works on any static host or sub-path.
// `npm run build:server` builds dist-server/ with the "api" provider (.env.server); the local
// server (server/) serves it. In dev, /api is proxied to that server on 127.0.0.1:8000.
export default defineConfig({
  base: './',
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
});
