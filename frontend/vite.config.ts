import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base './' keeps every asset path relative, so dist/ works on any static host or sub-path.
export default defineConfig({
  base: './',
  plugins: [react()],
});
