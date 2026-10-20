import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    port: 5174,
    // The FastAPI portal backend. Same-origin in production (it serves dist/).
    proxy: { '/api': { target: 'http://127.0.0.1:8090', changeOrigin: true } },
  },
});
