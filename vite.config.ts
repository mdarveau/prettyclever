import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  worker: { format: 'es' },
  // Allow serving through Cloudflare quick tunnels (`cloudflared tunnel --url ...`).
  server: { allowedHosts: ['.trycloudflare.com'] },
});
