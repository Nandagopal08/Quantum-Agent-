import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// API target for `npm run dev`: the local FastAPI server, or a Colab tunnel URL via VITE_API_PROXY.
const target = process.env.VITE_API_PROXY || "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: { output: { manualChunks: { recharts: ["recharts"], vendor: ["react", "react-dom", "react-router-dom", "axios"] } } },
  },
  server: { port: 5173, proxy: { "/api": { target, changeOrigin: true } } },
  test: { environment: "jsdom", globals: true, setupFiles: "./src/test-setup.js" },
});
