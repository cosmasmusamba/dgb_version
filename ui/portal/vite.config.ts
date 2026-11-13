import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    allowedHosts: [".trycloudflare.com"],
    proxy: {
      "/auth": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/inference": "http://127.0.0.1:8000",
      "/stream": "http://127.0.0.1:8000",
      "/v1": "http://127.0.0.1:8000"
    }
  }
});
