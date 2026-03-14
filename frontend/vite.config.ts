import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: parseInt(process.env.FRONTEND_PORT ?? "5173"),
    proxy: {
      "/api": `http://${process.env.BACKEND_HOST ?? "127.0.0.1"}:${process.env.BACKEND_PORT ?? "8000"}`,
    },
  },
});
