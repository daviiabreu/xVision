import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [tailwindcss(), react()],
  test: {
    environment: "jsdom",
    env: { VITE_API_URL: "http://api.test" },
    globals: true,
    setupFiles: ["./src/tests/setup.ts"],
    include: ["src/**/*.test.ts?(x)"],
    coverage: {
      provider: "v8",
      reporter: ["text", "cobertura"],
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/tests/**", "src/main.tsx", "src/vite-env.d.ts"],
      thresholds: { statements: 90, branches: 82, functions: 90, lines: 90 },
    },
  },
});
