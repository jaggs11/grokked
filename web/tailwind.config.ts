import type { Config } from "tailwindcss";
// Colors mirror the CSS vars in app/globals.css (BLUEPRINT §1.3).
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "var(--paper)", ink: "var(--ink)", scope: "var(--scope)",
        train: "var(--train)", val: "var(--val)",
        signal: "var(--signal)", noise: "var(--noise)",
      },
    },
  },
  plugins: [],
};
export default config;
