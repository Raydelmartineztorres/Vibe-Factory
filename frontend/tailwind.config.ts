import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        primary: {
          DEFAULT: "#2563eb",
          foreground: "#ffffff",
        },
        danger: {
          DEFAULT: "#dc2626",
          foreground: "#ffffff",
        },
        success: {
          DEFAULT: "#16a34a",
          foreground: "#ffffff",
        },
        warning: {
          DEFAULT: "#fbbf24",
          foreground: "#0f172a",
        },
      },
      borderRadius: {
        xl: "1rem",
      },
      boxShadow: {
        panel: "0 20px 45px rgba(15, 23, 42, 0.18)",
      },
    },
  },
  plugins: [],
};

export default config;


