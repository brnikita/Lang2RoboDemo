import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";

export default tseslint.config(
  { ignores: ["dist"] },
  js.configs.recommended,
  ...tseslint.configs.strict,
  {
    files: ["src/**/*.{ts,tsx}"],
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-hooks/set-state-in-effect": "off",
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],

      // No `any` type
      "@typescript-eslint/no-explicit-any": "error",

      // Allow non-null assertions (TS strict handles null safety)
      "@typescript-eslint/no-non-null-assertion": "off",

      // No unused variables (allow underscore-prefixed)
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],

      // No console.log, allow console.warn and console.error
      "no-console": ["error", { allow: ["warn", "error"] }],

      // Max 50 lines per function (skip blanks and comments)
      "max-lines-per-function": [
        "warn",
        { max: 50, skipBlankLines: true, skipComments: true },
      ],

      // No debugger statements
      "no-debugger": "error",

      // Strict equality
      eqeqeq: ["error", "always"],

      // No var declarations
      "no-var": "error",

      // Prefer const over let when not reassigned
      "prefer-const": "error",
    },
  },
);
