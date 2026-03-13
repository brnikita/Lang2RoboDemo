# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Principles

### Code Quality
- **DRY** — no duplicated logic. Extract shared behavior into reusable functions/classes.
- **SOLID** — single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion.
- **Limited method length** — methods must not exceed 20 lines. Decompose into smaller focused functions.
- **Clear module interfaces** — every module exposes a well-defined public API. Internals stay private. Use `__all__` in Python, barrel exports in TypeScript.

### Type Safety & Validation
- **Python**: Use **Pydantic** models for all data structures, API inputs/outputs, and config. Strict mode preferred. No raw dicts for structured data.
- **TypeScript/JavaScript**: **TypeScript strict mode**. No `any`. Define interfaces/types for all data shapes.

### Documentation
- **JSDoc**: All public JS/TS functions must have JSDoc with `@param`, `@returns`, and `@throws` tags.
- **Python docstrings**: All public Python functions and classes must have Google-style docstrings with Args, Returns, and Raises sections.

### Testing
- **TDD** — write tests first, then implementation. Red → Green → Refactor cycle.
- Every public function has at least one unit test. Integration tests for cross-module flows.
- Use `pytest` for Python, `vitest` or `jest` for TypeScript.
