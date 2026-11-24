---
name: notebook-coding-standards
description: >-
  Enforces strict Jupyter notebook and Python module coding conventions for this
  repository: all notebook imports MUST live in the top imports cell (Cell 1)
  only (never in the middle of a notebook), no `def` function definitions in
  notebooks, and no `.iloc` usage anywhere. Activate before creating or editing
  any `.ipynb` or `src/*.py` file.
---

# Notebook & Module Coding Standards

Follow these non-negotiable rules whenever creating, editing, or refactoring any Jupyter notebook (`.ipynb`) or helper module (`src/*.py`) in this repository:

## 1. All Notebook Imports at the Top Only (Cell 1)
- **NEVER** place `import ...` or `from ... import ...` statements in any code cell other than the top imports cell (Cell 1).
- Whenever a later cell needs a new library, helper function, or class, add its import to Cell 1 and reference the symbol in the downstream cell.

## 2. No Function Definitions in Notebooks
- **NEVER** define functions (`def` or `async def`) inside any `.ipynb` file.
- Place all reusable, single-purpose, type-hinted helper and plotting functions in `src/*.py` and import them in Cell 1.

## 3. No `.iloc` Anywhere
- **NEVER** use `.iloc` in notebooks or `src/*.py` modules (not even in comments or docstrings).
- Use `.loc`, boolean masks, `.to_numpy()[...]`, or index slicing instead.
