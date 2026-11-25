---
name: notebook-coding-standards
description: >-
  Enforces strict Jupyter notebook and Python module coding conventions for this
  repository: all notebook imports MUST live in the top imports cell (Cell 1)
  only (never in the middle of a notebook), no `def` function definitions in
  notebooks, no `.iloc` usage anywhere, and NO nested functions, nested classes
  inside functions, or anonymous `lambda` functions anywhere (all functions must
  be top-level or class methods, single-purpose, and explicitly named). Activate
  before creating or editing any `.ipynb`, `src/*.py`, or `tests/*.py` file.
---

# Notebook & Module Coding Standards

Follow these non-negotiable rules whenever creating, editing, or refactoring any Jupyter notebook (`.ipynb`), helper module (`src/*.py`), or test file (`tests/*.py`) in this repository:

## 1. All Notebook Imports at the Top Only (Cell 1)
- **NEVER** place `import ...` or `from ... import ...` statements in any code cell other than the top imports cell (Cell 1).
- Whenever a later cell needs a new library, helper function, or class, add its import to Cell 1 and reference the symbol in the downstream cell.

## 2. No Function Definitions in Notebooks
- **NEVER** define functions (`def` or `async def`) or `lambda` expressions inside any `.ipynb` file.
- Place all reusable, single-purpose, type-hinted helper and plotting functions in `src/*.py` and import them in Cell 1.

## 3. No `.iloc` Anywhere
- **NEVER** use `.iloc` in notebooks or `src/*.py` modules (not even in comments or docstrings).
- Use `.loc`, boolean masks, `.to_numpy()[...]`, or index slicing instead.

## 4. No Nested Functions, Nested Classes, or Lambdas — Single-Purpose & Explicitly Named Only
- **NEVER** define a function (`def` or `async def`) or `class` inside another function anywhere in `src/*.py` or `tests/*.py`.
- **NEVER** use anonymous `lambda` expressions anywhere in `src/*.py`, `tests/*.py`, or `notebooks/*.ipynb`; replace them with explicit top-level named helpers (with `functools.partial` or `operator.itemgetter` where argument binding is needed).
- Every function MUST have a **single clear responsibility** and an **explicit, descriptive name** (no vague abbreviations like `_cats`, `_make_row`, or multi-responsibility 200+ line monoliths).
