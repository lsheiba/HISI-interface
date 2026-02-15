# Repository Guidelines

## Working Rules
- **DO NOT commit and push changes without explicit approval from Leonard**
- Always ask for approval before committing changes
- Only commit after receiving explicit "yes" or "go ahead" from Leonard

## Project Structure & Module Organization
- `asr_interface/`: main Python package.
- `asr_interface/core`: shared protocols, config, and state store.
- `asr_interface/backends`: backend loaders and ASR processing implementations.
- `asr_interface/web`: FastAPI app entrypoint and HTTP endpoints.
- `asr_interface/handlers` and `asr_interface/utils`: streaming and audio utilities.
- `tests/`: pytest suite (`test_*.py`) for API and backend behavior.
- `scripts/`: frontend JS/CSS used by `index.html`; `assets/` stores icons/images; `docs/` has integration guides.

## Build, Test, and Development Commands
- `uv sync`: install runtime dependencies and create/update `.venv`.
- `uv sync --group dev`: install dev tooling (pytest, ruff, mypy, black, isort).
- `source .venv/bin/activate`: activate local environment.
- `hisi-interface serve --reload`: run the app locally (default `localhost:8000`).
- `pytest`: run the full test suite with coverage output.
- `ruff check asr_interface tests`: lint imports and code quality rules.
- `black asr_interface tests && isort asr_interface tests`: format code and imports.
- `mypy asr_interface`: run strict type checks.

## Coding Style & Naming Conventions
- Python 3.10+ with 4-space indentation and max line length 88.
- Follow Black formatting and isort `profile = "black"`.
- Use type hints broadly; mypy strict settings are enabled.
- Modules and files: `snake_case.py`; classes: `PascalCase`; functions/variables: `snake_case`.
- Keep backend-specific logic inside `asr_interface/backends/`, not in web routes.

## Testing Guidelines
- Framework: `pytest` (+ `pytest-asyncio`, `pytest-cov`).
- Naming: files `test_*.py` or `*_test.py`; functions `test_*`; classes `Test*`.
- Coverage targets `asr_interface` and generates terminal + `htmlcov/` reports.
- Add/update tests for every behavior change, especially API endpoints and backend loader paths.

## Commit & Pull Request Guidelines
- Prefer short, imperative commit subjects (e.g., `Add loading status endpoint`, `Fix backend registry bug`).
- Keep commits focused by concern; avoid mixing refactors and feature work.
- PRs should include: purpose, key changes, test evidence (`pytest`/lint output), and linked issue(s).
- Include screenshots or request/response examples for UI/API changes.
