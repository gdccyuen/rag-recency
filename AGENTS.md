# AGENTS.md

## Project Overview

Python 3.12 project providing an Open WebUI plugin for hybrid vector+keyword document search via Qdrant. Core components:

- `tools/document_search.py` — Open WebUI tool plugin (main entry point)
- `utils/build_document_store.py` — CLI to ingest documents into Qdrant
- `utils/list_document_store.py` — Export document summary to CSV
- `utils/copy_qdrant_to_qdrant.py` — Migrate data between Qdrant instances

## Build / Lint / Test Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Lint (ruff)
ruff check .
ruff format .

# Run document store builder
python3 utils/build_document_store.py <input_dir> [options]

# Run list tool
python3 utils/list_document_store.py [options]

# Docker (document store builder)
docker build -f utils/Dockerfile -t build-document-store utils/
```

**No test suite exists.** There are no tests, no `pytest.ini`, no `tests/` directory. If you add tests, use pytest and place them in a `tests/` directory. Run a single test with:
```bash
pytest tests/test_file.py::test_function_name -v
```

## Ruff Configuration

Configured in `pyproject.toml` with extended rule sets:

| Rule   | Purpose                                   |
|--------|-------------------------------------------|
| `ANN`  | Enforce type annotations on all functions |
| `B`    | Catch common bug-prone patterns           |
| `I`    | Import sorting (isort-compatible)         |
| `TCH`  | Use `TYPE_CHECKING` blocks for type-only imports |
| `UP`   | Enforce modern Python syntax              |

Always run `ruff check .` and `ruff format .` before completing a task.

## Code Style

### Imports
- Order: stdlib → third-party → local (enforced by ruff `I`)
- Use explicit, specific imports: `from llama_index.core import QueryBundle, VectorStoreIndex`
- Wrap type-only imports in `TYPE_CHECKING` blocks (enforced by `TCH`)
- Conditional imports for optional dependencies go inside functions (e.g., Ollama, HuggingFace embedders)

### Types
- All function parameters and return types must be annotated (enforced by `ANN`)
- Use modern Python 3.12+ type syntax: `list[str]`, `dict[str, Any]`, `Optional[str]`
- Use `typing` imports: `Any`, `Callable`, `Optional`, `Never`
- Use `collections.abc.Iterator` instead of `typing.Iterator`

### Naming
- Functions/methods: `snake_case` — `get_embedding_model`, `build_filters`
- Classes: `PascalCase` — `DeepInfraReranker`, `CitationIndex`, `Tools`
- Constants: `UPPER_SNAKE_CASE` — `CANDIDATES_PER_RESULT`, `RESULTS_MIN`
- Private: leading underscore — `self._index`, `_postprocess_nodes`

### Formatting
- 4-space indentation
- Double-quoted strings consistently
- Two blank lines between top-level definitions; one between methods
- Lines stay under ~100 characters (soft limit)

### Docstrings
- Module-level docstrings include structured metadata (title, author, version) for Open WebUI plugin discovery
- Function docstrings use `:param:` and `:returns:` Sphinx-style notation
- Not all functions require docstrings; helpers may be undocumented

### Error Handling
- Use `try/except Exception` with descriptive `f"..."` error messages
- Raise `RuntimeError` for operational failures, `NotImplementedError` for unsupported methods
- `assert` for internal consistency checks only
- Silent `except Exception: pass` acceptable for optional metadata access

### Pydantic
- Plugin config uses inner `Valves` class (nested in `Tools`) extending `pydantic.BaseModel`
- All config fields use `Field(default=..., description=...)`
- Use `model_dump_json()` for config serialization/comparison

### Async
- Use `asyncio.Lock()` for shared state synchronization
- Async methods prefixed with `a`: `_apostprocess_nodes`, `aretrieve`
- Double-checked locking pattern for lazy initialization with caching

### CLI Scripts
- Use `argparse` with `ArgumentDefaultsHelpFormatter`
- Always include `if __name__ == "__main__":` guard
- Entry point named `main()`

## Key Dependencies

`llama-index` (core + embeddings/vector-stores integrations), `qdrant-client`, `fastembed`, `aiohttp`, `pydantic`, `pymupdf4llm`, `docling`

## Conventions for Agents

- Never commit changes unless explicitly asked
- Run `ruff check .` and `ruff format .` after any code changes
- Follow existing patterns in neighboring files before introducing new ones
- Do not add comments unless requested
- Use `Optional[X]` rather than `X | None` to match existing code style
