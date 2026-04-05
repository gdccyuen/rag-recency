# RAG Recency

An Open WebUI plugin for hybrid vector+keyword document search via Qdrant, with recency-aware scoring that boosts more recent documents.

Forked from [daradib/openwebui-plugins](https://github.com/daradib/openwebui-plugins).

## Features

- **Hybrid search** — Dense vector + sparse BM25 retrieval via Qdrant
- **Recency weighting** — Gaussian decay boosts scores for recently modified documents (configurable half-life)
- **Reranking** — Optional semantic reranker to improve result quality
- **Incremental updates** — Only re-ingests changed files (rsync-like comparison using internal document dates)

## Quick Start

```bash
source .venv/bin/activate

# Build document store
python3 utils/build_document_store.py <input_dir> [options]

# Test search
python3 test_search.py "your query" [--top-k N] [--debug]

# Export document summary
python3 utils/list_document_store.py [options]
```

## Document Store Builder

```bash
python3 utils/build_document_store.py Docs/Basic \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection ssaskb \
  --embed-rerank-url http://127.0.0.1:9997 \
  --embedding-model mlx-community/Qwen3-Embedding-4B-4bit-DWQ \
  --chunk-size 512
```

### Date Handling

For recency scoring, `last_modified_date` is extracted from:

| File Type | Source | Fallback |
|-----------|--------|----------|
| `.pdf` | Internal PDF `/ModDate` metadata | Filesystem mtime |
| `.docx` | Internal `core_properties.modified` | Filesystem mtime |
| `.md`, `.txt` | Filesystem mtime | N/A |

## Recency Scoring

After semantic reranking, each result's score is adjusted:

```
weight = exp(-0.5 × (age_days / half_life_days)²)
final_score = rerank_score × (1 - α) + rerank_score × weight × α
```

Default: `α = 0.3`, `half_life = 365 days`. A brand-new document keeps its full score; a 1-year-old document loses ~15%; older documents are progressively discounted.

Use `--debug` with `test_search.py` to see per-result score breakdowns.

## Dependencies

- `llama-index` (core + embeddings/vector-stores integrations)
- `qdrant-client`
- `pymupdf` (PyMuPDF) — PDF internal date extraction
- `python-docx` — DOCX internal date extraction (via docling)
- `docling` — Document parsing

See `README.md` in the upstream repo for full installation instructions.
