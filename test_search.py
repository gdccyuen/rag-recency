#!/usr/bin/env python3

import asyncio
import json
import sys

sys.path.insert(0, ".")
from tools.document_search import Tools


async def main() -> None:
    tool = Tools()

    # --embed-rerank flag uses the embed-rerank service for both embed and rerank
    use_embed_rerank = "--embed-rerank" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if use_embed_rerank:
        tool.valves.qdrant_url = "http://localhost:6333"
        tool.valves.embed_rerank_url = "http://localhost:9000"
        tool.valves.embedding_model = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
        tool.valves.reranker_model = "reranker"
        provider = "embed-rerank (localhost:9000)"
    else:
        tool.valves.qdrant_url = "http://localhost:6333"
        tool.valves.ollama_base_url = "http://localhost:11434"
        tool.valves.embedding_model = "qwen3-embedding:0.6b"
        provider = "ollama (localhost:11434)"

    query = args[0] if len(args) > 0 else "what is the application form?"
    top_k = int(args[1]) if len(args) > 1 else 3

    print(f"Provider: {provider}")
    print(f"Query: {query}")
    print(f"Top-k: {top_k}")
    print()

    result = await tool.retrieve_documents(query, top_k=top_k)

    print()
    try:
        docs = json.loads(result)
    except json.JSONDecodeError:
        print(f"Error: {result}")
        sys.exit(1)

    for doc in docs:
        print(f"--- Result {doc['id']} (score: {doc['score']}) ---")
        print(f"  File: {doc['metadata'].get('file_name', 'N/A')}")
        print(f"  Text: {doc['text'][:200]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
