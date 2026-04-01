#!/usr/bin/env python3

import argparse
import asyncio
import json
import sys

from tools.document_search import Tools


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test document search tool")
    parser.add_argument("query", help="Search query string")
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of results to return"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print recency score calculation details"
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_arguments()

    tool = Tools()

    tool.valves.qdrant_url = "http://localhost:6333"
    tool.valves.qdrant_collection_name = "ssaskb"
    tool.valves.embed_rerank_url = "http://localhost:9000"
    tool.valves.embedding_model = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
    tool.valves.reranker_model = "reranker"

    print("Provider: embed-rerank (localhost:9000)")
    print(f"Query: {args.query}")
    print(f"Top-k: {args.top_k}")
    print()

    result = await tool.retrieve_documents(
        args.query, top_k=args.top_k, debug_recency=args.debug
    )

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
