#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from tqdm import tqdm


def get_qdrant_client(qdrant_url: str, api_key: str | None = None) -> QdrantClient:
    """
    Connect to Qdrant and return the client object.
    Supports both local (file-based) and remote Qdrant instances.
    """
    parsed_url = urlparse(qdrant_url, scheme="file")
    if parsed_url.scheme == "file":
        client = QdrantClient(path=parsed_url.path)
    else:
        client = QdrantClient(url=qdrant_url, api_key=api_key)
    return client


def get_qdrant_stats(
    qdrant_url: str,
    collection_name: str,
    api_key: str | None = None,
    batch_size: int = 1000,
) -> None:
    """
    Query Qdrant and print statistics about the stored documents.
    """
    print(f"Connecting to Qdrant at '{qdrant_url}' ... ", end="")
    client = get_qdrant_client(qdrant_url, api_key)
    print("connection successful.")

    if not client.collection_exists(collection_name):
        print(f"Error: Collection '{collection_name}' does not exist.")
        return

    total_points = client.count(collection_name=collection_name).count
    print(f"Total points (chunks): {total_points}")
    print()

    file_chunks: dict[str, list[int]] = defaultdict(list)
    chunk_lengths: list[int] = []

    offset = None
    with tqdm(
        total=total_points,
        desc="Scanning collection",
        miniters=batch_size,
        disable=None,
    ) as pbar:
        while True:
            points, next_page_offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=["file_path", "file_name", "_node_content"],
                with_vectors=False,
            )
            for point in points:
                file_path = point.payload.get("file_path", "unknown")
                file_name = point.payload.get("file_name", "unknown")
                file_key = f"{file_name} ({file_path})"

                node_content = point.payload.get("_node_content", "")
                if node_content:
                    try:
                        node_data = json.loads(node_content)
                        text = node_data.get("text", "")
                    except json.JSONDecodeError:
                        text = ""
                else:
                    text = ""
                text_len = len(text) if isinstance(text, str) else 0
                chunk_lengths.append(text_len)
                file_chunks[file_key].append(text_len)

            pbar.update(len(points))

            if next_page_offset is None:
                break
            offset = next_page_offset

    # Compute statistics
    num_files = len(file_chunks)
    chunks_per_file = [len(chunks) for chunks in file_chunks.values()]
    max_chunks = max(chunks_per_file) if chunks_per_file else 0
    avg_chunks = sum(chunks_per_file) / num_files if num_files else 0

    max_chunk_chars = max(chunk_lengths) if chunk_lengths else 0
    avg_chunk_chars = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    min_chunk_chars = min(chunk_lengths) if chunk_lengths else 0

    # Find file with most chunks
    largest_file = (
        max(file_chunks.items(), key=lambda x: len(x[1]))
        if file_chunks
        else ("N/A", [])
    )

    # Find largest chunk
    largest_chunk_file = "N/A"
    if chunk_lengths:
        max_idx = chunk_lengths.index(max_chunk_chars)
        # Reverse lookup to find file
        count = 0
        for file_key, lengths in file_chunks.items():
            for _ in lengths:
                if count == max_idx:
                    largest_chunk_file = file_key
                    break
                count += 1
            if largest_chunk_file != "N/A":
                break

    print("=" * 60)
    print("Collection Statistics")
    print("=" * 60)
    print(f"  Collection:       {collection_name}")
    print(f"  Total files:      {num_files}")
    print(f"  Total chunks:     {total_points}")
    print()
    print("Chunks per file:")
    print(f"  Max:              {max_chunks}")
    print(f"  Avg:              {avg_chunks:.1f}")
    print(f"  File with most:   {largest_file[0]} ({len(largest_file[1])} chunks)")
    print()
    print("Characters per chunk:")
    print(f"  Max:              {max_chunk_chars:,}")
    print(f"  Avg:              {avg_chunk_chars:,.0f}")
    print(f"  Min:              {min_chunk_chars:,}")
    print(f"  File with largest: {largest_chunk_file}")
    print()

    # Chunks over 4096 chars
    over_limit = [length for length in chunk_lengths if length > 4096]
    if over_limit:
        print(
            f"  Chunks > 4096 chars: {len(over_limit)} ({len(over_limit) / len(chunk_lengths) * 100:.1f}%)"
        )
        print(f"  Largest over-limit:  {max(over_limit):,} chars")
    else:
        print("  Chunks > 4096 chars: 0")


def main() -> None:
    """Main entry point of the script."""
    parser = argparse.ArgumentParser(
        description="Print statistics about documents stored in Qdrant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="./qdrant_db",
        help="Path to a local Qdrant directory or remote Qdrant instance URL",
    )
    parser.add_argument(
        "--qdrant-collection",
        type=str,
        default="llamacollection",
        help="Name of the Qdrant collection to query",
    )
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=None,
        help="API key for remote Qdrant instance (optional)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of points to retrieve per scroll request",
    )

    args = parser.parse_args()

    get_qdrant_stats(
        qdrant_url=args.qdrant_url,
        collection_name=args.qdrant_collection,
        api_key=args.qdrant_api_key,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
