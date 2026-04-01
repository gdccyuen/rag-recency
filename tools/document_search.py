"""
title: Document Search
author: daradib
author_url: https://github.com/daradib/
git_url: https://github.com/daradib/openwebui-plugins.git
description: Retrieves documents from a Qdrant vector store. Supports hybrid search for agentic knowledge base RAG.
requirements: llama-index-embeddings-openai, llama-index-vector-stores-qdrant
version: 0.3.0
license: AGPL-3.0-or-later
"""


# Notes:
#
# Connection caching and citation indexing use async locking, but assume a
# single-node/worker (default). If a multi-node/worker deployment of Open WebUI
# will call this tool from separate workers, consider modifying it to use Redis
# for state synchronization.

import asyncio
import json
import math
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

import aiohttp
from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    ExactMatchFilter,
    FilterCondition,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient

# Number of candidates to retrieve if reranking:
CANDIDATES_PER_RESULT = 10
CANDIDATES_MIN = 20
CANDIDATES_MAX = 100

# Number of search results:
RESULTS_MIN = 1
RESULTS_DEFAULT = 5
RESULTS_MAX = 50

# Recency weighting:
RECENCY_ALPHA = 0.3
RECENCY_HALFLIFE_DAYS = 365.0

DEBUG = True


class EmbedRerankReranker(BaseNodePostprocessor):
    """
    Reranker using an embed-rerank service (/v1/openai/rerank).
    """

    top_n: int = Field(description="Number of top results to return")
    base_url: str = Field(description="Base URL of the embed-rerank service")

    @classmethod
    def class_name(cls) -> str:
        return "EmbedRerankReranker"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        raise NotImplementedError

    async def _apostprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        """
        Rerank nodes using embed-rerank /v1/openai/rerank endpoint.
        """
        if not nodes:
            return []

        query_str = getattr(query_bundle, "query_str", "")
        if not query_str:
            return nodes[: self.top_n]

        documents = [node.get_content()[:4096] for node in nodes]

        url = f"{self.base_url.rstrip('/')}/v1/openai/rerank"
        payload = {
            "query": query_str,
            "documents": documents,
            "top_n": self.top_n,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Embed-rerank API error (status {response.status}): {error_text}"
                    )
                result = await response.json()

        results = result.get("data", [])

        # Build reranked node list preserving original node objects
        reranked_nodes = []
        for item in results:
            idx = item.get("index", 0)
            score = item.get("score", item.get("relevance_score", 0.0))
            if 0 <= idx < len(nodes):
                node = nodes[idx]
                node.score = score
                reranked_nodes.append(node)

        return reranked_nodes


class RecencyBooster(BaseNodePostprocessor):
    """Applies Gaussian decay based on document age to reranked scores."""

    halflife_days: float = Field(
        description="Half-life in days for Gaussian recency decay"
    )
    reference_date: Optional[datetime] = Field(default=None)
    debug: bool = Field(default=False)

    @classmethod
    def class_name(cls) -> str:
        return "RecencyBooster"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        raise NotImplementedError

    async def _apostprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        if not nodes:
            return []

        ref = self.reference_date or datetime.now(timezone.utc)

        for node in nodes:
            date_str = node.metadata.get("last_modified_date")
            if date_str:
                try:
                    doc_date = datetime.strptime(date_str, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    )
                    age_days = max(0.0, (ref - doc_date).total_seconds() / 86400.0)
                    weight = math.exp(-0.5 * (age_days / self.halflife_days) ** 2)
                    original_score = node.score
                    node.score = (
                        original_score * (1 - RECENCY_ALPHA)
                        + original_score * weight * RECENCY_ALPHA
                    )

                    if self.debug:
                        file_name = node.metadata.get("file_name", "unknown")
                        print(
                            f"[RECENCY] {file_name:40s} | date={date_str} | age={age_days:7.1f}d "
                            f"| weight={weight:.4f} | rerank_score={original_score:.4f} "
                            f"| final_score={node.score:.4f}"
                        )
                except (ValueError, TypeError):
                    pass

        nodes.sort(key=lambda n: n.score, reverse=True)
        return nodes


def get_embedding_model(
    embedding_model_name: str,
    embedding_query_instruction: Optional[str],
    embed_rerank_url: str,
) -> BaseEmbedding:
    """
    Initialize and return the model for embedding.
    """
    if embedding_query_instruction:
        query_instruction = str(embedding_query_instruction).strip()
    else:
        query_instruction = None
    return OpenAIEmbedding(
        model_name=embedding_model_name,
        api_key="not-needed",
        api_base=f"{embed_rerank_url.rstrip('/')}/v1",
        query_instruction=query_instruction,
    )


def get_reranker(
    top_n: int,
    embed_rerank_url: str,
) -> BaseNodePostprocessor:
    """
    Initialize and return the reranker.
    """
    return EmbedRerankReranker(top_n=top_n, base_url=embed_rerank_url)


def get_vector_index(
    qdrant_url: str,
    qdrant_collection_name: str,
    embedding_model: str,
    embedding_query_instruction: Optional[str],
    qdrant_api_key: Optional[str],
    embed_rerank_url: str,
) -> VectorStoreIndex:
    """
    Initialize and return the VectorStoreIndex object.
    """
    # Connect to the existing Qdrant vector store.
    parsed_url = urlparse(qdrant_url, scheme="file")
    if parsed_url.scheme == "file":
        aclient = AsyncQdrantClient(path=parsed_url.path)
        kwargs = {"aclient": aclient}
        # Workaround for https://github.com/run-llama/llama_index/issues/20002
        QdrantVectorStore.use_old_sparse_encoder = lambda self, collection_name: False
    else:
        kwargs = {"url": qdrant_url, "api_key": qdrant_api_key or ""}

    vector_store = QdrantVectorStore(
        collection_name=qdrant_collection_name,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
        **kwargs,
    )

    embed_model = get_embedding_model(
        embedding_model_name=embedding_model,
        embedding_query_instruction=embedding_query_instruction,
        embed_rerank_url=embed_rerank_url,
    )

    # Create the index object from the existing vector store.
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )

    return index


def build_filters(file_name: str) -> MetadataFilters:
    """
    Build a LlamaIndex MetadataFilters object to filter by filename.
    """
    return MetadataFilters(
        filters=[ExactMatchFilter(key="file_name", value=file_name)],
        condition=FilterCondition.AND,
    )


def get_node_page(node: NodeWithScore) -> Optional[int]:
    """
    Return page number of Node.
    """
    page = node.metadata.get("page") or node.metadata.get("source")
    if not page:
        try:
            page = node.metadata["doc_items"][0]["prov"][0]["page_no"]
        except Exception:
            pass
    return page


def clean_text(text: str) -> str:
    """
    Remove unwanted formatting and artifacts from text output.
    """
    # Remove HTML tags.
    text = re.sub(r"<[a-zA-Z/][^>]*>", "", text)
    # Replace multiple blank lines with a single blank line.
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    # Remove lines with only whitespace.
    text = re.sub(r"^\s*$", "", text, flags=re.MULTILINE)
    # Remove excessive whitespace within lines.
    text = re.sub(r" +", " ", text)
    # Replace 4 or more periods with just 3 periods.
    text = re.sub(r"\.{4,}", "...", text)
    # Remove backticks.
    # Unclosed backticks seem to cause issues with citation rendering.
    text = text.replace("`", "")
    # Remove citation references.
    # Workaround for https://github.com/open-webui/open-webui/issues/17062
    # with updated regular expression for Open WebUI 0.6.33.
    text = re.sub(r"\[[\d,\s]+\]", "", text)
    # Strip leading/trailing whitespace.
    return text.strip()


def clean_node(node: NodeWithScore, citation_id: int) -> dict:
    """
    Remove internal LlamaIndex node attributes.
    """
    metadata_fields_to_keep = {
        "file_name",
        "file_type",
        "last_modified_date",
        "title",
        "total_pages",
        "headings",
    }
    cleaned_node = {
        "id": citation_id,
        "id_": node.id_,
        "metadata": {
            k: v for k, v in node.metadata.items() if k in metadata_fields_to_keep
        },
        "text": clean_text(node.text),
        "score": node.score,
    }
    page = get_node_page(node)
    if page:
        cleaned_node["metadata"]["page"] = page
    return cleaned_node


class CitationIndex:
    def __init__(self) -> None:
        self._set = set()
        self._count = 0
        self._lock = asyncio.Lock()

    async def emit_citation(
        self, node: NodeWithScore, __event_emitter__: Callable[[dict], Any]
    ) -> None:
        source_name = node.metadata.get("file_name", "Retrieved Document")
        source_name += f" ({node.id_})"
        page_number = get_node_page(node)
        if page_number:
            source_name += f" - p. {page_number}"
        await __event_emitter__(
            {
                "type": "citation",
                "data": {
                    "document": [clean_text(node.text)],
                    "metadata": [
                        {
                            "source": source_name,
                        }
                    ],
                    "source": {"name": source_name},
                },
            }
        )

    async def add_if_not_exists(
        self,
        node: NodeWithScore,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> Optional[int]:
        # Lock required to prevent race conditions in check-and-set operation
        # and to ensure citations are emitted in citation_id order.
        async with self._lock:
            if node.id_ in self._set:
                return None
            else:
                if __event_emitter__:
                    await self.emit_citation(node, __event_emitter__)
                self._set.add(node.id_)
                self._count += 1
                return self._count


class Tools:
    """
    A toolset for interacting with an existing Qdrant vector store for Retrieval-Augmented Generation
    """

    class Valves(BaseModel):
        qdrant_url: str = Field(
            default="./qdrant_db",
            description="Path to a local Qdrant directory or remote Qdrant instance.",
        )
        qdrant_collection_name: str = Field(
            default="llamacollection",
            description="Qdrant collection containing both dense vectors and sparse vectors.",
        )
        qdrant_api_key: Optional[str] = Field(
            default=None,
            description="API key for remote Qdrant instance.",
        )
        embedding_model: str = Field(
            default="mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
            description="Model for query embeddings, which should match the model used to create the text embeddings.",
        )
        embedding_query_instruction: Optional[str] = Field(
            default=None,
            description="Instruction to prepend to query before embedding, e.g., 'query:'.",
        )
        reranker_model: Optional[str] = Field(
            default="vserifsaglam/Qwen3-Reranker-4B-4bit-MLX",
            description="Model for reranking search results. When set, retrieves more candidates to improve quality.",
        )
        embed_rerank_url: str = Field(
            default="http://localhost:9000",
            description="URL for embed-rerank service. Uses /api/v1/embed for embedding and /api/v1/rerank for reranking.",
        )

    def __init__(self) -> None:
        """
        Initialize the tool and its valves.
        Disables automatic citation handling to allow for custom citation events.
        """
        self.valves = self.Valves()
        self.citation = False
        self._index = None
        self._last_config = None
        self._lock = asyncio.Lock()

    async def retrieve_documents(
        self,
        query: str,
        top_k: int = RESULTS_DEFAULT,
        file_name: Optional[str] = None,
        debug_recency: bool = False,
        __metadata__: Optional[dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Retrieve relevant documents from the Qdrant vector store using hybrid search.

        :param query: Natural language search query
        :param top_k: Number of top documents to return
        :param file_name: Filename to optionally filter results by
        :param __metadata__: Injected by Open WebUI with information about the chat
        :param __event_emitter__: Injected by Open WebUI to send events to the frontend
        """

        async def emit_status(
            description: str, done: bool = False, hidden: bool = False
        ) -> None:
            """Helper function to emit status updates."""
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": description,
                            "done": done,
                            "hidden": hidden,
                        },
                    }
                )

        if top_k < RESULTS_MIN or top_k > RESULTS_MAX:
            return f"Error: top_k must be between {RESULTS_MIN} and {RESULTS_MAX}."

        if file_name:
            parsed_filters = build_filters(file_name)
        else:
            parsed_filters = None

        filter_desc = f" in {file_name}" if file_name else ""
        await emit_status(f"Searching{filter_desc} for: {query}")

        t_query_start = time.time()

        try:
            # Cache and reuse the VectorStoreIndex object.
            # Lock required to prevent concurrent requests from closing/recreating
            # the index simultaneously, which could cause client errors.
            async with self._lock:
                current_config = self.valves.model_dump_json()
                if not self._index or self._last_config != current_config:
                    if DEBUG:
                        t0 = time.time()
                    if self._index:
                        await self._index.vector_store._aclient.close()
                    self._index = get_vector_index(
                        qdrant_url=self.valves.qdrant_url,
                        qdrant_collection_name=self.valves.qdrant_collection_name,
                        embedding_model=self.valves.embedding_model,
                        embedding_query_instruction=self.valves.embedding_query_instruction,
                        qdrant_api_key=self.valves.qdrant_api_key,
                        embed_rerank_url=self.valves.embed_rerank_url,
                    )
                    self._last_config = current_config
                    if DEBUG:
                        print(f"[DEBUG] Index init: {time.time() - t0:.2f}s")

            # Determine number of candidates to retrieve if reranking.
            if self.valves.reranker_model:
                num_candidates = max(
                    CANDIDATES_MIN,
                    min(CANDIDATES_MAX, top_k * CANDIDATES_PER_RESULT),
                )
            else:
                num_candidates = top_k

            # Create a query engine with hybrid search mode and async execution.
            retriever = self._index.as_retriever(
                vector_store_query_mode="hybrid",
                similarity_top_k=num_candidates,
                filters=parsed_filters,
                use_async=True,
            )

            if DEBUG:
                t0 = time.time()
            nodes = await retriever.aretrieve(query)
            if DEBUG:
                print(
                    f"[DEBUG] Retrieve: {time.time() - t0:.2f}s, "
                    f"{len(nodes)} candidates"
                )

            # Rerank if reranker model is configured.
            if self.valves.reranker_model and nodes:
                await emit_status(
                    f"Reranking top {top_k} from {len(nodes)} candidates..."
                )
                ranker = get_reranker(
                    top_n=top_k,
                    embed_rerank_url=self.valves.embed_rerank_url,
                )
                if DEBUG:
                    t0 = time.time()
                nodes = await ranker.apostprocess_nodes(nodes, query_str=query)
                if DEBUG:
                    print(
                        f"[DEBUG] Rerank: {time.time() - t0:.2f}s, {len(nodes)} results"
                    )

            # Apply recency boost.
            if nodes:
                booster = RecencyBooster(
                    halflife_days=RECENCY_HALFLIFE_DAYS, debug=debug_recency
                )
                nodes = await booster.apostprocess_nodes(nodes)

            if nodes:
                await emit_status("Search complete.", done=True, hidden=True)
            else:
                await emit_status("No documents found.", done=True)
                return "No relevant documents found for the query."

            if __metadata__:
                # Lock required to prevent concurrent requests from creating
                # separate CitationIndex instances, which would cause citation_id
                # collisions and loss of citation state across the conversation.
                if "document_search_citation_index" not in __metadata__:
                    async with self._lock:
                        if "document_search_citation_index" not in __metadata__:
                            __metadata__["document_search_citation_index"] = (
                                CitationIndex()
                            )
                citation_index = __metadata__["document_search_citation_index"]
            else:
                citation_index = CitationIndex()

            documents = []
            for node in nodes:
                citation_id = await citation_index.add_if_not_exists(
                    node, __event_emitter__
                )
                if citation_id:
                    documents.append(clean_node(node, citation_id=citation_id))

            if DEBUG:
                elapsed = time.time() - t_query_start
                print(f"[DEBUG] Total query: {elapsed:.2f}s")

            return json.dumps(documents)

        except Exception as e:
            error_message = f"An error occurred during search: {e}"
            await emit_status(error_message, done=True, hidden=False)
            return error_message
