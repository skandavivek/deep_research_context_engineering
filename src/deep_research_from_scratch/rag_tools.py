
"""RAG Tools for Docs Assistant.

This module provides Qdrant-based retrieval tools for the docs assistant,
including vector search and document retrieval functionality.
"""

import os
from typing_extensions import Annotated

from langchain_core.tools import InjectedToolArg, tool
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

# ===== CONFIGURATION =====

# Lazy initialization for LangGraph Platform compatibility
_qdrant_client = None
_embeddings = None

# Default collection name
DEFAULT_COLLECTION = "docs_assistant"


def get_qdrant_client() -> QdrantClient:
    """Get or initialize Qdrant client lazily."""
    global _qdrant_client
    if _qdrant_client is None:
        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API_KEY")
        if not url or not api_key:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY environment variables must be set"
            )
        _qdrant_client = QdrantClient(url=url, api_key=api_key)
    return _qdrant_client


def get_embeddings() -> OpenAIEmbeddings:
    """Get or initialize embeddings model lazily."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings


# ===== RAG TOOL =====


@tool(parse_docstring=True)
def qdrant_search(
    query: str,
    top_k: Annotated[int, InjectedToolArg] = 5,
    collection_name: Annotated[str, InjectedToolArg] = DEFAULT_COLLECTION,
) -> str:
    """Search documentation using semantic similarity.

    Retrieves relevant document chunks from the vector database based on
    the query. Use this to find information in the indexed documentation.

    Args:
        query: The search query to find relevant documentation
        top_k: Number of results to return
        collection_name: Name of the Qdrant collection to search

    Returns:
        Formatted string of retrieved document chunks with sources
    """
    client = get_qdrant_client()
    embeddings = get_embeddings()

    # Generate query embedding
    query_vector = embeddings.embed_query(query)

    # Search Qdrant using query_points (new API)
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    )

    if not results.points:
        return "No relevant documents found for the query."

    # Format results
    formatted_output = "Retrieved Documentation:\n\n"
    for i, point in enumerate(results.points, 1):
        payload = point.payload
        formatted_output += f"--- DOCUMENT {i} (Score: {point.score:.3f}) ---\n"
        formatted_output += f"Source: {payload.get('source', 'Unknown')}\n"
        formatted_output += f"Content:\n{payload.get('content', '')}\n\n"
        formatted_output += "-" * 60 + "\n"

    return formatted_output
