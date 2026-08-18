#!/usr/bin/env python
"""Document ingestion script for Qdrant.

This script ingests markdown and text files from a docs directory into
a Qdrant Cloud collection for semantic search.

Usage:
    python scripts/ingest_docs.py --docs-dir ./docs --collection docs_assistant

Environment Variables Required:
    QDRANT_URL: Your Qdrant Cloud cluster URL
    QDRANT_API_KEY: Your Qdrant Cloud API key
    OPENAI_API_KEY: Your OpenAI API key (for embeddings)
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams


def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client from environment variables."""
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError(
            "QDRANT_URL and QDRANT_API_KEY environment variables must be set"
        )

    return QdrantClient(url=url, api_key=api_key)


def load_documents(docs_dir: Path) -> List[Dict]:
    """Load all markdown and text files from directory."""
    documents = []
    extensions = [".md", ".txt", ".markdown", ".rst"]

    for ext in extensions:
        for file_path in docs_dir.rglob(f"*{ext}"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():  # Skip empty files
                        documents.append(
                            {
                                "content": content,
                                "source": str(file_path.relative_to(docs_dir)),
                                "file_path": str(file_path),
                            }
                        )
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")

    return documents


def chunk_documents(
    documents: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Dict]:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    for doc in documents:
        doc_chunks = splitter.split_text(doc["content"])
        for i, chunk in enumerate(doc_chunks):
            chunks.append(
                {
                    "content": chunk,
                    "source": doc["source"],
                    "chunk_id": i,
                }
            )

    return chunks


def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 1536,  # text-embedding-3-small dimension
    recreate: bool = False,
) -> None:
    """Create Qdrant collection if it doesn't exist."""
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if exists and recreate:
        print(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection: {collection_name}")
    else:
        print(f"Collection {collection_name} already exists, appending documents")


def ingest_documents(
    docs_dir: Path,
    collection_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    recreate: bool = False,
) -> None:
    """Main ingestion function."""
    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize clients
    print("Initializing Qdrant client...")
    client = get_qdrant_client()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load and chunk documents
    print(f"Loading documents from {docs_dir}...")
    documents = load_documents(docs_dir)
    print(f"Found {len(documents)} documents")

    if not documents:
        print("No documents found. Exiting.")
        return

    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunks")

    # Create collection
    create_collection(client, collection_name, recreate=recreate)

    # Generate embeddings and upsert
    print("Generating embeddings and upserting to Qdrant...")
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["content"] for c in batch]
        vectors = embeddings.embed_documents(texts)

        points = [
            PointStruct(
                id=i + j,
                vector=vector,
                payload={
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                },
            )
            for j, (chunk, vector) in enumerate(zip(batch, vectors))
        ]

        client.upsert(collection_name=collection_name, points=points)
        print(f"  Upserted {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"\nSuccessfully ingested {len(chunks)} chunks to '{collection_name}'")
    print(f"You can now use the docs_assistant agent to query this collection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents into Qdrant for RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Ingest docs from ./docs directory
    python scripts/ingest_docs.py --docs-dir ./docs

    # Recreate collection with custom settings
    python scripts/ingest_docs.py --docs-dir ./docs --recreate --chunk-size 500

    # Use custom collection name
    python scripts/ingest_docs.py --docs-dir ./docs --collection my_docs
        """,
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        required=True,
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--collection",
        default="docs_assistant",
        help="Qdrant collection name (default: docs_assistant)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Chunk size in characters"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200, help="Overlap between chunks"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate collection if it exists",
    )

    args = parser.parse_args()

    if not args.docs_dir.exists():
        print(f"Error: Directory {args.docs_dir} does not exist")
        exit(1)

    ingest_documents(
        docs_dir=args.docs_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        recreate=args.recreate,
    )
