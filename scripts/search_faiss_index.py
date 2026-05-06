"""
Search the persisted FAISS vectorstore for the Puls-Events RAG POC.

Input:
    vectorstore/faiss_index/

Usage:
    conda activate local-ai-rag

    # Default sample queries
    python scripts/search_faiss_index.py

    # Custom query
    python scripts/search_faiss_index.py "concert gratuit à Paris"

    # Custom query with top-k
    python scripts/search_faiss_index.py "exposition art contemporain" --top-k 5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "faiss_index"

DEFAULT_EMBEDDING_MODEL = "mistral-embed"
DEFAULT_TOP_K = 4

DEFAULT_QUERIES = [
    "concert gratuit à Paris",
    "exposition d'art contemporain",
    "événement culturel pour enfants à Paris",
    "théâtre à Paris",
]


def validate_vectorstore_dir(vectorstore_dir: Path) -> None:
    """
    Ensure the FAISS vectorstore files exist before loading.
    """
    required_files = [
        vectorstore_dir / "index.faiss",
        vectorstore_dir / "index.pkl",
        vectorstore_dir / "build_metadata.json",
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing_display = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Missing FAISS vectorstore files. Run scripts/build_faiss_index.py first.\n"
            f"Missing files:\n{missing_display}"
        )


def build_embeddings() -> MistralAIEmbeddings:
    """
    Build the same embedding client used during vectorstore creation.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing MISTRAL_API_KEY. Add it to your local .env file before running this script."
        )

    embedding_model = os.getenv("MISTRAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    return MistralAIEmbeddings(model=embedding_model)


def load_vectorstore() -> FAISS:
    """
    Load the persisted FAISS vectorstore.

    allow_dangerous_deserialization=True is required by LangChain because
    FAISS metadata is stored in a pickle file. This is acceptable here because
    the index.pkl file is generated locally by our own build script.
    """
    validate_vectorstore_dir(VECTORSTORE_DIR)

    embeddings = build_embeddings()

    return FAISS.load_local(
        folder_path=str(VECTORSTORE_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def format_metadata(metadata: dict[str, Any]) -> str:
    """
    Format compact event metadata for terminal output.
    """
    lines = [
        f"Title: {metadata.get('title', 'N/A')}",
        f"Venue: {metadata.get('venue_name', 'N/A')}",
        f"Address: {metadata.get('address', 'N/A')}",
        f"City: {metadata.get('city', 'N/A')}",
        f"Dates: {metadata.get('first_timing_begin', 'N/A')} -> {metadata.get('last_timing_begin', 'N/A')}",
        f"Event ID: {metadata.get('event_id', 'N/A')}",
        f"Chunk: {metadata.get('chunk_index', 'N/A')}",
    ]

    source_url = metadata.get("source_url")
    registration_url = metadata.get("registration_url")

    if source_url:
        lines.append(f"Source URL: {source_url}")

    if registration_url:
        lines.append(f"Registration URL: {registration_url}")

    return "\n".join(lines)


def search_query(vectorstore: FAISS, query: str, top_k: int) -> None:
    """
    Run one semantic search query and print ranked results.
    """
    print("=" * 100)
    print(f"Query: {query}")
    print("=" * 100)

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    if not results:
        print("No results returned.")
        return

    for rank, (document, score) in enumerate(results, start=1):
        print(f"\n--- Result {rank} | FAISS distance: {score:.4f} ---")
        print(format_metadata(document.metadata))

        preview = document.page_content.replace("\n", " ").strip()
        if len(preview) > 500:
            preview = preview[:500].rstrip() + "..."

        print("\nChunk preview:")
        print(preview)

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the Puls-Events FAISS vectorstore."
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Optional search query. If omitted, default smoke-test queries are used.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return per query. Default: {DEFAULT_TOP_K}",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")

    print("Loading Puls-Events FAISS vectorstore")
    print(f"Vectorstore directory: {VECTORSTORE_DIR}")

    vectorstore = load_vectorstore()

    queries = [args.query] if args.query else DEFAULT_QUERIES

    for query in queries:
        search_query(
            vectorstore=vectorstore,
            query=query,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()