"""
Build the FAISS vector database for the Puls-Events RAG POC.

Input:
    data/processed/events_clean.csv

Output:
    vectorstore/faiss_index/index.faiss
    vectorstore/faiss_index/index.pkl
    vectorstore/faiss_index/build_metadata.json

Usage:
    conda activate local-ai-rag
    python scripts/build_faiss_index.py
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as exc:
    raise ImportError(
        "Missing dependency: langchain-text-splitters. "
        "Install it with: pip install -U langchain-text-splitters"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "events_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "vectorstore" / "faiss_index"
BUILD_METADATA_PATH = OUTPUT_DIR / "build_metadata.json"

DEFAULT_EMBEDDING_MODEL = "mistral-embed"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

REQUIRED_COLUMNS = [
    "event_id",
    "title",
    "text_for_embedding",
    "city",
    "venue_name",
    "address",
    "latitude",
    "longitude",
    "first_timing_begin",
    "last_timing_begin",
]


def clean_value(value: Any) -> Any:
    """
    Convert pandas/NumPy missing values to None and normalize simple string values.
    """
    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    return value


def load_events(input_csv: Path) -> pd.DataFrame:
    """
    Load and validate the cleaned event dataset.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input dataset: {input_csv}")

    df = pd.read_csv(input_csv)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {input_csv}: {missing_columns}")

    if df.empty:
        raise ValueError(f"Input dataset is empty: {input_csv}")

    empty_text_mask = df["text_for_embedding"].isna() | (
        df["text_for_embedding"].astype(str).str.strip() == ""
    )
    empty_text_count = int(empty_text_mask.sum())

    if empty_text_count > 0:
        raise ValueError(
            f"Found {empty_text_count} rows with empty text_for_embedding. "
            "Fix preprocessing before building the vectorstore."
        )

    return df


def row_to_document(row: pd.Series) -> Document:
    """
    Convert one cleaned event row into a LangChain Document.
    Metadata is kept compact but sufficient for retrieval, explanation, and Step 4 RAG.
    """
    metadata = {
        "event_id": clean_value(row.get("event_id")),
        "title": clean_value(row.get("title")),
        "city": clean_value(row.get("city")),
        "venue_name": clean_value(row.get("venue_name")),
        "address": clean_value(row.get("address")),
        "latitude": clean_value(row.get("latitude")),
        "longitude": clean_value(row.get("longitude")),
        "first_timing_begin": clean_value(row.get("first_timing_begin")),
        "last_timing_begin": clean_value(row.get("last_timing_begin")),
        "timings_count": clean_value(row.get("timings_count")),
        "registration_url": clean_value(row.get("registration_url")),
        "source_url": clean_value(row.get("source_url")),
    }

    # Remove keys that do not exist in the CSV or contain no useful value.
    metadata = {key: value for key, value in metadata.items() if value is not None}

    return Document(
        page_content=str(row["text_for_embedding"]).strip(),
        metadata=metadata,
    )


def build_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convert the full DataFrame into LangChain Documents.
    """
    documents = [row_to_document(row) for _, row in df.iterrows()]

    if not documents:
        raise ValueError("No documents were created from the cleaned dataset.")

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split event texts into chunks before embedding.

    The chunk size is intentionally moderate:
    - small enough for targeted retrieval
    - large enough to preserve useful event context
    """
    chunk_size = int(os.getenv("FAISS_CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
    chunk_overlap = int(os.getenv("FAISS_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created. Check text_for_embedding content.")

    chunk_counter_by_event: dict[str, int] = defaultdict(int)

    for chunk in chunks:
        event_id = str(chunk.metadata.get("event_id", "unknown"))
        chunk_index = chunk_counter_by_event[event_id]
        chunk_counter_by_event[event_id] += 1

        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_id"] = f"{event_id}::chunk_{chunk_index}"
        chunk.metadata["chunk_text_length"] = len(chunk.page_content)

    return chunks


def reset_output_dir(output_dir: Path) -> None:
    """
    Recreate the vectorstore output directory.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def build_vectorstore(chunks: list[Document]) -> FAISS:
    """
    Create the FAISS vectorstore with Mistral embeddings.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing MISTRAL_API_KEY. Add it to your local .env file before running this script."
        )

    embedding_model = os.getenv("MISTRAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    embeddings = MistralAIEmbeddings(
        model=embedding_model,
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vectorstore


def write_build_metadata(
    *,
    df: pd.DataFrame,
    documents: list[Document],
    chunks: list[Document],
    output_dir: Path,
) -> None:
    """
    Write a small audit file describing the vectorstore build.
    """
    embedding_model = os.getenv("MISTRAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    chunk_size = int(os.getenv("FAISS_CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
    chunk_overlap = int(os.getenv("FAISS_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))

    chunk_counts_by_event: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        event_id = str(chunk.metadata.get("event_id", "unknown"))
        chunk_counts_by_event[event_id] += 1

    metadata = {
        "project": "Puls-Events RAG POC",
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(INPUT_CSV.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "embedding_provider": "mistral",
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "source_event_count": int(len(df)),
        "document_count": int(len(documents)),
        "chunk_count": int(len(chunks)),
        "min_chunks_per_event": int(min(chunk_counts_by_event.values())),
        "max_chunks_per_event": int(max(chunk_counts_by_event.values())),
        "avg_chunks_per_event": round(
            sum(chunk_counts_by_event.values()) / len(chunk_counts_by_event),
            3,
        ),
        "required_metadata_fields": [
            "event_id",
            "title",
            "city",
            "venue_name",
            "address",
            "latitude",
            "longitude",
            "first_timing_begin",
            "last_timing_begin",
            "chunk_index",
            "chunk_id",
        ],
        "generated_files": [
            "index.faiss",
            "index.pkl",
            "build_metadata.json",
        ],
    }

    with BUILD_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def main() -> None:
    """Run the full FAISS vectorstore build pipeline.

    Loads cleaned events, chunks them, generates Mistral embeddings, persists
    the FAISS index under vectorstore/faiss_index/, and writes build metadata.
    """
    print("Building Puls-Events FAISS vectorstore")
    print(f"Input dataset: {INPUT_CSV}")

    df = load_events(INPUT_CSV)
    print(f"Loaded events: {len(df)}")

    documents = build_documents(df)
    print(f"Created documents: {len(documents)}")

    chunks = split_documents(documents)
    print(f"Created chunks: {len(chunks)}")

    reset_output_dir(OUTPUT_DIR)
    print(f"Output directory reset: {OUTPUT_DIR}")

    vectorstore = build_vectorstore(chunks)
    vectorstore.save_local(str(OUTPUT_DIR))
    print(f"FAISS vectorstore saved to: {OUTPUT_DIR}")

    write_build_metadata(
        df=df,
        documents=documents,
        chunks=chunks,
        output_dir=OUTPUT_DIR,
    )
    print(f"Build metadata saved to: {BUILD_METADATA_PATH}")

    print("Build complete")


if __name__ == "__main__":
    main()
