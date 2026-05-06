"""
Tests for the Puls-Events FAISS vectorstore.

These tests intentionally avoid brittle assertions tied to one specific data extraction,
such as exact event count, exact chunk count, or a fixed lower-bound date.

They validate stable vectorstore invariants:
- required files exist
- build metadata is coherent
- FAISS index size matches the stored docstore
- indexed chunks have required metadata
- indexed event IDs come from the processed source dataset
- key metadata values remain consistent with the cleaned CSV
- dates are parseable and logically ordered
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "events_clean.csv"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "faiss_index"
BUILD_METADATA_PATH = VECTORSTORE_DIR / "build_metadata.json"

EXPECTED_CITY = os.getenv("OPENAGENDA_CITY", "Paris")

REQUIRED_VECTORSTORE_FILES = [
    VECTORSTORE_DIR / "index.faiss",
    VECTORSTORE_DIR / "index.pkl",
    BUILD_METADATA_PATH,
]

REQUIRED_METADATA_FIELDS = [
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
]

REQUIRED_SOURCE_COLUMNS = [
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


class DummyEmbeddings(Embeddings):
    """
    Minimal embedding class used only to load a persisted LangChain FAISS vectorstore.

    These tests do not perform semantic search, so no real Mistral API call is needed.
    The vectorstore loader requires an Embeddings object, but the stored vectors are
    already inside the FAISS index.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


@pytest.fixture(scope="session")
def source_df() -> pd.DataFrame:
    assert PROCESSED_CSV.exists(), f"Missing processed dataset: {PROCESSED_CSV}"

    df = pd.read_csv(PROCESSED_CSV)

    missing_columns = [col for col in REQUIRED_SOURCE_COLUMNS if col not in df.columns]
    assert not missing_columns, f"Missing source CSV columns: {missing_columns}"

    assert not df.empty, "Processed source dataset is empty."

    return df


@pytest.fixture(scope="session")
def build_metadata() -> dict[str, Any]:
    assert BUILD_METADATA_PATH.exists(), f"Missing build metadata: {BUILD_METADATA_PATH}"

    with BUILD_METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    assert isinstance(metadata, dict), "Build metadata must be a JSON object."

    return metadata


@pytest.fixture(scope="session")
def vectorstore() -> FAISS:
    for path in REQUIRED_VECTORSTORE_FILES:
        assert path.exists(), f"Missing vectorstore file: {path}"

    return FAISS.load_local(
        folder_path=str(VECTORSTORE_DIR),
        embeddings=DummyEmbeddings(),
        allow_dangerous_deserialization=True,
    )


@pytest.fixture(scope="session")
def indexed_documents(vectorstore: FAISS) -> list[Any]:
    docstore_dict = getattr(vectorstore.docstore, "_dict", None)

    assert isinstance(
        docstore_dict, dict
    ), "Expected LangChain FAISS docstore to expose an internal document dictionary."

    documents = list(docstore_dict.values())

    assert documents, "Vectorstore docstore is empty."

    return documents


def normalize_optional_string(value: Any) -> str:
    if pd.isna(value) or value is None:
        return ""

    return str(value).strip()


def test_vectorstore_files_exist() -> None:
    for path in REQUIRED_VECTORSTORE_FILES:
        assert path.exists(), f"Missing vectorstore file: {path}"
        assert path.stat().st_size > 0, f"Vectorstore file is empty: {path}"


def test_build_metadata_is_coherent(build_metadata: dict[str, Any]) -> None:
    required_keys = [
        "project",
        "build_timestamp_utc",
        "input_csv",
        "output_dir",
        "embedding_provider",
        "embedding_model",
        "chunk_size",
        "chunk_overlap",
        "source_event_count",
        "document_count",
        "chunk_count",
        "required_metadata_fields",
        "generated_files",
    ]

    missing_keys = [key for key in required_keys if key not in build_metadata]
    assert not missing_keys, f"Missing build metadata keys: {missing_keys}"

    assert build_metadata["project"] == "Puls-Events RAG POC"
    assert build_metadata["embedding_provider"] == "mistral"
    assert isinstance(build_metadata["embedding_model"], str)
    assert build_metadata["embedding_model"].strip()

    assert int(build_metadata["chunk_size"]) > 0
    assert int(build_metadata["chunk_overlap"]) >= 0
    assert int(build_metadata["chunk_overlap"]) < int(build_metadata["chunk_size"])

    assert int(build_metadata["source_event_count"]) > 0
    assert int(build_metadata["document_count"]) > 0
    assert int(build_metadata["chunk_count"]) > 0

    assert int(build_metadata["document_count"]) == int(
        build_metadata["source_event_count"]
    )
    assert int(build_metadata["chunk_count"]) >= int(build_metadata["document_count"])

    required_metadata_fields = set(build_metadata["required_metadata_fields"])
    for field in REQUIRED_METADATA_FIELDS:
        assert field in required_metadata_fields, (
            f"Required metadata field missing from build metadata: {field}"
        )

    generated_files = set(build_metadata["generated_files"])
    assert {"index.faiss", "index.pkl", "build_metadata.json"}.issubset(generated_files)


def test_build_metadata_matches_source_dataset(
    build_metadata: dict[str, Any],
    source_df: pd.DataFrame,
) -> None:
    assert int(build_metadata["source_event_count"]) == len(source_df)
    assert int(build_metadata["document_count"]) == len(source_df)

    input_csv = Path(build_metadata["input_csv"])
    assert input_csv.as_posix() == "data/processed/events_clean.csv"


def test_faiss_index_size_matches_docstore(
    vectorstore: FAISS,
    indexed_documents: list[Any],
    build_metadata: dict[str, Any],
) -> None:
    assert vectorstore.index.ntotal > 0

    assert vectorstore.index.ntotal == len(indexed_documents)
    assert vectorstore.index.ntotal == int(build_metadata["chunk_count"])


def test_indexed_documents_have_text(indexed_documents: list[Any]) -> None:
    for document in indexed_documents:
        assert isinstance(document.page_content, str)
        assert document.page_content.strip(), "Indexed document has empty page_content."


def test_indexed_documents_have_required_metadata(indexed_documents: list[Any]) -> None:
    for document in indexed_documents:
        metadata = document.metadata

        missing_fields = [
            field
            for field in REQUIRED_METADATA_FIELDS
            if field not in metadata or normalize_optional_string(metadata[field]) == ""
        ]

        assert not missing_fields, (
            f"Indexed document is missing required metadata fields: {missing_fields}. "
            f"Metadata: {metadata}"
        )


def test_chunk_ids_are_unique(indexed_documents: list[Any]) -> None:
    chunk_ids = [document.metadata["chunk_id"] for document in indexed_documents]

    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk_id values found."


def test_chunk_indices_are_valid(indexed_documents: list[Any]) -> None:
    for document in indexed_documents:
        chunk_index = document.metadata["chunk_index"]

        assert isinstance(chunk_index, int)
        assert chunk_index >= 0


def test_indexed_event_ids_exist_in_source_dataset(
    indexed_documents: list[Any],
    source_df: pd.DataFrame,
) -> None:
    source_event_ids = set(source_df["event_id"].astype(str))
    indexed_event_ids = {
        str(document.metadata["event_id"]) for document in indexed_documents
    }

    missing_from_source = indexed_event_ids - source_event_ids

    assert not missing_from_source, (
        "Some indexed event IDs do not exist in the processed CSV: "
        f"{sorted(missing_from_source)[:10]}"
    )


def test_every_source_event_has_at_least_one_indexed_chunk(
    indexed_documents: list[Any],
    source_df: pd.DataFrame,
) -> None:
    source_event_ids = set(source_df["event_id"].astype(str))
    indexed_event_ids = {
        str(document.metadata["event_id"]) for document in indexed_documents
    }

    missing_from_index = source_event_ids - indexed_event_ids

    assert not missing_from_index, (
        "Some source events were not indexed into FAISS: "
        f"{sorted(missing_from_index)[:10]}"
    )


def test_indexed_metadata_matches_source_dataset(
    indexed_documents: list[Any],
    source_df: pd.DataFrame,
) -> None:
    source_by_event_id = {
        str(row["event_id"]): row for _, row in source_df.iterrows()
    }

    fields_to_compare = [
        "title",
        "city",
        "venue_name",
        "address",
        "first_timing_begin",
        "last_timing_begin",
    ]

    for document in indexed_documents:
        event_id = str(document.metadata["event_id"])
        source_row = source_by_event_id[event_id]

        for field in fields_to_compare:
            indexed_value = normalize_optional_string(document.metadata.get(field))
            source_value = normalize_optional_string(source_row.get(field))

            assert indexed_value == source_value, (
                f"Metadata mismatch for event_id={event_id}, field={field}. "
                f"Indexed={indexed_value!r}, source={source_value!r}"
            )


def test_indexed_city_matches_configured_scope(indexed_documents: list[Any]) -> None:
    expected_city_normalized = EXPECTED_CITY.strip().lower()

    assert expected_city_normalized, "Expected city cannot be empty."

    for document in indexed_documents:
        indexed_city = str(document.metadata["city"]).strip().lower()
        assert indexed_city == expected_city_normalized


def test_indexed_coordinates_are_valid_numbers(indexed_documents: list[Any]) -> None:
    for document in indexed_documents:
        latitude = float(document.metadata["latitude"])
        longitude = float(document.metadata["longitude"])

        assert -90 <= latitude <= 90
        assert -180 <= longitude <= 180


def test_indexed_dates_are_parseable_and_ordered(indexed_documents: list[Any]) -> None:
    for document in indexed_documents:
        first_timing = pd.to_datetime(
            document.metadata["first_timing_begin"],
            errors="coerce",
            utc=True,
        )
        last_timing = pd.to_datetime(
            document.metadata["last_timing_begin"],
            errors="coerce",
            utc=True,
        )

        assert not pd.isna(first_timing), (
            f"Invalid first_timing_begin for metadata: {document.metadata}"
        )
        assert not pd.isna(last_timing), (
            f"Invalid last_timing_begin for metadata: {document.metadata}"
        )

        assert last_timing >= first_timing, (
            "last_timing_begin must be greater than or equal to first_timing_begin. "
            f"Metadata: {document.metadata}"
        )