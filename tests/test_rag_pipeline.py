"""
Tests for the Puls-Events RAG pipeline.

These tests validate the structural behavior of the LangChain + Mistral + FAISS
RAG system before building the annotated QA dataset and running RAGAS.

They intentionally avoid brittle assertions:
- no exact event titles
- no exact FAISS scores
- no exact LLM wording
- no fixed number of indexed chunks

The tests require:
- a valid local .env file with MISTRAL_API_KEY
- an existing local FAISS vectorstore under vectorstore/faiss_index/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

import sys

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from pulsevents_rag.rag_chain import (  # noqa: E402
    answer_question,
    format_sources,
    load_config,
    load_vectorstore,
    retrieve_relevant_documents,
)


SMOKE_TEST_QUERY = "Je cherche une exposition à Paris"
SMOKE_TEST_TOP_K = 3


def _is_non_empty_string(value: Any) -> bool:
    """
    Return True when value is a non-empty displayable string.

    Args:
        value: value to validate.

    Returns:
        bool: True if value is a non-empty string after stripping.
    """
    return isinstance(value, str) and bool(value.strip())


def test_load_config_has_valid_defaults() -> None:
    """
    Validate that the RAG config resolves expected project paths and defaults.

    This test does not call the Mistral API.
    """
    config = load_config()

    assert config.project_root == PROJECT_ROOT
    assert config.vectorstore_path.exists()
    assert config.vectorstore_path.is_dir()
    assert _is_non_empty_string(config.embedding_model)
    assert _is_non_empty_string(config.chat_model)
    assert isinstance(config.default_top_k, int)
    assert config.default_top_k >= 1


def test_vectorstore_loads_with_vectors() -> None:
    """
    Validate that the local FAISS vectorstore can be loaded and contains vectors.

    This confirms that Step 4 can reuse the Step 3 vectorstore.
    """
    vectorstore = load_vectorstore()

    assert hasattr(vectorstore, "index")
    assert vectorstore.index.ntotal > 0


def test_retrieve_relevant_documents_returns_ranked_documents() -> None:
    """
    Validate that retrieval returns non-empty documents and numeric scores.

    The exact documents and scores are not asserted because OpenAgenda data and
    vector search ranking can change after valid rebuilds.
    """
    results = retrieve_relevant_documents(
        SMOKE_TEST_QUERY,
        top_k=SMOKE_TEST_TOP_K,
    )

    assert isinstance(results, list)
    assert 1 <= len(results) <= SMOKE_TEST_TOP_K

    for document, score in results:
        assert isinstance(document, Document)
        assert _is_non_empty_string(document.page_content)
        assert isinstance(document.metadata, dict)

        # FAISS scores should normally be floats. The RAG helper allows None
        # defensively, so the test accepts either but checks type when present.
        if score is not None:
            assert isinstance(score, float)


def test_format_sources_returns_required_traceability_fields() -> None:
    """
    Validate source formatting used by CLI display and future RAGAS traces.

    These fields make evaluation auditable without requiring full event content
    to be duplicated in metadata.
    """
    results = retrieve_relevant_documents(
        SMOKE_TEST_QUERY,
        top_k=SMOKE_TEST_TOP_K,
    )
    sources = format_sources(results)

    assert isinstance(sources, list)
    assert len(sources) == len(results)

    required_source_keys = {
        "rank",
        "score",
        "event_id",
        "title",
        "venue_name",
        "address",
        "city",
        "first_timing_begin",
        "last_timing_begin",
        "date_range",
        "conditions",
        "registration_url",
        "latitude",
        "longitude",
        "origin_agenda",
        "chunk_id",
        "chunk_index",
        "chunk_text_length",
        "chunk_preview",
    }

    for index, source in enumerate(sources, start=1):
        assert required_source_keys.issubset(source.keys())
        assert source["rank"] == index
        assert _is_non_empty_string(source["event_id"])
        assert _is_non_empty_string(source["title"])
        assert _is_non_empty_string(source["city"])
        assert source["city"].lower() == "paris"
        assert _is_non_empty_string(source["chunk_id"])
        assert _is_non_empty_string(source["chunk_preview"])


def test_answer_question_returns_expected_structure() -> None:
    """
    Validate the public RAG entry point.

    This test calls the chat model, but it does not assert exact wording because
    model output can vary slightly while still being valid.
    """
    result = answer_question(
        SMOKE_TEST_QUERY,
        top_k=SMOKE_TEST_TOP_K,
    )

    expected_keys = {
        "question",
        "answer",
        "contexts",
        "sources",
        "model",
        "embedding_model",
        "top_k",
    }

    assert isinstance(result, dict)
    assert expected_keys.issubset(result.keys())

    assert result["question"] == SMOKE_TEST_QUERY
    assert _is_non_empty_string(result["answer"])

    assert isinstance(result["contexts"], list)
    assert 1 <= len(result["contexts"]) <= SMOKE_TEST_TOP_K
    assert all(_is_non_empty_string(context) for context in result["contexts"])

    assert isinstance(result["sources"], list)
    assert len(result["sources"]) == len(result["contexts"])

    assert _is_non_empty_string(result["model"])
    assert _is_non_empty_string(result["embedding_model"])
    assert result["top_k"] == SMOKE_TEST_TOP_K


@pytest.mark.parametrize(
    "bad_question",
    [
        "",
        "   ",
    ],
)
def test_answer_question_rejects_empty_question(bad_question: str) -> None:
    """
    Validate that empty questions fail explicitly instead of reaching the LLM.
    """
    with pytest.raises(ValueError, match="question must not be empty"):
        answer_question(bad_question, top_k=SMOKE_TEST_TOP_K)


@pytest.mark.parametrize(
    "bad_top_k",
    [
        0,
        -1,
    ],
)
def test_retrieve_relevant_documents_rejects_invalid_top_k(bad_top_k: int) -> None:
    """
    Validate that invalid retrieval limits fail explicitly.
    """
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        retrieve_relevant_documents(SMOKE_TEST_QUERY, top_k=bad_top_k)


def test_retrieve_relevant_documents_rejects_non_integer_top_k() -> None:
    """
    Validate that top_k must be an integer when provided.
    """
    with pytest.raises(TypeError, match="top_k must be an integer"):
        retrieve_relevant_documents(SMOKE_TEST_QUERY, top_k="3")  # type: ignore[arg-type]