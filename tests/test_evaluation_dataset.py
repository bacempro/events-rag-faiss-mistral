"""
Tests for the Puls-Events annotated QA evaluation dataset.

These tests validate data/evaluation/annotated_qa_dataset.jsonl before it is
used by the RAGAS evaluation script.

The dataset is expected to contain human-annotated question/reference-answer
pairs grounded in real events from data/processed/events_clean.csv.

The tests intentionally validate structure and traceability, not semantic
quality. Semantic quality will be evaluated later by RAGAS and manual review.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_DATASET_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "annotated_qa_dataset.jsonl"
)
PROCESSED_EVENTS_PATH = PROJECT_ROOT / "data" / "processed" / "events_clean.csv"

EXPECTED_ROW_COUNT = 12

REQUIRED_FIELDS = {
    "id",
    "question",
    "reference_answer",
    "reference_event_ids",
    "reference_event_titles",
    "question_type",
    "expected_topics",
    "notes",
}

ALLOWED_QUESTION_TYPES = {
    "recommendation",
    "negative",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Load a JSONL file as a list of dictionaries.

    Args:
        path: JSONL file path.

    Returns:
        list[dict[str, Any]]: parsed rows.
    """
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise AssertionError(
                    f"Line {line_number} must contain a JSON object, "
                    f"got {type(row).__name__}"
                )

            rows.append(row)

    return rows


def _is_non_empty_string(value: Any) -> bool:
    """
    Return True when value is a non-empty string after stripping.

    Args:
        value: value to inspect.

    Returns:
        bool: True if value is a non-empty string.
    """
    return isinstance(value, str) and bool(value.strip())


def _load_processed_event_ids() -> set[str]:
    """
    Load event IDs from the processed event dataset.

    Returns:
        set[str]: event IDs as strings.
    """
    assert PROCESSED_EVENTS_PATH.exists(), (
        f"Processed events file is missing: {PROCESSED_EVENTS_PATH}"
    )

    events = pd.read_csv(PROCESSED_EVENTS_PATH)

    assert "event_id" in events.columns, (
        "Processed events dataset must contain an 'event_id' column."
    )

    return set(events["event_id"].astype(str))


def test_evaluation_dataset_file_exists() -> None:
    """
    Validate that the annotated QA dataset file exists.
    """
    assert EVALUATION_DATASET_PATH.exists(), (
        f"Evaluation dataset is missing: {EVALUATION_DATASET_PATH}"
    )
    assert EVALUATION_DATASET_PATH.is_file()
    assert EVALUATION_DATASET_PATH.stat().st_size > 0


def test_evaluation_dataset_is_valid_jsonl() -> None:
    """
    Validate that every non-empty line is a JSON object.
    """
    rows = _load_jsonl(EVALUATION_DATASET_PATH)

    assert rows, "Evaluation dataset must not be empty."
    assert all(isinstance(row, dict) for row in rows)


def test_evaluation_dataset_has_expected_size() -> None:
    """
    Validate the expected POC evaluation dataset size.

    The current evaluation set intentionally contains 12 items:
    - 10 positive recommendation questions
    - 2 negative insufficient-context questions
    """
    rows = _load_jsonl(EVALUATION_DATASET_PATH)

    assert len(rows) == EXPECTED_ROW_COUNT


def test_evaluation_dataset_ids_are_unique_and_ordered() -> None:
    """
    Validate that QA IDs are unique and follow the qa_XXX convention.
    """
    rows = _load_jsonl(EVALUATION_DATASET_PATH)

    ids = [row.get("id") for row in rows]

    assert len(ids) == len(set(ids)), "QA IDs must be unique."

    expected_ids = [f"qa_{index:03d}" for index in range(1, EXPECTED_ROW_COUNT + 1)]

    assert ids == expected_ids


def test_evaluation_dataset_required_fields_are_present() -> None:
    """
    Validate that every row contains the required schema fields.
    """
    rows = _load_jsonl(EVALUATION_DATASET_PATH)

    for row in rows:
        missing_fields = REQUIRED_FIELDS - set(row.keys())
        assert not missing_fields, (
            f"Row {row.get('id', '<missing id>')} is missing fields: "
            f"{sorted(missing_fields)}"
        )


def test_evaluation_dataset_field_types_are_valid() -> None:
    """
    Validate basic field types and non-empty text fields.
    """
    rows = _load_jsonl(EVALUATION_DATASET_PATH)

    for row in rows:
        row_id = row.get("id", "<missing id>")

        assert _is_non_empty_string(row["id"]), f"{row_id}: id must be non-empty."
        assert _is_non_empty_string(row["question"]), (
            f"{row_id}: question must be non-empty."
        )
        assert _is_non_empty_string(row["reference_answer"]), (
            f"{row_id}: reference_answer must be non-empty."
        )
        assert _is_non_empty_string(row["question_type"]), (
            f"{row_id}: question_type must be non-empty."
        )
        assert _is_non_empty_string(row["notes"]), (
            f"{row_id}: notes must be non-empty."
        )

        assert isinstance(row["reference_event_ids"], list), (
            f"{row_id}: reference_event_ids must be a list."
        )
        assert isinstance(row["reference_event_titles"], list), (
            f"{row_id}: reference_event_titles must be a list."
        )
        assert isinstance(row["expected_topics"], list), (
            f"{row_id}: expected_topics must be a list."
        )

        assert all(_is_non_empty_string(value) for value in row["expected_topics"]), (
            f"{row_id}: expected_topics must contain only non-empty strings."
        )


def test_question_types_are_allowed() -> None:
    """
    Validate that question_type values are controlled.

    This prevents accidental typo categories from entering the evaluation file.
    """
    rows = _load_jsonl(EVALUATION_DATASET_PATH)

    for row in rows:
        assert row["question_type"] in ALLOWED_QUESTION_TYPES, (
            f"{row['id']}: unsupported question_type {row['question_type']!r}."
        )


def test_positive_and_negative_reference_event_rules() -> None:
    """
    Validate reference-event rules.

    Positive recommendation rows must point to at least one real event.
    Negative rows must not point to a reference event, because they are designed
    to check insufficient-context behavior.
    """
    rows = _load_jsonl(EVALUATION_DATASET_PATH)

    positive_rows = [row for row in rows if row["question_type"] == "recommendation"]
    negative_rows = [row for row in rows if row["question_type"] == "negative"]

    assert len(positive_rows) == 10
    assert len(negative_rows) == 2

    for row in positive_rows:
        assert row["reference_event_ids"], (
            f"{row['id']}: recommendation rows must have reference_event_ids."
        )
        assert row["reference_event_titles"], (
            f"{row['id']}: recommendation rows must have reference_event_titles."
        )

    for row in negative_rows:
        assert row["reference_event_ids"] == [], (
            f"{row['id']}: negative rows must not have reference_event_ids."
        )
        assert row["reference_event_titles"] == [], (
            f"{row['id']}: negative rows must not have reference_event_titles."
        )


def test_reference_event_ids_exist_in_processed_dataset() -> None:
    """
    Validate that all positive reference_event_ids exist in events_clean.csv.

    This guarantees the annotated dataset is grounded in the current processed
    OpenAgenda dataset.
    """
    rows = _load_jsonl(EVALUATION_DATASET_PATH)
    processed_event_ids = _load_processed_event_ids()

    for row in rows:
        for event_id in row["reference_event_ids"]:
            assert str(event_id) in processed_event_ids, (
                f"{row['id']}: reference event_id {event_id!r} does not exist in "
                f"{PROCESSED_EVENTS_PATH}."
            )


def test_reference_event_titles_are_consistent_with_event_ids() -> None:
    """
    Validate that positive rows provide one title per referenced event.

    This does not compare exact title text against events_clean.csv because small
    typographic differences can appear after normalization. The existence check
    is handled by event_id validation.
    """
    rows = _load_jsonl(EVALUATION_DATASET_PATH)

    for row in rows:
        assert len(row["reference_event_titles"]) == len(row["reference_event_ids"]), (
            f"{row['id']}: reference_event_titles and reference_event_ids must have "
            "the same length."
        )

        assert all(
            _is_non_empty_string(title) for title in row["reference_event_titles"]
        ), f"{row['id']}: reference_event_titles must contain non-empty strings."