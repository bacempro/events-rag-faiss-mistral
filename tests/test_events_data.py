from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


PROCESSED_DATASET_PATH = Path("data/processed/events_clean.csv")

REQUIRED_COLUMNS = {
    "event_id",
    "slug",
    "title",
    "description",
    "long_description",
    "date_range",
    "first_timing_begin",
    "first_timing_end",
    "last_timing_begin",
    "last_timing_end",
    "next_timing_begin",
    "next_timing_end",
    "timings_count",
    "city",
    "postal_code",
    "district",
    "department",
    "region",
    "country_code",
    "venue_name",
    "address",
    "latitude",
    "longitude",
    "conditions",
    "registration_email",
    "registration_url",
    "origin_agenda_uid",
    "origin_agenda_title",
    "source",
    "text_for_embedding",
}

REQUIRED_NON_EMPTY_FIELDS = {
    "event_id",
    "title",
    "description",
    "first_timing_begin",
    "city",
    "venue_name",
    "address",
    "latitude",
    "longitude",
    "text_for_embedding",
}


def load_rows() -> list[dict[str, Any]]:
    assert PROCESSED_DATASET_PATH.exists(), (
        f"Processed dataset not found: {PROCESSED_DATASET_PATH}. "
        "Run: python scripts/preprocess_events.py"
    )

    with PROCESSED_DATASET_PATH.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def parse_iso_datetime(value: Any) -> datetime:
    assert value and value.strip(), "Datetime value is empty"

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AssertionError(f"Invalid ISO datetime: {value}") from exc


def test_processed_dataset_exists() -> None:
    assert PROCESSED_DATASET_PATH.exists()


def test_processed_dataset_is_not_empty() -> None:
    rows = load_rows()

    assert len(rows) > 0


def test_required_columns_exist() -> None:
    rows = load_rows()

    assert rows, "Dataset is empty"

    actual_columns = set(rows[0].keys())
    missing_columns = REQUIRED_COLUMNS - actual_columns

    assert not missing_columns, f"Missing columns: {sorted(missing_columns)}"


def test_required_fields_are_not_empty() -> None:
    rows = load_rows()

    missing_by_field: dict[str, int] = {}

    for field in REQUIRED_NON_EMPTY_FIELDS:
        missing_count = sum(
            1 for row in rows
            if not str(row.get(field, "")).strip()
        )

        if missing_count:
            missing_by_field[field] = missing_count

    assert not missing_by_field, f"Missing required values: {missing_by_field}"


def test_all_events_are_in_paris() -> None:
    rows = load_rows()

    invalid_rows = [
        {
            "event_id": row.get("event_id"),
            "city": row.get("city"),
        }
        for row in rows
        if row.get("city", "").strip().lower() != "paris"
    ]

    assert not invalid_rows, f"Found non-Paris events: {invalid_rows[:10]}"


def test_all_events_have_at_least_one_recent_or_future_timing() -> None:
    """
    Validate the API-side extraction rule.

    The fetch script requests:
    timings[gte] = today - 365 days

    OpenAgenda events may have a firstTiming older than the API lower bound
    when the event is recurring or long-running. Therefore, the correct
    dataset-level check is that the event still has at least one occurrence
    in the accepted period.

    In the cleaned CSV, last_timing_begin is the best available proxy for this:
    if the last timing is older than the minimum date, the event cannot have
    any valid occurrence in the requested period.
    """
    rows = load_rows()
    min_allowed_date = date.today() - timedelta(days=365)

    invalid_rows = []

    for row in rows:
        last_timing_value = row.get("last_timing_begin") or row.get("first_timing_begin")
        last_timing_begin = parse_iso_datetime(last_timing_value)

        if last_timing_begin.date() < min_allowed_date:
            invalid_rows.append(
                {
                    "event_id": row.get("event_id"),
                    "title": row.get("title"),
                    "first_timing_begin": row.get("first_timing_begin"),
                    "last_timing_begin": row.get("last_timing_begin"),
                }
            )

    assert not invalid_rows, (
        f"Found events with no timing after {min_allowed_date.isoformat()}: "
        f"{invalid_rows[:10]}"
    )


def test_coordinates_are_valid_for_paris_area() -> None:
    """
    Validate coordinates are plausible for Paris.

    This is intentionally a broad bounding-box check, not a precise GIS test.
    """
    rows = load_rows()

    invalid_rows = []

    for row in rows:
        try:
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
        except ValueError:
            invalid_rows.append(
                {
                    "event_id": row.get("event_id"),
                    "latitude": row.get("latitude"),
                    "longitude": row.get("longitude"),
                    "reason": "non_numeric_coordinate",
                }
            )
            continue

        if not (48.80 <= latitude <= 48.92 and 2.20 <= longitude <= 2.48):
            invalid_rows.append(
                {
                    "event_id": row.get("event_id"),
                    "latitude": latitude,
                    "longitude": longitude,
                    "reason": "outside_paris_bounding_box",
                }
            )

    assert not invalid_rows, f"Invalid Paris coordinates: {invalid_rows[:10]}"


def test_text_for_embedding_is_usable() -> None:
    rows = load_rows()

    invalid_rows = [
        {
            "event_id": row.get("event_id"),
            "text_for_embedding_length": len(row.get("text_for_embedding", "")),
        }
        for row in rows
        if len(row.get("text_for_embedding", "").strip()) < 80
    ]

    assert not invalid_rows, (
        "Some events have text_for_embedding shorter than 80 characters: "
        f"{invalid_rows[:10]}"
    )


def test_timings_count_is_positive_integer() -> None:
    rows = load_rows()

    invalid_rows = []

    for row in rows:
        try:
            timings_count = int(row["timings_count"])
        except ValueError:
            invalid_rows.append(
                {
                    "event_id": row.get("event_id"),
                    "timings_count": row.get("timings_count"),
                    "reason": "not_an_integer",
                }
            )
            continue

        if timings_count <= 0:
            invalid_rows.append(
                {
                    "event_id": row.get("event_id"),
                    "timings_count": timings_count,
                    "reason": "not_positive",
                }
            )

    assert not invalid_rows, f"Invalid timings_count values: {invalid_rows[:10]}"