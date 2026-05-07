"""Normalize raw OpenAgenda events into a clean tabular dataset.

Input:
    data/raw/openagenda_events_raw.json

Output:
    data/processed/events_clean.csv
    data/processed/events_rejected.json

Usage:
    conda activate local-ai-rag
    python scripts/preprocess_events.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


RAW_INPUT_PATH = Path("data/raw/openagenda_events_raw.json")
CLEAN_OUTPUT_PATH = Path("data/processed/events_clean.csv")
REJECTED_OUTPUT_PATH = Path("data/processed/events_rejected.json")


REQUIRED_OUTPUT_FIELDS = [
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
]


CSV_COLUMNS = [
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
]


class PreprocessingError(RuntimeError):
    """Raised when preprocessing cannot continue."""


def load_raw_events(path: Path) -> list[dict[str, Any]]:
    """Load the raw OpenAgenda events list from a JSON extraction file."""
    if not path.exists():
        raise PreprocessingError(f"Raw OpenAgenda file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    events = payload.get("events")

    if not isinstance(events, list):
        raise PreprocessingError("Expected raw payload to contain an 'events' list.")

    return events


def normalize_whitespace(value: Any) -> str:
    """Strip HTML tags, collapse whitespace, and return a clean string."""
    if value is None:
        return ""

    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_localized_text(value: Any, preferred_language: str = "fr") -> str:
    """
    Extract text from OpenAgenda multilingual fields.

    Example input:
    {
        "fr": "...",
        "en": "..."
    }
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return normalize_whitespace(value)

    if isinstance(value, dict):
        preferred = value.get(preferred_language)
        if preferred:
            return normalize_whitespace(preferred)

        for fallback_value in value.values():
            if fallback_value:
                return normalize_whitespace(fallback_value)

    return ""


def get_nested_value(data: dict[str, Any], *keys: str) -> Any:
    """Traverse a nested dict by an arbitrary key path, returning None on any miss."""
    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return None

        current = current.get(key)

    return current


def parse_float(value: Any) -> float | None:
    """Parse a value as float, returning None if conversion is not possible."""
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_timing_value(event: dict[str, Any], timing_key: str, value_key: str) -> str:
    """Extract and normalize a single value from a named timing dict on an event."""
    timing = event.get(timing_key)

    if not isinstance(timing, dict):
        return ""

    return normalize_whitespace(timing.get(value_key))


def extract_registration(event: dict[str, Any]) -> tuple[str, str]:
    """
    Extract registration email and URL if available.

    OpenAgenda registration can vary. This function supports common shapes:
    - list of dicts
    - dict with values
    - plain string
    """
    registration = event.get("registration")

    email = ""
    url = ""

    def inspect_value(value: Any) -> None:
        """Recursively scan a registration value and populate the enclosing email/url variables."""
        nonlocal email, url

        if value is None:
            return

        if isinstance(value, dict):
            possible_type = normalize_whitespace(
                value.get("type") or value.get("kind") or value.get("name")
            ).lower()

            possible_value = normalize_whitespace(
                value.get("value")
                or value.get("url")
                or value.get("link")
                or value.get("email")
                or value.get("href")
            )

            if possible_value:
                if "@" in possible_value and not email:
                    email = possible_value
                elif possible_value.startswith(("http://", "https://")) and not url:
                    url = possible_value
                elif possible_type in {"email", "mail"} and not email:
                    email = possible_value
                elif possible_type in {"url", "link", "website", "reservation"} and not url:
                    url = possible_value

            for nested_value in value.values():
                inspect_value(nested_value)

        elif isinstance(value, list):
            for item in value:
                inspect_value(item)

        elif isinstance(value, str):
            cleaned = normalize_whitespace(value)

            if "@" in cleaned and not email:
                email = cleaned
            elif cleaned.startswith(("http://", "https://")) and not url:
                url = cleaned

    inspect_value(registration)

    return email, url


def build_text_for_embedding(row: dict[str, Any]) -> str:
    """Build the concatenated French text used as input for vectorization.

    Combines title, description, long description, dates, conditions, and location
    into a single normalized string ready for Mistral embeddings.
    """
    parts = []

    field_templates = [
        ("title", "Titre: {value}"),
        ("description", "Description: {value}"),
        ("long_description", "Description détaillée: {value}"),
        ("date_range", "Dates: {value}"),
        ("conditions", "Conditions: {value}"),
    ]

    for field_name, template in field_templates:
        value = normalize_whitespace(row.get(field_name))
        if value:
            parts.append(template.format(value=value))

    venue_name = normalize_whitespace(row.get("venue_name"))
    address = normalize_whitespace(row.get("address"))
    postal_code = normalize_whitespace(row.get("postal_code"))
    city = normalize_whitespace(row.get("city"))

    location_fragments = [
        fragment for fragment in [venue_name, address, postal_code, city] if fragment
    ]

    if location_fragments:
        parts.append(f"Lieu: {', '.join(location_fragments)}")

    return normalize_whitespace("\n".join(parts))


def transform_event(event: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """Map one raw OpenAgenda event dict to a clean row dict.

    Returns a tuple of (clean_row, rejection_reasons). If rejection_reasons is
    non-empty, clean_row is None and the event should be written to the
    rejected-events log.
    """
    rejection_reasons: list[str] = []

    location = event.get("location")
    if not isinstance(location, dict):
        location = {}
        rejection_reasons.append("missing_location")

    origin_agenda = event.get("originAgenda")
    if not isinstance(origin_agenda, dict):
        origin_agenda = {}

    latitude = parse_float(location.get("latitude"))
    longitude = parse_float(location.get("longitude"))

    registration_email, registration_url = extract_registration(event)

    row: dict[str, Any] = {
        "event_id": event.get("uid"),
        "slug": normalize_whitespace(event.get("slug")),
        "title": get_localized_text(event.get("title")),
        "description": get_localized_text(event.get("description")),
        "long_description": get_localized_text(event.get("longDescription")),
        "date_range": get_localized_text(event.get("dateRange")),
        "first_timing_begin": extract_timing_value(event, "firstTiming", "begin"),
        "first_timing_end": extract_timing_value(event, "firstTiming", "end"),
        "last_timing_begin": extract_timing_value(event, "lastTiming", "begin"),
        "last_timing_end": extract_timing_value(event, "lastTiming", "end"),
        "next_timing_begin": extract_timing_value(event, "nextTiming", "begin"),
        "next_timing_end": extract_timing_value(event, "nextTiming", "end"),
        "timings_count": len(event.get("timings") or []),
        "city": normalize_whitespace(location.get("city")),
        "postal_code": normalize_whitespace(location.get("postalCode")),
        "district": normalize_whitespace(location.get("district")),
        "department": normalize_whitespace(location.get("department")),
        "region": normalize_whitespace(location.get("region")),
        "country_code": normalize_whitespace(location.get("countryCode")),
        "venue_name": get_localized_text(location.get("name")),
        "address": normalize_whitespace(location.get("address")),
        "latitude": latitude,
        "longitude": longitude,
        "conditions": get_localized_text(event.get("conditions")),
        "registration_email": registration_email,
        "registration_url": registration_url,
        "origin_agenda_uid": origin_agenda.get("uid"),
        "origin_agenda_title": get_localized_text(origin_agenda.get("title")),
        "source": "OpenAgenda",
    }

    row["text_for_embedding"] = build_text_for_embedding(row)

    for field_name in REQUIRED_OUTPUT_FIELDS:
        if row.get(field_name) in (None, ""):
            rejection_reasons.append(f"missing_required_field:{field_name}")

    if row["timings_count"] <= 0:
        rejection_reasons.append("missing_timings")

    if not is_valid_iso_datetime(row["first_timing_begin"]):
        rejection_reasons.append("invalid_first_timing_begin")

    if latitude is None:
        rejection_reasons.append("invalid_latitude")

    if longitude is None:
        rejection_reasons.append("invalid_longitude")

    if rejection_reasons:
        return None, rejection_reasons

    return row, []


def is_valid_iso_datetime(value: Any) -> bool:
    """Return True if value is a non-empty string parseable as an ISO 8601 datetime."""
    if not isinstance(value, str) or not value.strip():
        return False

    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def write_clean_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write clean event rows to a CSV file with the defined column schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def write_rejected_events(
    rejected_events: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write the rejected-events audit list to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(rejected_events, file, ensure_ascii=False, indent=2)


def preprocess_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process all raw events and split them into clean rows and rejected records."""
    clean_rows: list[dict[str, Any]] = []
    rejected_events: list[dict[str, Any]] = []

    for event in events:
        row, rejection_reasons = transform_event(event)

        if row is None:
            rejected_events.append(
                {
                    "uid": event.get("uid"),
                    "title_fr": get_localized_text(event.get("title")),
                    "rejection_reasons": rejection_reasons,
                }
            )
            continue

        clean_rows.append(row)

    return clean_rows, rejected_events


def print_summary(
    *,
    raw_count: int,
    clean_count: int,
    rejected_count: int,
    rejected_events: list[dict[str, Any]],
) -> None:
    """Print a preprocessing result summary and rejection-reason breakdown to stdout."""
    print("OpenAgenda preprocessing completed")
    print("=" * 40)
    print(f"Raw events: {raw_count}")
    print(f"Clean events: {clean_count}")
    print(f"Rejected events: {rejected_count}")

    if rejected_events:
        reason_counts: dict[str, int] = {}

        for rejected_event in rejected_events:
            for reason in rejected_event["rejection_reasons"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        print()
        print("Rejection reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda item: item[0]):
            print(f"- {reason}: {count}")

    print()
    print(f"Clean dataset written to: {CLEAN_OUTPUT_PATH}")
    print(f"Rejected rows written to: {REJECTED_OUTPUT_PATH}")


def main() -> int:
    """Load raw events, preprocess them, write outputs, and print a summary."""
    try:
        events = load_raw_events(RAW_INPUT_PATH)
        clean_rows, rejected_events = preprocess_events(events)

        if not clean_rows:
            raise PreprocessingError(
                "No clean events produced. Check raw input and preprocessing rules."
            )

        write_clean_csv(clean_rows, CLEAN_OUTPUT_PATH)
        write_rejected_events(rejected_events, REJECTED_OUTPUT_PATH)

        print_summary(
            raw_count=len(events),
            clean_count=len(clean_rows),
            rejected_count=len(rejected_events),
            rejected_events=rejected_events,
        )

        return 0

    except PreprocessingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON input: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())