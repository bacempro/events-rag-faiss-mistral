from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


RAW_INPUT_PATH = Path("data/raw/openagenda_events_raw.json")
REPORT_OUTPUT_PATH = Path("reports/openagenda_raw_schema_summary.json")


IMPORTANT_FIELDS = [
    "uid",
    "title",
    "description",
    "longDescription",
    "dateRange",
    "timings",
    "firstTiming",
    "lastTiming",
    "nextTiming",
    "location",
    "conditions",
    "registration",
    "keywords",
]


def load_raw_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Raw OpenAgenda file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    events = payload.get("events")

    if not isinstance(events, list):
        raise ValueError("Expected raw payload to contain an 'events' list.")

    return events


def is_missing(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str) and not value.strip():
        return True

    if isinstance(value, list) and len(value) == 0:
        return True

    if isinstance(value, dict) and len(value) == 0:
        return True

    return False


def get_nested_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return sorted(value.keys())

    return []


def get_language_keys(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []

    return sorted(
        key for key, nested_value in value.items()
        if isinstance(key, str) and len(key) <= 5 and nested_value
    )


def profile_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    top_level_key_counts: Counter[str] = Counter()
    location_key_counts: Counter[str] = Counter()
    title_language_counts: Counter[str] = Counter()
    description_language_counts: Counter[str] = Counter()
    long_description_language_counts: Counter[str] = Counter()

    missing_counts: Counter[str] = Counter()

    timing_counts = {
        "events_with_timings": 0,
        "events_without_timings": 0,
        "total_timing_slots": 0,
    }

    for event in events:
        top_level_key_counts.update(event.keys())

        for field in IMPORTANT_FIELDS:
            if is_missing(event.get(field)):
                missing_counts[field] += 1

        location = event.get("location")
        if isinstance(location, dict):
            location_key_counts.update(location.keys())

        title_language_counts.update(get_language_keys(event.get("title")))
        description_language_counts.update(get_language_keys(event.get("description")))
        long_description_language_counts.update(
            get_language_keys(event.get("longDescription"))
        )

        timings = event.get("timings")
        if isinstance(timings, list) and timings:
            timing_counts["events_with_timings"] += 1
            timing_counts["total_timing_slots"] += len(timings)
        else:
            timing_counts["events_without_timings"] += 1

    examples = []
    for event in events[:5]:
        location = event.get("location") or {}
        first_timing = event.get("firstTiming") or {}

        examples.append(
            {
                "uid": event.get("uid"),
                "title_fr": (event.get("title") or {}).get("fr"),
                "description_fr": (event.get("description") or {}).get("fr"),
                "date_range_fr": (event.get("dateRange") or {}).get("fr"),
                "first_timing_begin": first_timing.get("begin"),
                "first_timing_end": first_timing.get("end"),
                "location_name": location.get("name"),
                "location_city": location.get("city"),
                "location_postal_code": location.get("postalCode"),
                "location_address": location.get("address"),
                "location_geo": location.get("geo"),
            }
        )

    return {
        "total_events": len(events),
        "top_level_key_counts": dict(top_level_key_counts.most_common()),
        "location_key_counts": dict(location_key_counts.most_common()),
        "language_counts": {
            "title": dict(title_language_counts.most_common()),
            "description": dict(description_language_counts.most_common()),
            "longDescription": dict(long_description_language_counts.most_common()),
        },
        "missing_important_field_counts": dict(missing_counts.most_common()),
        "timing_counts": timing_counts,
        "examples": examples,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("OpenAgenda raw schema inspection")
    print("=" * 40)
    print(f"Total events: {report['total_events']}")
    print()

    print("Missing important fields:")
    for field, count in report["missing_important_field_counts"].items():
        print(f"- {field}: {count}")

    print()
    print("Title languages:")
    for language, count in report["language_counts"]["title"].items():
        print(f"- {language}: {count}")

    print()
    print("Description languages:")
    for language, count in report["language_counts"]["description"].items():
        print(f"- {language}: {count}")

    print()
    print("Location keys:")
    for key, count in report["location_key_counts"].items():
        print(f"- {key}: {count}")

    print()
    print("Timing counts:")
    for key, value in report["timing_counts"].items():
        print(f"- {key}: {value}")


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def main() -> int:
    events = load_raw_events(RAW_INPUT_PATH)
    report = profile_events(events)

    print_summary(report)
    write_report(report, REPORT_OUTPUT_PATH)

    print()
    print(f"Schema summary written to: {REPORT_OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())