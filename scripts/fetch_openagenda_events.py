"""Fetch raw Paris cultural events from the OpenAgenda API.

Applies the main filters at request time:
    - city: configured via OPENAGENDA_CITY (default: Paris)
    - timings[gte]: today - 365 days (future events included)

Output:
    data/raw/openagenda_events_raw.json

Usage:
    conda activate local-ai-rag
    python scripts/fetch_openagenda_events.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # Keeps the script usable even if python-dotenv is missing.
    load_dotenv = None


API_BASE_URL = "https://api.openagenda.com/v2"
DEFAULT_OUTPUT_PATH = Path("data/raw/openagenda_events_raw.json")
DEFAULT_PAGE_SIZE = 100


class OpenAgendaFetchError(RuntimeError):
    """Raised when OpenAgenda data extraction fails."""


def load_environment() -> None:
    """Load environment variables from .env when python-dotenv is available."""
    if load_dotenv is not None:
        load_dotenv()


def require_env(name: str) -> str:
    """Return the value of a required environment variable, raising if absent or empty."""
    value = os.getenv(name)
    if not value:
        raise OpenAgendaFetchError(
            f"Missing required environment variable: {name}. "
            f"Add it to your local .env file."
        )
    return value


def get_headers(api_key: str) -> dict[str, str]:
    """Build HTTP request headers for the OpenAgenda API."""
    return {
        "key": api_key,
        "Accept": "application/json",
        "User-Agent": "pulsevents-rag-poc/0.1",
    }


def request_json(
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, Any],
    timeout: int = 30,
) -> dict[str, Any]:
    """Execute a GET request and return the parsed JSON body.

    Raises OpenAgendaFetchError on HTTP errors or non-JSON/unexpected responses.
    """
    response = requests.get(url, headers=headers, params=params, timeout=timeout)

    if response.status_code >= 400:
        raise OpenAgendaFetchError(
            "OpenAgenda API request failed.\n"
            f"URL: {response.url}\n"
            f"Status: {response.status_code}\n"
            f"Body: {response.text[:1000]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise OpenAgendaFetchError(
            f"OpenAgenda API returned non-JSON response for URL: {response.url}"
        ) from exc

    if not isinstance(payload, dict):
        raise OpenAgendaFetchError(
            f"Unexpected OpenAgenda API response type: {type(payload).__name__}"
        )

    return payload


def resolve_agenda_uid(
    *,
    api_key: str,
    agenda_slug: str,
) -> int:
    """
    Resolve an OpenAgenda agenda UID from its slug.

    This avoids hard-coding a UID and keeps the project easier to reproduce.
    """
    url = f"{API_BASE_URL}/agendas"
    params = {
        "slug": agenda_slug,
        "size": 1,
        "includeFields[]": ["uid", "slug", "title"],
    }

    payload = request_json(
        url,
        headers=get_headers(api_key),
        params=params,
    )

    agendas = extract_items(payload)

    if not agendas:
        raise OpenAgendaFetchError(
            f"No OpenAgenda agenda found for slug: {agenda_slug!r}"
        )

    uid = agendas[0].get("uid")

    if uid is None:
        raise OpenAgendaFetchError(
            f"Agenda found for slug {agenda_slug!r}, but response has no uid."
        )

    return int(uid)


def extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract item list from common OpenAgenda response shapes.

    OpenAgenda responses may expose records under 'events', 'agendas',
    or another top-level list depending on the endpoint.
    """
    candidate_keys = ("events", "agendas", "items", "results", "data")

    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value

    for value in payload.values():
        if isinstance(value, list):
            return value

    return []


def extract_after(payload: dict[str, Any]) -> list[Any] | None:
    """
    Extract cursor for the next page if OpenAgenda returns one.

    The API v2 navigation can use an 'after' cursor for efficient pagination.
    """
    after = payload.get("after")

    if after is None:
        return None

    if isinstance(after, list):
        return after

    return [after]


def fetch_events(
    *,
    api_key: str,
    agenda_uid: int,
    city: str,
    start_date: date,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int = 20,
) -> dict[str, Any]:
    """
    Fetch raw OpenAgenda event data.

    Filtering is done in the API call:
    - city=Paris
    - timings[gte]=today - 365 days

    The returned JSON preserves raw event payloads plus extraction metadata.
    """
    url = f"{API_BASE_URL}/agendas/{agenda_uid}/events"

    all_events: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    after: list[Any] | None = None

    for page_number in range(1, max_pages + 1):
        params: dict[str, Any] = {
            "city": city,
            "timings[gte]": start_date.isoformat(),
            "timings[tz]": "Europe/Paris",
            "detailed": 1,
            "size": page_size,
        }

        if after:
            params["after[]"] = after

        payload = request_json(
            url,
            headers=get_headers(api_key),
            params=params,
        )

        events = extract_items(payload)
        all_events.extend(events)

        pages.append(
            {
                "page_number": page_number,
                "event_count": len(events),
                "after": extract_after(payload),
            }
        )

        after = extract_after(payload)

        if not events or not after:
            break

    return {
        "metadata": {
            "source": "OpenAgenda API v2",
            "agenda_uid": agenda_uid,
            "city": city,
            "timings_gte": start_date.isoformat(),
            "timezone": "Europe/Paris",
            "page_size": page_size,
            "max_pages": max_pages,
            "total_events": len(all_events),
            "api_filtering": {
                "city": city,
                "timings[gte]": start_date.isoformat(),
                "timings[tz]": "Europe/Paris",
            },
        },
        "pages": pages,
        "events": all_events,
    }


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Serialize payload to a JSON file, creating parent directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main() -> int:
    """Fetch Paris cultural events from OpenAgenda and write the raw JSON payload."""
    load_environment()

    try:
        api_key = require_env("OPENAGENDA_API_KEY")
        agenda_slug = os.getenv("OPENAGENDA_AGENDA_SLUG", "culture")
        city = os.getenv("OPENAGENDA_CITY", "Paris")

        start_date = date.today() - timedelta(days=365)

        agenda_uid = resolve_agenda_uid(
            api_key=api_key,
            agenda_slug=agenda_slug,
        )

        payload = fetch_events(
            api_key=api_key,
            agenda_uid=agenda_uid,
            city=city,
            start_date=start_date,
        )

        write_json(payload, DEFAULT_OUTPUT_PATH)

        print("OpenAgenda raw events fetched successfully.")
        print(f"Agenda slug: {agenda_slug}")
        print(f"Agenda UID: {agenda_uid}")
        print(f"City: {city}")
        print(f"timings[gte]: {start_date.isoformat()}")
        print(f"Events fetched: {payload['metadata']['total_events']}")
        print(f"Output: {DEFAULT_OUTPUT_PATH}")

        return 0

    except OpenAgendaFetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    except requests.RequestException as exc:
        print(f"HTTP ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
