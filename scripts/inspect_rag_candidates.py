"""
Inspect candidate events for the Puls-Events annotated QA dataset.

This script runs retrieval-only queries against the existing FAISS vectorstore and
prints/export candidate event sources. It is intended to help select real events
for data/evaluation/annotated_qa_dataset.jsonl.

It does not call the chat model. It only uses the retriever.

Usage examples:
    python scripts/inspect_rag_candidates.py
    python scripts/inspect_rag_candidates.py --top-k 8
    python scripts/inspect_rag_candidates.py --query "Je cherche une exposition gratuite à Paris"
    python scripts/inspect_rag_candidates.py --output-csv data/evaluation/candidate_events_for_qa.csv
    python scripts/inspect_rag_candidates.py --output-jsonl data/evaluation/candidate_events_for_qa.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pulsevents_rag.rag_chain import (  # noqa: E402
    format_sources,
    retrieve_relevant_documents,
)


DEFAULT_TOP_K = 8

DEFAULT_QUERIES = [
    "Je cherche une exposition à Paris",
    "Je cherche une exposition gratuite à Paris",
    "Je cherche un concert ou un événement musical à Paris",
    "Je cherche une sortie culturelle pour des enfants à Paris",
    "Je cherche une activité familiale à Paris",
    "Je cherche une pièce de théâtre ou un spectacle à Paris",
    "Je cherche un événement culturel gratuit à Paris",
    "Je cherche un événement avec inscription ou réservation à Paris",
    "Je cherche une visite guidée culturelle à Paris",
    "Propose-moi trois sorties culturelles à Paris avec les lieux et les dates",
]

EXPORT_COLUMNS = [
    "query_id",
    "query",
    "rank",
    "score",
    "event_id",
    "title",
    "venue_name",
    "address",
    "city",
    "date_range",
    "first_timing_begin",
    "last_timing_begin",
    "conditions",
    "registration_url",
    "origin_agenda",
    "chunk_id",
    "chunk_index",
    "chunk_text_length",
    "chunk_preview",
]


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.

    Returns:
        argparse.ArgumentParser: configured parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Inspect retrieved event candidates for the Puls-Events annotated "
            "QA dataset."
        )
    )

    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help=(
            "Custom query to inspect. Can be used multiple times. "
            "If omitted, a default candidate query set is used."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of FAISS chunks to retrieve per query. Default: {DEFAULT_TOP_K}.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV export path, e.g. data/evaluation/candidate_events_for_qa.csv.",
    )

    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL export path, e.g. data/evaluation/candidate_events_for_qa.jsonl.",
    )

    parser.add_argument(
        "--preview-chars",
        type=int,
        default=450,
        help="Maximum characters to display/export for chunk previews. Default: 450.",
    )

    parser.add_argument(
        "--deduplicate-events",
        action="store_true",
        help=(
            "Keep only the best-ranked chunk per event_id for each query. "
            "Useful when multiple chunks from the same event dominate results."
        ),
    )

    return parser


def normalize_text(value: Any) -> str:
    """
    Convert a value to a clean one-line string.

    Args:
        value: raw value.

    Returns:
        str: normalized string.
    """
    if value is None:
        return ""

    text = " ".join(str(value).split()).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""

    return text


def truncate_text(value: Any, max_chars: int) -> str:
    """
    Normalize and truncate text for display/export.

    Args:
        value: raw value.
        max_chars: maximum number of characters.

    Returns:
        str: truncated text.
    """
    text = normalize_text(value)

    if max_chars <= 0:
        return ""

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3].rstrip() + "..."


def format_score(score: Any) -> str:
    """
    Format a score for terminal display.

    Args:
        score: raw score.

    Returns:
        str: formatted score.
    """
    if score is None:
        return "N/A"

    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return normalize_text(score)


def deduplicate_sources_by_event_id(
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Keep only the first retrieved source per event_id.

    Args:
        sources: formatted source dictionaries.

    Returns:
        list[dict[str, Any]]: deduplicated source list.
    """
    seen_event_ids: set[str] = set()
    deduplicated: list[dict[str, Any]] = []

    for source in sources:
        event_id = normalize_text(source.get("event_id"))

        # If event_id is missing, keep the row rather than hiding potential data issues.
        if not event_id:
            deduplicated.append(source)
            continue

        if event_id in seen_event_ids:
            continue

        seen_event_ids.add(event_id)
        deduplicated.append(source)

    return deduplicated


def source_to_export_row(
    query_id: str,
    query: str,
    source: dict[str, Any],
    preview_chars: int,
) -> dict[str, Any]:
    """
    Convert a formatted source into a flat export row.

    Args:
        query_id: stable query identifier.
        query: query text.
        source: source dictionary from format_sources.
        preview_chars: preview truncation length.

    Returns:
        dict[str, Any]: export row.
    """
    row = {
        "query_id": query_id,
        "query": query,
    }

    for column in EXPORT_COLUMNS:
        if column in {"query_id", "query"}:
            continue

        value = source.get(column)

        if column == "chunk_preview":
            row[column] = truncate_text(value, preview_chars)
        else:
            row[column] = normalize_text(value)

    return row


def print_query_results(
    query_id: str,
    query: str,
    sources: list[dict[str, Any]],
    preview_chars: int,
) -> None:
    """
    Print retrieved sources for one query.

    Args:
        query_id: query identifier.
        query: query text.
        sources: formatted sources.
        preview_chars: preview length.
    """
    print()
    print("=" * 100)
    print(f"{query_id} — {query}")
    print("=" * 100)

    if not sources:
        print("No retrieved sources.")
        return

    for source in sources:
        rank = normalize_text(source.get("rank")) or "?"
        score = format_score(source.get("score"))
        event_id = normalize_text(source.get("event_id")) or "N/A"
        title = normalize_text(source.get("title")) or "N/A"
        venue = normalize_text(source.get("venue_name")) or "N/A"
        address = normalize_text(source.get("address")) or "N/A"
        city = normalize_text(source.get("city")) or "N/A"
        date_range = normalize_text(source.get("date_range")) or "N/A"
        conditions = normalize_text(source.get("conditions"))
        registration_url = normalize_text(source.get("registration_url"))
        chunk_id = normalize_text(source.get("chunk_id")) or "N/A"
        preview = truncate_text(source.get("chunk_preview"), preview_chars)

        print("-" * 100)
        print(f"Rank        : {rank}")
        print(f"Score       : {score}")
        print(f"Event ID    : {event_id}")
        print(f"Title       : {title}")
        print(f"Venue       : {venue}")
        print(f"Address     : {address}")
        print(f"City        : {city}")
        print(f"Dates       : {date_range}")
        print(f"Chunk ID    : {chunk_id}")

        if conditions:
            print(f"Conditions  : {conditions}")

        if registration_url:
            print(f"Registration: {registration_url}")

        if preview:
            print()
            print("Preview:")
            print(preview)

    print("-" * 100)


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """
    Write candidate rows to a CSV file.

    Args:
        rows: export rows.
        output_path: output CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    """
    Write candidate rows to a JSONL file.

    Args:
        rows: export rows.
        output_path: output JSONL path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def inspect_queries(
    queries: list[str],
    top_k: int,
    preview_chars: int,
    deduplicate_events: bool,
) -> list[dict[str, Any]]:
    """
    Run retrieval for all queries and return export rows.

    Args:
        queries: queries to inspect.
        top_k: number of chunks per query.
        preview_chars: preview truncation length.
        deduplicate_events: whether to keep one chunk per event per query.

    Returns:
        list[dict[str, Any]]: flat export rows.
    """
    all_rows: list[dict[str, Any]] = []

    for query_index, query in enumerate(queries, start=1):
        query_id = f"q{query_index:03d}"

        retrieved_documents = retrieve_relevant_documents(query, top_k=top_k)
        sources = format_sources(retrieved_documents)

        if deduplicate_events:
            sources = deduplicate_sources_by_event_id(sources)

        print_query_results(
            query_id=query_id,
            query=query,
            sources=sources,
            preview_chars=preview_chars,
        )

        for source in sources:
            all_rows.append(
                source_to_export_row(
                    query_id=query_id,
                    query=query,
                    source=source,
                    preview_chars=preview_chars,
                )
            )

    return all_rows


def main() -> int:
    """
    CLI entry point.

    Returns:
        int: process exit code.
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    if args.preview_chars < 0:
        parser.error("--preview-chars must be >= 0")

    queries = args.query if args.query else DEFAULT_QUERIES

    try:
        rows = inspect_queries(
            queries=queries,
            top_k=args.top_k,
            preview_chars=args.preview_chars,
            deduplicate_events=args.deduplicate_events,
        )

        print()
        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)
        print(f"Queries inspected : {len(queries)}")
        print(f"Candidate rows    : {len(rows)}")
        print(f"Top-k per query   : {args.top_k}")
        print(f"Deduplicated      : {args.deduplicate_events}")

        if args.output_csv:
            write_csv(rows, args.output_csv)
            print(f"CSV written       : {args.output_csv}")

        if args.output_jsonl:
            write_jsonl(rows, args.output_jsonl)
            print(f"JSONL written     : {args.output_jsonl}")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130

    except Exception as exc:
        print("\nERROR: candidate inspection failed.", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())