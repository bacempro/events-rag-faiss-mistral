"""
CLI demo interface for the Puls-Events RAG POC.

Usage examples:
    python scripts/chat_with_events.py "Je cherche une exposition gratuite à Paris"
    python scripts/chat_with_events.py "Que faire avec des enfants ce week-end ?" --top-k 5
    python scripts/chat_with_events.py --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pulsevents_rag.rag_chain import answer_question, load_config  # noqa: E402


SEPARATOR = "=" * 90
SOURCE_SEPARATOR = "-" * 90
DEFAULT_INTERACTIVE_EXIT_COMMANDS = {"exit", "quit", "q", "stop"}


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser.

    Returns:
        argparse.ArgumentParser: configured parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Ask questions about Paris cultural events using the Puls-Events "
            "LangChain + Mistral + FAISS RAG system."
        )
    )

    parser.add_argument(
        "question",
        nargs="?",
        help=(
            "Question to ask. Omit this argument when using --interactive. "
            'Example: "Je cherche une exposition gratuite à Paris"'
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Number of FAISS chunks to retrieve. "
            "Defaults to RAG_TOP_K from .env, or 5 if not configured."
        ),
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive chat-like CLI session.",
    )

    parser.add_argument(
        "--hide-sources",
        action="store_true",
        help="Only print the generated answer, without retrieved sources.",
    )

    parser.add_argument(
        "--preview-chars",
        type=int,
        default=350,
        help="Maximum number of characters shown for each retrieved chunk preview.",
    )

    return parser


def normalize_question(question: str | None) -> str:
    """
    Validate and normalize a CLI question.

    Args:
        question: raw question from CLI or interactive prompt.

    Returns:
        str: normalized question.

    Raises:
        ValueError: if the question is empty.
    """
    normalized = (question or "").strip()
    if not normalized:
        raise ValueError("Question cannot be empty.")

    return normalized


def truncate_text(text: Any, max_chars: int) -> str:
    """
    Normalize and truncate text for terminal display.

    Args:
        text: raw text.
        max_chars: maximum displayed characters.

    Returns:
        str: truncated text.
    """
    if text is None:
        return ""

    normalized = " ".join(str(text).split())
    if max_chars <= 0:
        return ""

    if len(normalized) <= max_chars:
        return normalized

    return normalized[: max_chars - 3].rstrip() + "..."


def format_score(score: Any) -> str:
    """
    Format a FAISS score or distance for display.

    Args:
        score: raw score value.

    Returns:
        str: formatted score.
    """
    if score is None:
        return "Non disponible"

    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def get_source_value(source: dict[str, Any], key: str, fallback: str = "Non renseigné") -> str:
    """
    Get a clean display value from source metadata.

    Args:
        source: source metadata dictionary.
        key: metadata key.
        fallback: displayed value when missing.

    Returns:
        str: display value.
    """
    value = source.get(key)

    if value is None:
        return fallback

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return fallback

    return text


def print_header(title: str) -> None:
    """
    Print a section header.

    Args:
        title: section title.
    """
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)


def print_question(question: str) -> None:
    """
    Print the user question.

    Args:
        question: user question.
    """
    print_header("QUESTION")
    print(question)


def print_answer(answer: str) -> None:
    """
    Print the generated answer.

    Args:
        answer: generated answer.
    """
    print_header("RÉPONSE")
    print(answer)


def print_sources(sources: list[dict[str, Any]], preview_chars: int) -> None:
    """
    Print retrieved source metadata.

    Args:
        sources: source metadata returned by answer_question.
        preview_chars: max chunk preview length.
    """
    print_header("SOURCES RETROUVÉES")

    if not sources:
        print("Aucune source retournée.")
        return

    for source in sources:
        rank = get_source_value(source, "rank", fallback="?")
        title = get_source_value(source, "title")
        venue_name = get_source_value(source, "venue_name")
        date_range = get_source_value(source, "date_range")
        address = get_source_value(source, "address")
        city = get_source_value(source, "city")
        event_id = get_source_value(source, "event_id")
        chunk_id = get_source_value(source, "chunk_id")
        score = format_score(source.get("score"))
        preview = truncate_text(source.get("chunk_preview"), preview_chars)

        print(SOURCE_SEPARATOR)
        print(f"Source #{rank}")
        print(f"Titre        : {title}")
        print(f"Lieu         : {venue_name}")
        print(f"Dates        : {date_range}")
        print(f"Adresse      : {address}")
        print(f"Ville        : {city}")
        print(f"Score FAISS  : {score}")
        print(f"Event ID     : {event_id}")
        print(f"Chunk ID     : {chunk_id}")

        conditions = get_source_value(source, "conditions", fallback="")
        registration_url = get_source_value(source, "registration_url", fallback="")

        if conditions:
            print(f"Conditions   : {conditions}")

        if registration_url:
            print(f"Inscription  : {registration_url}")

        if preview:
            print()
            print("Aperçu chunk :")
            print(preview)

    print(SOURCE_SEPARATOR)


def print_run_metadata(result: dict[str, Any]) -> None:
    """
    Print compact model/runtime metadata.

    Args:
        result: RAG result dictionary.
    """
    model = result.get("model", "Non disponible")
    embedding_model = result.get("embedding_model", "Non disponible")
    top_k = result.get("top_k", "Non disponible")
    context_count = len(result.get("contexts", []))

    print_header("MÉTADONNÉES")
    print(f"Modèle chat       : {model}")
    print(f"Modèle embedding  : {embedding_model}")
    print(f"Top-k demandé     : {top_k}")
    print(f"Contextes reçus   : {context_count}")


def run_single_question(
    question: str,
    top_k: int | None,
    hide_sources: bool,
    preview_chars: int,
) -> int:
    """
    Run one RAG question and print the result.

    Args:
        question: user question.
        top_k: number of chunks to retrieve.
        hide_sources: whether to hide retrieved sources.
        preview_chars: source preview length.

    Returns:
        int: process exit code.
    """
    try:
        normalized_question = normalize_question(question)
        result = answer_question(normalized_question, top_k=top_k)

        print_question(result["question"])
        print_answer(result["answer"])

        if not hide_sources:
            print_sources(result.get("sources", []), preview_chars=preview_chars)

        print_run_metadata(result)
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130

    except Exception as exc:
        print("\nERROR: RAG query failed.", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def run_interactive(
    top_k: int | None,
    hide_sources: bool,
    preview_chars: int,
) -> int:
    """
    Run the interactive CLI mode.

    Args:
        top_k: number of chunks to retrieve.
        hide_sources: whether to hide retrieved sources.
        preview_chars: source preview length.

    Returns:
        int: process exit code.
    """
    config = load_config()
    effective_top_k = top_k if top_k is not None else config.default_top_k

    print(SEPARATOR)
    print("Puls-Events RAG — Mode interactif")
    print(SEPARATOR)
    print("Posez une question sur les événements culturels à Paris.")
    print(
        "Commandes de sortie : "
        + ", ".join(sorted(DEFAULT_INTERACTIVE_EXIT_COMMANDS))
    )
    print(f"Top-k utilisé : {effective_top_k}")
    print(SEPARATOR)

    while True:
        try:
            question = input("\nQuestion > ").strip()

            if not question:
                print("Question vide ignorée.")
                continue

            if question.lower() in DEFAULT_INTERACTIVE_EXIT_COMMANDS:
                print("Fin du mode interactif.")
                return 0

            result = answer_question(question, top_k=effective_top_k)

            print_answer(result["answer"])

            if not hide_sources:
                print_sources(result.get("sources", []), preview_chars=preview_chars)

            print_run_metadata(result)

        except KeyboardInterrupt:
            print("\nFin du mode interactif.")
            return 130

        except Exception as exc:
            print("\nERROR: RAG query failed.", file=sys.stderr)
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            print("Vous pouvez corriger le problème puis poser une autre question.")


def main() -> int:
    """
    CLI entry point.

    Returns:
        int: process exit code.
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be >= 1")

    if args.preview_chars < 0:
        parser.error("--preview-chars must be >= 0")

    if args.interactive:
        return run_interactive(
            top_k=args.top_k,
            hide_sources=args.hide_sources,
            preview_chars=args.preview_chars,
        )

    if not args.question:
        parser.error(
            "question is required unless --interactive is used. "
            'Example: python scripts/chat_with_events.py "Je cherche une exposition à Paris"'
        )

    return run_single_question(
        question=args.question,
        top_k=args.top_k,
        hide_sources=args.hide_sources,
        preview_chars=args.preview_chars,
    )


if __name__ == "__main__":
    raise SystemExit(main())