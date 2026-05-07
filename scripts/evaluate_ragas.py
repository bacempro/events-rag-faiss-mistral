"""
Run RAGAS evaluation for the Puls-Events RAG POC.

This script:
1. Loads the human-annotated QA dataset from JSONL.
2. Runs the existing RAG pipeline for each question.
3. Builds a RAGAS-compatible evaluation dataset.
4. Evaluates the answers with selected RAGAS metrics.
5. Writes detailed results and summary files under reports/.

Usage examples:
    python scripts/evaluate_ragas.py
    python scripts/evaluate_ragas.py --top-k 5
    python scripts/evaluate_ragas.py --limit 3
    python scripts/evaluate_ragas.py --question-id qa_001 --question-id qa_003
    python scripts/evaluate_ragas.py --answers-jsonl reports/ragas_answer_traces.jsonl

Expected outputs:
    reports/ragas_evaluation_results.csv
    reports/ragas_evaluation_summary.json

Prerequisites:
    pip install "ragas>=0.2,<0.4" "datasets>=2.18"

Required environment:
    MISTRAL_API_KEY in .env
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from datasets import Dataset  # noqa: E402

from pulsevents_rag.rag_chain import (  # noqa: E402
    answer_question,
    get_chat_model,
    get_embeddings,
    load_config,
)


DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "annotated_qa_dataset.jsonl"
DEFAULT_RESULTS_CSV_PATH = PROJECT_ROOT / "reports" / "ragas_evaluation_results.csv"
DEFAULT_SUMMARY_JSON_PATH = PROJECT_ROOT / "reports" / "ragas_evaluation_summary.json"
DEFAULT_ANSWERS_JSONL_PATH = PROJECT_ROOT / "reports" / "ragas_answer_traces.jsonl"

DEFAULT_TOP_K = 5

REQUIRED_QA_FIELDS = {
    "id",
    "question",
    "reference_answer",
    "reference_event_ids",
    "reference_event_titles",
    "question_type",
    "expected_topics",
    "notes",
}

KNOWN_NON_METRIC_COLUMNS = {
    "id",
    "question",
    "user_input",
    "answer",
    "response",
    "contexts",
    "retrieved_contexts",
    "ground_truth",
    "reference",
    "reference_answer",
    "reference_event_ids",
    "reference_event_titles",
    "question_type",
    "expected_topics",
    "notes",
    "model",
    "embedding_model",
    "top_k",
    "source_event_ids",
    "source_titles",
    "source_count",
    "error",
}


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser.

    Returns:
        argparse.ArgumentParser: configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation for the Puls-Events RAG POC."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=(
            "Path to annotated QA JSONL dataset. "
            f"Default: {DEFAULT_DATASET_PATH}"
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV_PATH,
        help=(
            "Path where detailed RAGAS CSV results will be written. "
            f"Default: {DEFAULT_RESULTS_CSV_PATH}"
        ),
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON_PATH,
        help=(
            "Path where summary JSON will be written. "
            f"Default: {DEFAULT_SUMMARY_JSON_PATH}"
        ),
    )

    parser.add_argument(
        "--answers-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional path for raw RAG answer traces before RAGAS scoring. "
            f"Example: {DEFAULT_ANSWERS_JSONL_PATH}"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve per question. Default: {DEFAULT_TOP_K}.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of QA rows to evaluate. "
            "Useful for smoke tests before running the full dataset."
        ),
    )

    parser.add_argument(
        "--question-id",
        action="append",
        default=None,
        help=(
            "Evaluate only specific QA IDs. Can be used multiple times, e.g. "
            "--question-id qa_001 --question-id qa_003."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue if one RAG answer generation fails. Failed rows will be "
            "kept with an error field and empty answer/context."
        ),
    )

    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help=(
            "Only run the RAG pipeline and write answer traces. "
            "Useful to debug generated answers before invoking RAGAS."
        ),
    )

    return parser


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Load a JSONL file.

    Args:
        path: JSONL path.

    Returns:
        list[dict[str, Any]]: parsed rows.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if a row is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Annotated QA dataset not found: {path}")

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object, "
                    f"got {type(row).__name__}."
                )

            rows.append(row)

    if not rows:
        raise ValueError(f"Annotated QA dataset is empty: {path}")

    return rows


def validate_qa_rows(rows: list[dict[str, Any]]) -> None:
    """
    Validate the minimum schema needed by the evaluator.

    Args:
        rows: annotated QA rows.

    Raises:
        ValueError: if the dataset structure is invalid.
    """
    seen_ids: set[str] = set()

    for index, row in enumerate(rows, start=1):
        missing_fields = REQUIRED_QA_FIELDS - set(row.keys())
        if missing_fields:
            raise ValueError(
                f"QA row #{index} is missing required fields: "
                f"{sorted(missing_fields)}"
            )

        row_id = str(row["id"]).strip()
        if not row_id:
            raise ValueError(f"QA row #{index} has an empty id.")

        if row_id in seen_ids:
            raise ValueError(f"Duplicate QA id: {row_id}")

        seen_ids.add(row_id)

        if not str(row["question"]).strip():
            raise ValueError(f"{row_id}: question must not be empty.")

        if not str(row["reference_answer"]).strip():
            raise ValueError(f"{row_id}: reference_answer must not be empty.")

        if not isinstance(row["reference_event_ids"], list):
            raise ValueError(f"{row_id}: reference_event_ids must be a list.")

        if not isinstance(row["reference_event_titles"], list):
            raise ValueError(f"{row_id}: reference_event_titles must be a list.")

        if not isinstance(row["expected_topics"], list):
            raise ValueError(f"{row_id}: expected_topics must be a list.")


def select_rows(
    rows: list[dict[str, Any]],
    question_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """
    Select rows by explicit IDs and/or limit.

    Args:
        rows: all QA rows.
        question_ids: optional selected IDs.
        limit: optional row limit.

    Returns:
        list[dict[str, Any]]: selected rows.

    Raises:
        ValueError: if requested IDs do not exist.
    """
    selected_rows = rows

    if question_ids:
        wanted_ids = set(question_ids)
        available_ids = {str(row["id"]) for row in rows}
        missing_ids = sorted(wanted_ids - available_ids)

        if missing_ids:
            raise ValueError(f"Unknown question IDs requested: {missing_ids}")

        selected_rows = [row for row in rows if str(row["id"]) in wanted_ids]

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1 when provided.")

        selected_rows = selected_rows[:limit]

    if not selected_rows:
        raise ValueError("No QA rows selected for evaluation.")

    return selected_rows


def normalize_text(value: Any) -> str:
    """
    Convert a value to a clean string.

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


def normalize_contexts(contexts: Any) -> list[str]:
    """
    Normalize retrieved contexts to a list of non-empty strings.

    Args:
        contexts: raw contexts.

    Returns:
        list[str]: clean contexts.
    """
    if not isinstance(contexts, list):
        return []

    clean_contexts: list[str] = []

    for context in contexts:
        text = normalize_text(context)
        if text:
            clean_contexts.append(text)

    return clean_contexts


def join_list(values: Any) -> str:
    """
    Join list-like values into a compact string for CSV output.

    Args:
        values: list-like value.

    Returns:
        str: joined string.
    """
    if not isinstance(values, list):
        return normalize_text(values)

    return " | ".join(normalize_text(value) for value in values if normalize_text(value))


def run_rag_for_row(
    row: dict[str, Any],
    top_k: int,
    continue_on_error: bool,
) -> dict[str, Any]:
    """
    Run the existing RAG pipeline for one QA row.

    Args:
        row: annotated QA row.
        top_k: retrieval depth.
        continue_on_error: whether to convert errors into row payloads.

    Returns:
        dict[str, Any]: enriched evaluation row.

    Raises:
        Exception: re-raises RAG errors unless continue_on_error is True.
    """
    row_id = str(row["id"])
    question = str(row["question"])

    try:
        result = answer_question(question, top_k=top_k)

        contexts = normalize_contexts(result.get("contexts", []))
        sources = result.get("sources", [])

        source_event_ids = []
        source_titles = []

        if isinstance(sources, list):
            for source in sources:
                if not isinstance(source, dict):
                    continue

                event_id = normalize_text(source.get("event_id"))
                title = normalize_text(source.get("title"))

                if event_id:
                    source_event_ids.append(event_id)

                if title:
                    source_titles.append(title)

        return {
            "id": row_id,
            "question": question,
            "answer": normalize_text(result.get("answer")),
            "contexts": contexts,
            "ground_truth": normalize_text(row["reference_answer"]),
            "reference_answer": normalize_text(row["reference_answer"]),
            "reference_event_ids": [str(value) for value in row["reference_event_ids"]],
            "reference_event_titles": [
                str(value) for value in row["reference_event_titles"]
            ],
            "question_type": normalize_text(row["question_type"]),
            "expected_topics": [str(value) for value in row["expected_topics"]],
            "notes": normalize_text(row["notes"]),
            "model": normalize_text(result.get("model")),
            "embedding_model": normalize_text(result.get("embedding_model")),
            "top_k": result.get("top_k", top_k),
            "source_event_ids": source_event_ids,
            "source_titles": source_titles,
            "source_count": len(sources) if isinstance(sources, list) else 0,
            "error": "",
        }

    except Exception as exc:
        if not continue_on_error:
            raise

        return {
            "id": row_id,
            "question": question,
            "answer": "",
            "contexts": [],
            "ground_truth": normalize_text(row["reference_answer"]),
            "reference_answer": normalize_text(row["reference_answer"]),
            "reference_event_ids": [str(value) for value in row["reference_event_ids"]],
            "reference_event_titles": [
                str(value) for value in row["reference_event_titles"]
            ],
            "question_type": normalize_text(row["question_type"]),
            "expected_topics": [str(value) for value in row["expected_topics"]],
            "notes": normalize_text(row["notes"]),
            "model": "",
            "embedding_model": "",
            "top_k": top_k,
            "source_event_ids": [],
            "source_titles": [],
            "source_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_rag_answer_rows(
    qa_rows: list[dict[str, Any]],
    top_k: int,
    continue_on_error: bool,
) -> list[dict[str, Any]]:
    """
    Run RAG over all selected QA rows.

    Args:
        qa_rows: selected annotated rows.
        top_k: retrieval depth.
        continue_on_error: whether to continue after answer-generation errors.

    Returns:
        list[dict[str, Any]]: answer rows.
    """
    answer_rows: list[dict[str, Any]] = []

    for index, row in enumerate(qa_rows, start=1):
        row_id = str(row["id"])
        question = str(row["question"])

        print()
        print("=" * 100)
        print(f"[{index}/{len(qa_rows)}] {row_id}")
        print("=" * 100)
        print(question)

        answer_row = run_rag_for_row(
            row=row,
            top_k=top_k,
            continue_on_error=continue_on_error,
        )

        if answer_row["error"]:
            print(f"ERROR: {answer_row['error']}")
        else:
            print(f"Answer chars : {len(answer_row['answer'])}")
            print(f"Contexts     : {len(answer_row['contexts'])}")
            print(f"Source events: {join_list(answer_row['source_event_ids'])}")

        answer_rows.append(answer_row)

    return answer_rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """
    Write rows as JSONL.

    Args:
        rows: rows to write.
        path: output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_ragas_dataset(answer_rows: list[dict[str, Any]]) -> Dataset:
    """
    Build a RAGAS-compatible Hugging Face Dataset.

    The script includes both older and newer common column names:
    - question / answer / contexts / ground_truth
    - user_input / response / retrieved_contexts / reference

    This makes the dataset easier to inspect and more tolerant of minor RAGAS
    API differences across v0.2/v0.3 releases.

    Args:
        answer_rows: generated RAG answer rows.

    Returns:
        Dataset: RAGAS input dataset.
    """
    ragas_rows: list[dict[str, Any]] = []

    for row in answer_rows:
        ragas_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "user_input": row["question"],
                "answer": row["answer"],
                "response": row["answer"],
                "contexts": row["contexts"],
                "retrieved_contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
                "reference": row["ground_truth"],
                "reference_answer": row["reference_answer"],
                "reference_event_ids": row["reference_event_ids"],
                "reference_event_titles": row["reference_event_titles"],
                "question_type": row["question_type"],
                "expected_topics": row["expected_topics"],
                "notes": row["notes"],
                "model": row["model"],
                "embedding_model": row["embedding_model"],
                "top_k": row["top_k"],
                "source_event_ids": row["source_event_ids"],
                "source_titles": row["source_titles"],
                "source_count": row["source_count"],
                "error": row["error"],
            }
        )

    return Dataset.from_list(ragas_rows)


def import_ragas_evaluate() -> Any:
    """
    Import ragas.evaluate with a clear error message.

    Returns:
        callable: ragas.evaluate function.

    Raises:
        RuntimeError: if ragas is not installed.
    """
    try:
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS is not installed. Install it with:\n"
            'pip install "ragas>=0.2,<0.4" "datasets>=2.18"'
        ) from exc

    return evaluate


def import_ragas_metrics() -> list[Any]:
    """
    Import RAGAS metrics with compatibility across v0.2/v0.3 style APIs.

    Returns:
        list[Any]: metric instances/objects.

    Raises:
        RuntimeError: if required metrics cannot be imported.
    """
    try:
        import ragas.metrics as metrics_module
    except ImportError as exc:
        raise RuntimeError(
            "Could not import ragas.metrics. Check your RAGAS installation."
        ) from exc

    metric_specs = [
        {
            "label": "faithfulness",
            "object_names": ["faithfulness"],
            "class_names": ["Faithfulness"],
        },
        {
            "label": "answer_relevancy",
            "object_names": ["answer_relevancy", "answer_relevance"],
            "class_names": ["AnswerRelevancy", "ResponseRelevancy"],
        },
        {
            "label": "context_precision",
            "object_names": ["context_precision"],
            "class_names": [
                "ContextPrecision",
                "LLMContextPrecisionWithReference",
                "LLMContextPrecisionWithoutReference",
            ],
        },
        {
            "label": "context_recall",
            "object_names": ["context_recall"],
            "class_names": ["ContextRecall", "LLMContextRecall"],
        },
    ]

    selected_metrics: list[Any] = []

    for spec in metric_specs:
        metric = None

        for object_name in spec["object_names"]:
            candidate = getattr(metrics_module, object_name, None)
            if candidate is not None:
                metric = candidate
                break

        if metric is None:
            for class_name in spec["class_names"]:
                candidate_class = getattr(metrics_module, class_name, None)
                if candidate_class is not None:
                    metric = candidate_class()
                    break

        if metric is None:
            raise RuntimeError(
                f"Could not import required RAGAS metric: {spec['label']}. "
                "Your installed RAGAS version may have an incompatible API. "
                "Use: pip install \"ragas>=0.2,<0.4\""
            )

        selected_metrics.append(metric)

    return selected_metrics


def build_ragas_evaluator_models() -> tuple[Any, Any]:
    """
    Build RAGAS evaluator LLM and embedding model.

    The underlying models are the same LangChain Mistral integrations used by
    the RAG pipeline. RAGAS v0.2/v0.3 can usually wrap LangChain models directly;
    when wrapper classes are available, this function uses them explicitly.

    Returns:
        tuple[Any, Any]: evaluator LLM and embeddings.
    """
    config = load_config()

    langchain_llm = get_chat_model(config)
    langchain_embeddings = get_embeddings(config)

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        return (
            LangchainLLMWrapper(langchain_llm),
            LangchainEmbeddingsWrapper(langchain_embeddings),
        )

    except Exception:
        # Some RAGAS versions auto-wrap LangChain models inside evaluate().
        return langchain_llm, langchain_embeddings


def call_ragas_evaluate(dataset: Dataset, metrics: list[Any]) -> Any:
    """
    Call ragas.evaluate while only passing kwargs supported by the installed version.

    Args:
        dataset: RAGAS-compatible dataset.
        metrics: selected metrics.

    Returns:
        Any: RAGAS evaluation result.
    """
    evaluate = import_ragas_evaluate()
    evaluator_llm, evaluator_embeddings = build_ragas_evaluator_models()

    signature = inspect.signature(evaluate)

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "metrics": metrics,
    }

    optional_kwargs = {
        "llm": evaluator_llm,
        "embeddings": evaluator_embeddings,
        "raise_exceptions": False,
    }

    for key, value in optional_kwargs.items():
        if key in signature.parameters:
            kwargs[key] = value

    return evaluate(**kwargs)


def result_to_dataframe(result: Any) -> Any:
    """
    Convert a RAGAS result object to a pandas DataFrame.

    Args:
        result: raw RAGAS evaluation result.

    Returns:
        pandas.DataFrame: detailed metric rows.

    Raises:
        RuntimeError: if conversion is not possible.
    """
    if hasattr(result, "to_pandas"):
        return result.to_pandas()

    if hasattr(result, "to_dataframe"):
        return result.to_dataframe()

    raise RuntimeError(
        "Could not convert RAGAS result to a DataFrame. "
        "The installed RAGAS version returned an unsupported result object."
    )


def is_number(value: Any) -> bool:
    """
    Return True if value is a finite numeric value.

    Args:
        value: value to inspect.

    Returns:
        bool: True for finite numbers.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(number)


def find_metric_columns(dataframe: Any) -> list[str]:
    """
    Detect numeric metric columns in a RAGAS results dataframe.

    Args:
        dataframe: pandas DataFrame.

    Returns:
        list[str]: metric column names.
    """
    metric_columns: list[str] = []

    for column in dataframe.columns:
        if column in KNOWN_NON_METRIC_COLUMNS:
            continue

        series = dataframe[column]
        non_null_values = [value for value in series.tolist() if value is not None]

        if not non_null_values:
            continue

        numeric_count = sum(1 for value in non_null_values if is_number(value))

        if numeric_count > 0:
            metric_columns.append(column)

    return metric_columns


def build_summary(
    dataframe: Any,
    selected_row_count: int,
    top_k: int,
    output_csv_path: Path,
    answer_trace_path: Path | None,
) -> dict[str, Any]:
    """
    Build a compact JSON evaluation summary.

    Args:
        dataframe: detailed RAGAS results dataframe.
        selected_row_count: number of evaluated rows.
        top_k: retrieval depth.
        output_csv_path: CSV output path.
        answer_trace_path: optional answer trace path.

    Returns:
        dict[str, Any]: summary payload.
    """
    metric_columns = find_metric_columns(dataframe)

    metric_means: dict[str, float | None] = {}

    for column in metric_columns:
        numeric_values = [
            float(value)
            for value in dataframe[column].tolist()
            if is_number(value)
        ]

        if numeric_values:
            metric_means[column] = sum(numeric_values) / len(numeric_values)
        else:
            metric_means[column] = None

    failed_rows = 0

    if "error" in dataframe.columns:
        failed_rows = sum(1 for value in dataframe["error"].tolist() if normalize_text(value))

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DEFAULT_DATASET_PATH),
        "row_count": selected_row_count,
        "top_k": top_k,
        "metric_columns": metric_columns,
        "metric_means": metric_means,
        "failed_rag_rows": failed_rows,
        "results_csv_path": str(output_csv_path),
        "answer_trace_jsonl_path": str(answer_trace_path) if answer_trace_path else None,
        "notes": (
            "Scores are LLM-judge based and can vary slightly between runs. "
            "Use them as comparative POC indicators, not absolute truth."
        ),
    }


def write_dataframe_csv(dataframe: Any, path: Path) -> None:
    """
    Write DataFrame to CSV.

    Args:
        dataframe: pandas DataFrame.
        path: output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False, encoding="utf-8")


def write_json(payload: dict[str, Any], path: Path) -> None:
    """
    Write JSON payload.

    Args:
        payload: JSON-serializable payload.
        path: output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


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

    try:
        qa_rows = load_jsonl(args.dataset)
        validate_qa_rows(qa_rows)

        selected_rows = select_rows(
            rows=qa_rows,
            question_ids=args.question_id,
            limit=args.limit,
        )

        print("=" * 100)
        print("Puls-Events RAGAS evaluation")
        print("=" * 100)
        print(f"Dataset rows selected : {len(selected_rows)}")
        print(f"Top-k                 : {args.top_k}")
        print(f"Skip RAGAS            : {args.skip_ragas}")
        print(f"Dataset path          : {args.dataset}")

        answer_rows = build_rag_answer_rows(
            qa_rows=selected_rows,
            top_k=args.top_k,
            continue_on_error=args.continue_on_error,
        )

        if args.answers_jsonl:
            write_jsonl(answer_rows, args.answers_jsonl)
            print()
            print(f"Answer traces written : {args.answers_jsonl}")

        if args.skip_ragas:
            print()
            print("RAGAS evaluation skipped by --skip-ragas.")
            return 0

        rows_with_errors = [row for row in answer_rows if row["error"]]
        if rows_with_errors:
            failed_ids = [row["id"] for row in rows_with_errors]
            raise RuntimeError(
                "Some RAG rows failed before RAGAS evaluation. "
                f"Failed IDs: {failed_ids}. "
                "Rerun with --continue-on-error only if you intentionally want "
                "failed rows included in the output."
            )

        ragas_dataset = build_ragas_dataset(answer_rows)
        metrics = import_ragas_metrics()

        print()
        print("=" * 100)
        print("Running RAGAS")
        print("=" * 100)
        print("Metrics:")
        for metric in metrics:
            metric_name = getattr(metric, "name", None) or type(metric).__name__
            print(f"- {metric_name}")

        result = call_ragas_evaluate(
            dataset=ragas_dataset,
            metrics=metrics,
        )

        dataframe = result_to_dataframe(result)

        write_dataframe_csv(dataframe, args.output_csv)

        summary = build_summary(
            dataframe=dataframe,
            selected_row_count=len(selected_rows),
            top_k=args.top_k,
            output_csv_path=args.output_csv,
            answer_trace_path=args.answers_jsonl,
        )

        write_json(summary, args.summary_json)

        print()
        print("=" * 100)
        print("EVALUATION COMPLETE")
        print("=" * 100)
        print(f"Results CSV  : {args.output_csv}")
        print(f"Summary JSON : {args.summary_json}")
        print()
        print("Metric means:")
        for metric_name, value in summary["metric_means"].items():
            if value is None:
                print(f"- {metric_name}: N/A")
            else:
                print(f"- {metric_name}: {value:.4f}")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130

    except Exception as exc:
        print("\nERROR: RAGAS evaluation failed.", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())