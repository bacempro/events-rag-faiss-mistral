# Puls-Events — RAG POC for Paris Cultural Events

A Retrieval-Augmented Generation (RAG) proof of concept that lets you ask natural-language questions about Paris cultural events and get grounded answers backed by real OpenAgenda data.

Built as **OpenClassrooms Data Engineer Project 11**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Step 1 — Fetch events](#step-1--fetch-events)
  - [Step 2 — Preprocess events](#step-2--preprocess-events)
  - [Step 3 — Build the vector index](#step-3--build-the-vector-index)
  - [Step 4 — Chat with events](#step-4--chat-with-events)
  - [Step 5 — Evaluate with RAGAS](#step-5--evaluate-with-ragas)
- [Running Tests](#running-tests)
- [Scripts Reference](#scripts-reference)
- [RAGAS Evaluation Results](#ragas-evaluation-results)
- [Known Limitations](#known-limitations)
- [Environment Variables Reference](#environment-variables-reference)

---

## Overview

Puls-Events answers questions like:

- *"Je cherche un concert de jazz à Paris ce mois-ci"*
- *"Propose-moi une sortie culturelle gratuite pour enfants"*
- *"Y a-t-il du théâtre contemporain à Paris ?"*

The system:

1. Fetches Paris cultural events from the [OpenAgenda](https://openagenda.com/) API (1-year rolling window).
2. Cleans and structures the data into a tabular dataset.
3. Chunks event texts and builds a FAISS vector index using Mistral embeddings.
4. At query time, retrieves the most semantically relevant event chunks and generates a grounded French answer via a Mistral chat model through LangChain.
5. Evaluates answer quality using four RAGAS metrics on an annotated Q/A dataset.

The chatbot **does not hallucinate events** — if retrieved context is insufficient, it says so rather than inventing facts.

---

## Architecture

```
OpenAgenda API
      │
      ▼
fetch_openagenda_events.py  →  data/raw/openagenda_events_raw.json
      │
      ▼
preprocess_events.py        →  data/processed/events_clean.csv
      │
      ▼
build_faiss_index.py        →  vectorstore/faiss_index/
  (Mistral embeddings)              index.faiss
                                    index.pkl
                                    build_metadata.json
      │
      ▼
  Query time
      │
      ▼
rag_chain.py  ──────────────  FAISS similarity search (top-k chunks)
  (LangChain)                       │
      │                             ▼
      └──────────────────►  Mistral chat model  →  Grounded French answer
```

**Key design choices:**

- Filtering happens at API call time (`timings[gte]`), not in preprocessing — only Paris events within the last 12 months (or future) are fetched.
- Embeddings use `mistral-embed`; the chat model uses `mistral-small-latest` (configurable via `.env`).
- FAISS uses the CPU backend (`faiss-cpu`) — no GPU required.
- The vectorstore is loaded once per process and cached (`@lru_cache`).
- Answers are grounded in retrieved context; the prompt explicitly forbids invented events, dates, venues, or prices.
- No conversation memory is implemented — this is a single-turn POC.

---

## Project Structure

```
pulsevents/
├── .env.example                   # Environment variable template (copy to .env)
├── environment.yml                # Conda environment definition
├── requirements.txt               # pip dependency list
│
├── data/
│   ├── raw/
│   │   └── openagenda_events_raw.json     # Raw OpenAgenda API extraction
│   ├── processed/
│   │   ├── events_clean.csv               # Cleaned event dataset (RAG input)
│   │   └── events_rejected.json           # Rejected records audit log
│   └── evaluation/
│       ├── annotated_qa_dataset.jsonl     # 12-row annotated Q/A dataset for RAGAS
│       ├── candidate_events_for_qa.csv    # Candidate events used to build the dataset
│       └── candidate_events_for_qa.jsonl
│
├── docs/
│   └── events_clean_schema.md     # Cleaned dataset schema contract
│
├── reports/
│   ├── openagenda_raw_schema_summary.json # Raw schema inspection output
│   ├── ragas_answer_traces.jsonl          # Full RAG answer traces
│   ├── ragas_evaluation_results.csv       # Per-row RAGAS metric scores
│   └── ragas_evaluation_summary.json      # Aggregated RAGAS summary
│
├── scripts/
│   ├── setup_environment.py       # Creates/reuses the Conda environment
│   ├── check_environment.py       # Validates imports, FAISS, and API config
│   ├── fetch_openagenda_events.py # Fetches raw Paris events from OpenAgenda
│   ├── inspect_openagenda_raw.py  # Inspects raw schema field completeness
│   ├── preprocess_events.py       # Normalises raw events into clean CSV
│   ├── build_faiss_index.py       # Builds FAISS vectorstore with Mistral embeddings
│   ├── search_faiss_index.py      # Semantic search smoke-test utility
│   ├── chat_with_events.py        # CLI demo (single question or interactive)
│   ├── inspect_rag_candidates.py  # Retrieval-only helper for QA dataset design
│   └── evaluate_ragas.py          # Runs RAG + RAGAS evaluation, writes reports
│
├── src/
│   └── pulsevents_rag/
│       ├── __init__.py
│       └── rag_chain.py           # Reusable RAG engine (main package module)
│
├── tests/
│   ├── test_events_data.py        # Validates processed event dataset
│   ├── test_vectorstore_data.py   # Validates FAISS vectorstore integrity
│   ├── test_rag_pipeline.py       # Validates RAG module runtime behaviour
│   └── test_evaluation_dataset.py # Validates annotated Q/A dataset structure
│
└── vectorstore/
    └── faiss_index/
        ├── index.faiss            # FAISS vector index
        ├── index.pkl              # LangChain docstore metadata (pickle)
        └── build_metadata.json    # Vectorstore build audit metadata
```

---

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python 3.11 (managed by Conda)
- A **Mistral API key** — get one at [console.mistral.ai](https://console.mistral.ai/)
- An **OpenAgenda API key** — get one at [openagenda.com](https://openagenda.com/) (required only to re-fetch raw data)

---

## Installation

**1. Clone the repository**

```bash
git clone <repository-url>
cd pulsevents
```

**2. Create the Conda environment and install dependencies**

```bash
python scripts/setup_environment.py
```

This creates a reusable Conda environment named `local-ai-rag`, installs all dependencies from `requirements.txt`, and registers a Jupyter kernel.

**3. Activate the environment**

```bash
conda activate local-ai-rag
```

**4. Validate the environment**

```bash
python scripts/check_environment.py
```

Expected output: all imports succeed, FAISS CPU smoke test passes, and the Mistral package is found.

---

## Configuration

Copy the template and fill in your secrets:

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```dotenv
OPENAGENDA_API_KEY=<your-openagenda-key>
MISTRAL_API_KEY=<your-mistral-key>
```

All other variables have working defaults (see [Environment Variables Reference](#environment-variables-reference)).

---

## Pipeline Walkthrough

Run these steps in order when setting up from scratch. If you already have the vectorstore built, skip straight to [Step 4](#step-4--chat-with-events).

### Step 1 — Fetch events

```bash
python scripts/fetch_openagenda_events.py
```

Fetches Paris cultural events from OpenAgenda with a 1-year rolling timing window and writes the raw payload to `data/raw/openagenda_events_raw.json`.

Optionally inspect the raw schema before preprocessing:

```bash
python scripts/inspect_openagenda_raw.py
```

### Step 2 — Preprocess events

```bash
python scripts/preprocess_events.py
```

Normalises the raw extraction into `data/processed/events_clean.csv` (clean rows) and `data/processed/events_rejected.json` (malformed records). Builds the `text_for_embedding` field used for vectorization.

Validate the processed dataset:

```bash
pytest tests/test_events_data.py -v
```

### Step 3 — Build the vector index

```bash
python scripts/build_faiss_index.py
```

Chunks `text_for_embedding` with `RecursiveCharacterTextSplitter`, generates embeddings via the Mistral API, and builds a persisted LangChain FAISS vectorstore under `vectorstore/faiss_index/`.

> **Note:** This step calls the Mistral embeddings API and may take a few minutes depending on the number of events.

Smoke-test semantic search:

```bash
python scripts/search_faiss_index.py
# Or with a custom query:
python scripts/search_faiss_index.py "concert jazz à Paris" --top-k 5
```

Validate vectorstore integrity:

```bash
pytest tests/test_vectorstore_data.py -v
```

### Step 4 — Chat with events

Single question:

```bash
python scripts/chat_with_events.py "Je cherche une exposition gratuite à Paris" --top-k 5
```

Interactive mode:

```bash
python scripts/chat_with_events.py --interactive
```

Additional CLI options:

| Option | Description |
|---|---|
| `--top-k N` | Number of chunks to retrieve (default: `RAG_TOP_K` from `.env`) |
| `--hide-sources` | Print only the answer, suppress retrieved source details |
| `--preview-chars N` | Chunk preview length in source display (default: 200) |

You can also use the RAG module directly in Python:

```python
import sys
sys.path.insert(0, "src")

from pulsevents_rag.rag_chain import answer_question

result = answer_question("Propose-moi trois sorties culturelles à Paris", top_k=5)
print(result["answer"])
print(result["sources"])
```

The `answer_question` function returns:

| Key | Type | Description |
|---|---|---|
| `answer` | `str` | Generated grounded answer |
| `contexts` | `list[str]` | Retrieved chunk texts used as context |
| `sources` | `list[dict]` | Rich source metadata per retrieved chunk |
| `model` | `str` | Mistral chat model used |
| `embedding_model` | `str` | Mistral embedding model used |
| `top_k` | `int` | Effective number of chunks retrieved |

### Step 5 — Evaluate with RAGAS

Run the full evaluation over the annotated 12-row Q/A dataset:

```bash
python scripts/evaluate_ragas.py --top-k 5 --answers-jsonl reports/ragas_answer_traces.jsonl
```

> **Windows only:** if you see an OpenMP runtime conflict, run this first in PowerShell:
> ```powershell
> $env:KMP_DUPLICATE_LIB_OK="TRUE"
> ```

Useful variants:

```bash
# Smoke test on 2 rows, skip RAGAS scoring
python scripts/evaluate_ragas.py --limit 2 --skip-ragas

# Evaluate specific questions only
python scripts/evaluate_ragas.py --question-id qa_001 --question-id qa_003
```

Results are written to:

- `reports/ragas_evaluation_results.csv` — per-row scores
- `reports/ragas_evaluation_summary.json` — aggregated metrics + validity diagnostics
- `reports/ragas_answer_traces.jsonl` — full RAG answer traces (when `--answers-jsonl` is provided)

---

## Running Tests

Run the full test suite:

```bash
pytest -v
```

Or run individual suites:

```bash
pytest tests/test_events_data.py -v        # Processed dataset validation
pytest tests/test_vectorstore_data.py -v   # FAISS vectorstore integrity
pytest tests/test_rag_pipeline.py -v       # RAG pipeline runtime behaviour
pytest tests/test_evaluation_dataset.py -v # Annotated Q/A dataset structure
```

> `test_rag_pipeline.py` and `test_vectorstore_data.py` require a valid `MISTRAL_API_KEY` and a built vectorstore.

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/setup_environment.py` | Creates/reuses the `local-ai-rag` Conda env and installs dependencies |
| `scripts/check_environment.py` | Validates Python version, package imports, FAISS CPU, and Mistral readiness |
| `scripts/fetch_openagenda_events.py` | Fetches raw Paris events from OpenAgenda with API-side timing filter |
| `scripts/inspect_openagenda_raw.py` | Reports field completeness across the raw extraction |
| `scripts/preprocess_events.py` | Cleans raw events → `events_clean.csv` + `events_rejected.json` |
| `scripts/build_faiss_index.py` | Builds FAISS vectorstore with Mistral embeddings and saves to `vectorstore/` |
| `scripts/search_faiss_index.py` | Semantic search smoke test against the built vectorstore |
| `scripts/chat_with_events.py` | CLI interface for single-question and interactive RAG usage |
| `scripts/inspect_rag_candidates.py` | Retrieval-only helper used for Q/A dataset construction |
| `scripts/evaluate_ragas.py` | Runs RAGAS evaluation over the annotated Q/A dataset |

---

## RAGAS Evaluation Results

Evaluation was run over a 12-row annotated Q/A dataset (10 positive recommendation cases + 2 negative insufficient-context cases) using `top_k=5`.

| Metric | Score | Interpretation |
|---|---|---|
| **Faithfulness** | 0.788 | Good — answers are mostly supported by retrieved context |
| **Answer Relevancy** | 0.695 | Acceptable — answers generally address the question |
| **Context Precision** | 0.480 | Moderate — retrieved chunks contain relevant material but also noise |
| **Context Recall** | 0.479 | Moderate — retrieved chunks do not always cover all reference information |

All 12 rows were evaluated with 0 failed RAG rows and no unavailable metrics.

> Scores are LLM-judge-based (Mistral as evaluator) and should be treated as directional POC indicators, not absolute ground truth. Runs may produce slightly different values due to model non-determinism.

---

## Known Limitations

- **Retrieval quality** is the main improvement area — context precision and recall are moderate (~0.48). Chunking strategy, metadata filtering, or a reranking step could improve both.
- **Single-turn only** — no conversation history is tracked. Each question is answered independently.
- **API dependency** — both the data pipeline (OpenAgenda) and the RAG chain (Mistral) require live API access and valid keys.
- **Vectorstore rebuild cost** — rebuilding the FAISS index requires Mistral API calls for ~1 900 chunks, which takes time and consumes API quota.
- **Windows runtime workaround** — a `KMP_DUPLICATE_LIB_OK=TRUE` environment variable is needed on some Windows setups before running RAGAS due to an OpenMP duplicate-library conflict. This is a local workaround, not a production fix.
- **RAGAS stability** — RAGAS 0.3.x with a Mistral evaluator can produce metric-level exceptions. The `answer_relevancy` metric uses `AnswerRelevancy(strictness=1)` as a stability fix; this may need revisiting on RAGAS upgrades.
- **Dataset freshness** — the raw dataset is a snapshot from a specific fetch date. Re-running `fetch_openagenda_events.py` shifts the 1-year window and will change the events in the index.

---

## Environment Variables Reference

All variables are read from `.env` (or the shell environment). Copy `.env.example` to `.env` to get started.

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAGENDA_API_KEY` | Yes (fetch only) | — | OpenAgenda API key |
| `OPENAGENDA_AGENDA_SLUG` | No | `culture` | OpenAgenda agenda slug to query |
| `OPENAGENDA_CITY` | No | `Paris` | City filter for the OpenAgenda fetch |
| `MISTRAL_API_KEY` | Yes | — | Mistral API key (required from Step 3 onward) |
| `MISTRAL_EMBEDDING_MODEL` | No | `mistral-embed` | Mistral model used for embeddings |
| `MISTRAL_CHAT_MODEL` | No | `mistral-small-latest` | Mistral model used for answer generation |
| `RAG_TOP_K` | No | `5` | Default number of chunks retrieved per query |
| `RAG_TEMPERATURE` | No | `0.0` | Chat model temperature (0 = deterministic) |
| `FAISS_CHUNK_SIZE` | No | `1000` | Max characters per text chunk |
| `FAISS_CHUNK_OVERLAP` | No | `150` | Overlap characters between consecutive chunks |
| `FAISS_INDEX_PATH` | No | `vectorstore/faiss_index` | Path to the persisted FAISS index |
