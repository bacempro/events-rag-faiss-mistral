# Project Progress

## Metadata
- Project: Puls-Events RAG POC — OpenClassrooms Data Engineer Project 11
- Current step: Step 2 — Collect and preprocess OpenAgenda data
- Status: Step 2 completed and locally validated by user
- Last updated: 2026-05-04
- Source files reviewed:
  - `Project11_PlusEvents_Exercice_text.txt`
  - `project11_pulsevents_plan.md`
  - `openclassrooms_project_agent_prompt.md`
- Related project files:
  - `requirements.txt`
  - `environment.yml`
  - `.env.example`
  - `.gitignore`
  - `scripts/setup_environment.py`
  - `scripts/check_environment.py`
  - `src/pulsevents_rag/__init__.py`
  - `scripts/fetch_openagenda_events.py`
  - `scripts/inspect_openagenda_raw.py`
  - `scripts/preprocess_events.py`
  - `docs/events_clean_schema.md`
  - `reports/openagenda_raw_schema_summary.json`
  - `data/raw/openagenda_events_raw.json`
  - `data/processed/events_clean.csv`
  - `data/processed/events_rejected.json`
  - `tests/test_events_data.py`

## Step 1 — Prepare the development environment
- Date added: 2026-05-04
- Status: Completed

### Step Goal
- Configure a reproducible Python development environment for the Puls-Events RAG POC.
- Install and validate core libraries required by the project:
  - Python 3
  - LangChain
  - FAISS CPU backend
  - Mistral integration support
- Provide setup documentation and validation scripts so the environment can be recreated later.

### Constraints
- Project must be reproducible through documented setup commands.
- FAISS must use the CPU backend to avoid GPU-related installation/runtime issues.
- Environment should support later RAG steps:
  - OpenAgenda data extraction and preprocessing
  - text chunking
  - embedding/vectorization
  - FAISS vector index creation
  - LangChain orchestration
  - Mistral API calls
- User preferred a reusable Conda environment for local-model/RAG work rather than a project-specific name.

### Work Completed
- Created a reusable Conda environment setup approach.
- Selected generic Conda environment name: `local-ai-rag`.
- Created dependency files:
  - `requirements.txt`
  - `environment.yml`
- Created setup automation script:
  - `scripts/setup_environment.py`
- Created validation script:
  - `scripts/check_environment.py`
- Created initial project folder structure:
  - `data/raw/`
  - `data/processed/`
  - `notebooks/`
  - `reports/`
  - `scripts/`
  - `src/pulsevents_rag/`
  - `tests/`
  - `vectorstore/`
- Created configuration/documentation starter files:
  - `.env.example`
  - `.gitignore`
- User ran the setup and validation locally.
- User confirmed:
  - Conda environment exists.
  - Requirements are installed.
  - Tests/checks are OK.

### Key Decisions
- Decision: Use a new Conda environment instead of an existing one.
  - Reason: Better reproducibility, fewer dependency conflicts, cleaner OpenClassrooms delivery.
  - Impact: Future commands should assume `conda activate local-ai-rag` before running scripts.

- Decision: Use generic environment name `local-ai-rag`.
  - Reason: User wanted an environment reusable for local models and RAG projects, not specific to Puls-Events.
  - Impact: Do not rename the environment without user confirmation.

- Decision: Use FAISS CPU package.
  - Reason: Exercise specifically warns to ensure FAISS CPU backend is installed and working.
  - Impact: Dependency should remain `faiss-cpu`; do not replace with GPU FAISS unless explicitly needed later.

- Decision: Include both `requirements.txt` and `environment.yml`.
  - Reason: `requirements.txt` is simple for pip-based reproducibility; `environment.yml` is useful for Conda recreation.
  - Impact: Keep both files synchronized when dependencies change.

- Decision: Keep `.env.example` but do not require `.env` yet.
  - Reason: Mistral API usage is needed in later steps, but Step 1 only validates installation/import readiness.
  - Impact: `MISTRAL_API_KEY` will become required when implementing embeddings or LLM calls.

### User Questions / Confusions / Problems
- Question / confusion: User initially preferred using an existing Conda environment if possible.
  - Resolution / explanation: A separate environment was recommended for reproducibility and dependency isolation. User agreed to create a new reusable environment.
  - Impact on project: All future setup/run instructions should target `local-ai-rag`.

- Question / confusion: User wanted to know whether any more information was needed before starting Step 1.
  - Resolution / explanation: No additional information was needed; Step 1 requirements were sufficiently defined by the project instructions and plan.
  - Impact on project: Step 1 was implemented without further blocking questions.

### Learning Notes Discussed
- Topic: Environment isolation for RAG/local-model projects.
  - What was learned: A dedicated Conda environment reduces version conflicts with packages such as `faiss-cpu`, `numpy`, `pydantic`, and LangChain packages.
  - Why it matters here: Later steps will combine data processing, vector indexing, LangChain, and Mistral dependencies; isolation makes failures easier to diagnose.

- Topic: CPU FAISS validation.
  - What was learned: FAISS should be validated with a small smoke test, not only by checking package installation.
  - Why it matters here: The project requires a functional vector store in Step 3, so Step 1 must verify that FAISS can actually create/search an index.

### Assumptions
- User has Miniconda installed and available from the terminal.
- User is working locally on a machine capable of running Python 3.11 environment packages.
- User has successfully executed the setup and validation scripts locally.
- Mistral API key is not required for Step 1 validation.
- OpenAgenda API access and geographic scope will be handled in Step 2.

### Files Created
- `requirements.txt` — pip dependency list for the RAG POC.
- `environment.yml` — Conda environment definition for `local-ai-rag`.
- `.env.example` — template for future environment variables, including Mistral API key placeholder.
- `.gitignore` — excludes local env files, caches, data outputs, and generated artifacts.
- `scripts/setup_environment.py` — creates/reuses the Conda env, installs dependencies, registers Jupyter kernel, and runs validation.
- `scripts/check_environment.py` — validates Python version, imports, LangChain readiness, FAISS CPU smoke test, and Mistral package readiness.
- `src/pulsevents_rag/__init__.py` — initial Python package marker.
- `data/raw/.gitkeep` — keeps raw data directory in version control.
- `data/processed/.gitkeep` — keeps processed data directory in version control.
- `vectorstore/.gitkeep` — keeps vector store directory in version control before index files exist.

### Files Updated
- None after local validation.

### Commands to Run
```bash
# From the project root, create/reuse the Conda env and install dependencies
python scripts/setup_environment.py

# Activate the environment
conda activate local-ai-rag

# Validate the environment manually
python scripts/check_environment.py
```

### Validation
- Check: Run setup script.
  - Expected result: Conda environment `local-ai-rag` exists and dependencies install successfully.

- Check: Run environment validation script.
  - Expected result: Imports succeed for required libraries.

- Check: Run FAISS smoke test.
  - Expected result: FAISS creates a small CPU index and returns search results without error.

- Check: User local validation.
  - Expected result: User confirms tests/checks are OK.
  - Actual result: User confirmed environment exists, requirements are installed, and tests are OK.

### Blockers / Risks
- Mistral API key is not configured yet; this is acceptable for Step 1 but will matter in later steps.
- Dependency versions may need adjustment if later LangChain/Mistral APIs require different package combinations.
- OpenAgenda API details and selected geographic scope are not defined yet; this is a Step 2 decision.

### Open Questions
- Which geographic scope should be used for OpenAgenda data in Step 2?
  - Candidate likely scope: Paris or Île-de-France, but this has not been confirmed.
- Should Step 2 use the OpenAgenda API directly, exported datasets, or both as fallback?
- Which event fields should be mandatory in the cleaned dataset?
- Which date logic should be used exactly for “recent events of less than one year”:
  - events starting after `today - 365 days`
  - events updated after `today - 365 days`
  - or both start/end date rules?

### Next Step Notes
- Next step is Step 2 — Collect and preprocess OpenAgenda data.
- Before implementation, reread the Step 2 instructions from the original project file.
- Step 2 must define:
  - data source/API endpoint
  - geographic scope
  - date filtering rule
  - raw output format
  - processed output format
  - minimum required event fields
  - unit-test strategy for recency and geography validation
- Recommended output files for Step 2 are likely:
  - `scripts/fetch_openagenda_events.py`
  - `scripts/preprocess_events.py`
  - `data/raw/openagenda_events_raw.json` or `.jsonl`
  - `data/processed/events_clean.csv` or `.parquet`
  - `tests/test_events_data.py`
- Do not implement vectorization or FAISS indexing in Step 2 except for notes needed by Step 3.

### Resume Context for AI
- Important technical facts:
  - Project: Puls-Events RAG POC for OpenClassrooms Data Engineer Project 11.
  - Required stack: Python 3, LangChain, FAISS CPU, Mistral API.
  - User created and validated a new Conda environment named `local-ai-rag`.
  - Step 1 is complete.
  - User confirmed all requirements installed and tests/checks are OK.
  - Project structure has been initialized.
  - Mistral API key is not required yet but will be needed later.
  - FAISS must remain CPU-based unless user explicitly changes this.

- Things not to change without confirmation:
  - Do not rename Conda environment `local-ai-rag`.
  - Do not switch from `faiss-cpu` to GPU FAISS.
  - Do not select the OpenAgenda geographic scope silently if it materially affects the dataset and demo.
  - Do not implement later steps before the user explicitly asks.

- Later change notes affecting this step:
  - None.

## Step 2 — Collect and preprocess OpenAgenda data
- Date added: 2026-05-04
- Status: Completed and locally validated by user

### Step Goal
- Retrieve public cultural event data from OpenAgenda for a selected geographic scope.
- Keep only events matching the selected Paris scope and the required one-year/future timing window through the API request.
- Preserve raw OpenAgenda data for traceability.
- Inspect the raw schema before preprocessing.
- Normalize the raw events into a clean tabular dataset ready for Step 3 vectorization and FAISS indexing.
- Add Python tests validating geographic scope, recency/timing constraints, required fields, coordinates, and embedding text readiness.

### Constraints
- Geographic scope: Paris.
- Data source: OpenAgenda API only; no fallback source for this project.
- Agenda used by the user: `culture`.
- Resolved OpenAgenda agenda UID: `86244142`.
- Recency rule must be handled in the API call, not as primary filtering in preprocessing.
- API date rule used: `timings[gte] = 2025-05-04`, equivalent to `today - 365 days` on 2026-05-04.
- Future events are included.
- Preprocessing must only normalize data and reject malformed/unusable rows.
- Do not implement embeddings, FAISS indexing, LangChain RAG, or Mistral calls in Step 2.
- Tests must validate that the processed dataset satisfies the selected geographic and timing constraints.

### Work Completed
- Created raw OpenAgenda extraction script:
  - `scripts/fetch_openagenda_events.py`
- The script:
  - loads OpenAgenda configuration from local environment variables
  - resolves the agenda UID from the configured agenda slug
  - calls OpenAgenda API v2
  - requests Paris events with `timings[gte]` applied at API-call level
  - uses `detailed=1`
  - paginates through results
  - writes raw payload and extraction metadata to `data/raw/openagenda_events_raw.json`
- User ran the fetch script locally.
- Fetch result confirmed by user:
  - Agenda slug: `culture`
  - Agenda UID: `86244142`
  - City: `Paris`
  - `timings[gte]`: `2025-05-04`
  - Events fetched: `1030`
  - Output: `data/raw/openagenda_events_raw.json`
- Created raw schema inspection script:
  - `scripts/inspect_openagenda_raw.py`
- User ran the schema inspection locally.
- Schema inspection result confirmed by user:
  - Total events: `1030`
  - `title.fr`: `1030 / 1030`
  - `description.fr`: `1030 / 1030`
  - `location.city`: `1030 / 1030`
  - `location.latitude`: `1030 / 1030`
  - `location.longitude`: `1030 / 1030`
  - events with timings: `1030 / 1030`
  - total timing slots: `3251`
  - missing `longDescription`: `4`
  - missing `keywords`: `802`
  - missing `nextTiming`: `714`
  - missing `registration`: `664`
  - missing `conditions`: `507`
- Created schema definition document:
  - `docs/events_clean_schema.md`
- Defined the cleaned dataset target schema before implementing preprocessing.
- Created preprocessing script:
  - `scripts/preprocess_events.py`
- The preprocessing script:
  - loads `data/raw/openagenda_events_raw.json`
  - extracts French title, description, long description, date range, timing summary, location, registration, origin agenda, and conditions
  - normalizes whitespace and strips HTML tags
  - creates `text_for_embedding`
  - writes clean rows to `data/processed/events_clean.csv`
  - writes rejected rows to `data/processed/events_rejected.json`
  - rejects only malformed/unusable records, not records based on the primary one-year API filter
- User ran preprocessing locally.
- Preprocessing result confirmed by user:
  - Raw events: `1030`
  - Clean events: `1030`
  - Rejected events: `0`
- Created dataset validation tests:
  - `tests/test_events_data.py`
- Tests validate:
  - processed dataset exists
  - processed dataset is not empty
  - required columns exist
  - required fields are non-empty
  - all rows have `city = Paris`
  - all rows have at least one recent or future timing
  - coordinates are plausible for Paris
  - `text_for_embedding` is usable
  - `timings_count` is a positive integer
- Initial recency test failed because it checked `first_timing_begin`.
- The test was corrected to validate `last_timing_begin` as the available CSV proxy for “at least one timing in the accepted API window.”
- User reran tests locally and confirmed all tests passed.

### Key Decisions
- Decision: Use Paris as the geographic scope.
  - Reason: User confirmed Paris is enough; it is simple, defensible, and suitable for a cultural-event demo.
  - Impact: Tests validate `city = Paris` and Paris-area coordinates.

- Decision: Use OpenAgenda API only, with no fallback source.
  - Reason: User explicitly said not to worry about fallback for this project.
  - Impact: Step 2 reproducibility depends on OpenAgenda API availability and the configured API key.

- Decision: Use agenda slug `culture`.
  - Reason: User ran the fetch script with `OPENAGENDA_AGENDA_SLUG=culture` and successfully retrieved data.
  - Impact: Current raw dataset is tied to OpenAgenda agenda UID `86244142`.

- Decision: Apply primary recency filtering in the API request, not in preprocessing.
  - Reason: User explicitly requested not to bring more data than needed and not to rely on preprocessing for the one-year filter.
  - Impact: `scripts/fetch_openagenda_events.py` is responsible for `timings[gte]`; `scripts/preprocess_events.py` only normalizes and validates structure.

- Decision: Include future events.
  - Reason: User confirmed future events should be included.
  - Impact: The API lower-bound rule is `timings[gte] = today - 365 days`, with no upper bound.

- Decision: Preserve raw API response separately from processed data.
  - Reason: Traceability and easier debugging.
  - Impact: Raw data is stored in `data/raw/openagenda_events_raw.json`; clean data is stored in `data/processed/events_clean.csv`.

- Decision: Build `text_for_embedding` in Step 2 but do not generate embeddings.
  - Reason: Step 2 prepares data for Step 3; vectorization belongs to Step 3.
  - Impact: Step 3 can use `text_for_embedding` directly for chunking/vectorization.

- Decision: Correct recency validation from `first_timing_begin` to `last_timing_begin` proxy.
  - Reason: OpenAgenda can return long-running or recurring events whose `firstTiming` is older than the API lower bound but whose later timings are inside the valid period.
  - Impact: Tests now validate that each event has at least one valid recent/future timing, instead of incorrectly requiring the first historical timing to be recent.

### User Questions / Confusions / Problems
- Question / confusion: User asked why events older than one year appeared even though the API call filtered by `timings[gte]`.
  - Resolution / explanation: The test checked `first_timing_begin`, but OpenAgenda returns recurring/long-running events when at least one timing matches the `timings[gte]` filter. `firstTiming` can predate the window while `lastTiming` or another timing is valid.
  - Impact on project: Recency test was corrected to validate `last_timing_begin` as the available proxy for an accepted timing in the current CSV schema.

- Question / confusion: User wanted to avoid excessive data retrieval and avoid relying on preprocessing for one-year filtering.
  - Resolution / explanation: The API call uses `timings[gte]` and preprocessing does not apply primary date filtering.
  - Impact on project: Step 2 respects data-minimization and keeps filtering responsibility in the extraction script.

- Question / confusion: User deleted the earlier project setup documentation file and wants it generated only at the end of the project.
  - Resolution / explanation: Current progress file no longer tracks that file as a current project artifact.
  - Impact on project: Do not add or update that documentation file in intermediate steps unless the user explicitly asks.

### Learning Notes Discussed
- Topic: Difference between OpenAgenda event-level data and timing-level filtering.
  - What was learned: An OpenAgenda event can be old at the event level but still valid for the selected time window if it has one or more recent/future timings.
  - Why it matters here: Dataset validation should test accepted timings, not blindly test the first historical timing.

- Topic: API filtering versus preprocessing validation.
  - What was learned: The extraction script should reduce retrieved data with API parameters, while preprocessing should normalize and reject malformed records only.
  - Why it matters here: This keeps Step 2 aligned with the user's data-minimization requirement and makes tests a validation layer, not a replacement for extraction logic.

- Topic: Schema inspection before preprocessing.
  - What was learned: Inspecting all 1030 raw records confirmed that French title/description, location, coordinates, and timings are consistently present.
  - Why it matters here: The preprocessing script can safely treat those fields as required while keeping `longDescription`, `registration`, `conditions`, and `keywords` optional.

### Assumptions
- User has a valid OpenAgenda API key configured locally.
- Local `.env` contains or can contain:
  - `OPENAGENDA_API_KEY`
  - `OPENAGENDA_AGENDA_SLUG=culture`
  - `OPENAGENDA_CITY=Paris`
- User works from the project root when running scripts.
- Conda environment remains `local-ai-rag`.
- `data/raw/openagenda_events_raw.json` contains the 1030-event API extraction confirmed by the user.
- `data/processed/events_clean.csv` contains 1030 clean rows and 0 rejected rows after preprocessing.
- `tests/test_events_data.py` passes locally after correcting the recency test.

### Files Created
- `scripts/fetch_openagenda_events.py` — fetches raw OpenAgenda events with Paris and `timings[gte]` filters applied in the API call.
- `scripts/inspect_openagenda_raw.py` — inspects raw OpenAgenda schema and field completeness across fetched events.
- `docs/events_clean_schema.md` — documents the target cleaned dataset schema and preprocessing contract.
- `reports/openagenda_raw_schema_summary.json` — stores raw schema inspection output.
- `scripts/preprocess_events.py` — converts raw OpenAgenda events into a clean CSV dataset and rejected-events file.
- `data/raw/openagenda_events_raw.json` — raw OpenAgenda extraction with metadata and events.
- `data/processed/events_clean.csv` — clean structured event dataset ready for Step 3.
- `data/processed/events_rejected.json` — rejected event audit file; currently empty because 0 events were rejected.
- `tests/test_events_data.py` — pytest validation suite for processed event data.

### Files Updated
- `progress.md` — appended Step 2 completion details and updated global metadata.

### Commands to Run
```bash
# Activate the environment
conda activate local-ai-rag

# Fetch raw OpenAgenda events for Paris with API-side timing filter
python scripts/fetch_openagenda_events.py

# Inspect the raw schema
python scripts/inspect_openagenda_raw.py

# Preprocess raw events into clean CSV
python scripts/preprocess_events.py

# Validate processed dataset
pytest tests/test_events_data.py -v
```

### Validation
- Check: Run OpenAgenda fetch script.
  - Expected result: `data/raw/openagenda_events_raw.json` is created with Paris events and API metadata.
  - Actual result confirmed by user: 1030 events fetched from agenda slug `culture`, UID `86244142`, city `Paris`, `timings[gte] = 2025-05-04`.

- Check: Run raw schema inspection script.
  - Expected result: `reports/openagenda_raw_schema_summary.json` is created.
  - Actual result confirmed by user: 1030 events inspected; title, description, location, coordinates, and timings are present for all events.

- Check: Run preprocessing script.
  - Expected result: `data/processed/events_clean.csv` and `data/processed/events_rejected.json` are created.
  - Actual result confirmed by user: 1030 clean events, 0 rejected events.

- Check: Run dataset tests.
  - Expected result: all tests pass.
  - Actual result confirmed by user: all tests passed after correcting recency validation from `first_timing_begin` to `last_timing_begin` proxy.

### Blockers / Risks
- OpenAgenda API availability and response shape may change.
- Current recency validation uses `last_timing_begin` as a proxy because the cleaned CSV does not preserve the complete timings list.
- A more precise future improvement would store derived timing fields such as `first_valid_timing_begin` and `valid_timings_count` based on the API lower bound.
- Mistral API key is still not required for Step 2, but will be required in later RAG/LLM steps.
- The cleaned dataset is tied to data fetched on 2026-05-04 with lower bound `2025-05-04`; rerunning later will shift the dynamic one-year window unless the script is modified to accept a fixed reference date.

### Open Questions
- None blocking for Step 2.
- For Step 3, decide which embedding model to use:
  - Mistral embeddings via API
  - local embedding model
  - temporary deterministic/test embedding only for development
- For Step 3, decide whether to preserve full event-level records as metadata or create one vector per text chunk with duplicated event metadata.

### Next Step Notes
- Next step is Step 3 — Build the FAISS vector database.
- Step 3 should use `data/processed/events_clean.csv` as input.
- Step 3 should chunk or otherwise prepare `text_for_embedding` before embedding/indexing.
- Step 3 should create a reproducible vector database build script.
- Step 3 should store FAISS index files under `vectorstore/`.
- Step 3 should store metadata linked to vectors, including at minimum:
  - `event_id`
  - `title`
  - `description`
  - `first_timing_begin`
  - `last_timing_begin`
  - `city`
  - `venue_name`
  - `address`
  - `latitude`
  - `longitude`
- Step 3 should include a basic semantic search validation command or script.
- Do not implement LangChain RAG answering yet unless explicitly requested; that belongs to Step 4.

### Resume Context for AI
- Important technical facts:
  - Project: Puls-Events RAG POC for OpenClassrooms Data Engineer Project 11.
  - Step 1 is complete and locally validated.
  - Step 2 is complete and locally validated.
  - Conda environment name remains `local-ai-rag`.
  - FAISS must remain CPU-based unless the user explicitly changes this.
  - OpenAgenda agenda slug used for Step 2: `culture`.
  - OpenAgenda agenda UID resolved during fetch: `86244142`.
  - Geographic scope: Paris.
  - API-side date filter used: `timings[gte] = 2025-05-04` on 2026-05-04.
  - Raw fetch produced 1030 events.
  - Schema inspection confirmed required French title/description, location, coordinates, and timings are present for all 1030 events.
  - Preprocessing produced 1030 clean rows and 0 rejected rows.
  - Dataset tests pass locally.
  - `text_for_embedding` exists in the clean CSV and is ready for Step 3.
  - Mistral API key is still not required until embedding/LLM integration.
- Things not to change without confirmation:
  - Do not rename Conda environment `local-ai-rag`.
  - Do not switch from `faiss-cpu` to GPU FAISS.
  - Do not change geographic scope from Paris.
  - Do not add fallback data sources for OpenAgenda.
  - Do not move primary one-year filtering from the API call into preprocessing.
  - Do not implement LangChain RAG answering during Step 3 unless the user explicitly requests it.
  - Do not add or update intermediate project setup documentation unless the user explicitly asks; the user wants that generated at the end of the project.
- Later change notes affecting this step:
  - None.

