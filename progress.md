# Project Progress

## Metadata
- Project: Puls-Events RAG POC — OpenClassrooms Data Engineer Project 11
- Current step: Step 1 — Prepare the development environment
- Status: Completed and locally validated by user
- Last updated: 2026-05-04
- Source files reviewed:
  - `Project11_PlusEvents_Exercice_text.txt`
  - `project11_pulsevents_plan.md`
  - `openclassrooms_project_agent_prompt.md`
- Related project files:
  - `README.md`
  - `requirements.txt`
  - `environment.yml`
  - `.env.example`
  - `.gitignore`
  - `scripts/setup_environment.py`
  - `scripts/check_environment.py`
  - `src/pulsevents_rag/__init__.py`

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
  - `README.md`
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
- `README.md` — setup and validation instructions for the project environment.
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
