"""
Core RAG chain for the Puls-Events RAG POC.

This module loads the locally generated FAISS vectorstore, retrieves relevant
OpenAgenda event chunks, and generates grounded answers with a Mistral chat model.

Public entry point:
    answer_question(question: str, top_k: int = 5) -> dict
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VECTORSTORE_PATH = PROJECT_ROOT / "vectorstore" / "faiss_index"

DEFAULT_EMBEDDING_MODEL = "mistral-embed"
DEFAULT_CHAT_MODEL = "mistral-small-latest"
DEFAULT_TOP_K = 5

MAX_CONTEXT_CHARS_PER_CHUNK = 2500
CHUNK_PREVIEW_CHARS = 350


@dataclass(frozen=True)
class RagConfig:
    """Configuration for the Puls-Events RAG chain."""

    project_root: Path
    vectorstore_path: Path
    embedding_model: str
    chat_model: str
    default_top_k: int


def load_config() -> RagConfig:
    """
    Load RAG configuration from `.env` and environment variables.

    Returns:
        RagConfig: normalized project configuration.

    Raises:
        ValueError: if RAG_TOP_K is not a positive integer.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    top_k_raw = os.getenv("RAG_TOP_K", str(DEFAULT_TOP_K)).strip()
    try:
        default_top_k = int(top_k_raw)
    except ValueError as exc:
        raise ValueError(f"RAG_TOP_K must be an integer, got: {top_k_raw!r}") from exc

    if default_top_k < 1:
        raise ValueError(f"RAG_TOP_K must be >= 1, got: {default_top_k}")

    vectorstore_path = Path(
        os.getenv("FAISS_INDEX_PATH", str(DEFAULT_VECTORSTORE_PATH))
    ).expanduser()

    if not vectorstore_path.is_absolute():
        vectorstore_path = PROJECT_ROOT / vectorstore_path

    return RagConfig(
        project_root=PROJECT_ROOT,
        vectorstore_path=vectorstore_path,
        embedding_model=os.getenv(
            "MISTRAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        ).strip(),
        chat_model=os.getenv("MISTRAL_CHAT_MODEL", DEFAULT_CHAT_MODEL).strip(),
        default_top_k=default_top_k,
    )


def validate_environment(config: RagConfig) -> None:
    """
    Validate required runtime configuration.

    Args:
        config: RAG configuration.

    Raises:
        RuntimeError: if MISTRAL_API_KEY is missing.
        FileNotFoundError: if the FAISS vectorstore files are missing.
    """
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is missing. Add it to your local .env file before "
            "running the RAG chain."
        )

    required_files = [
        config.vectorstore_path / "index.faiss",
        config.vectorstore_path / "index.pkl",
    ]

    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "FAISS vectorstore files are missing. Expected locally generated files:\n"
            f"{missing}\n"
            "Run the existing vectorstore build script only if the index is genuinely "
            "missing or inconsistent."
        )


def get_embeddings(config: RagConfig | None = None) -> MistralAIEmbeddings:
    """
    Instantiate the Mistral embedding client.

    The API key is intentionally read from the environment by langchain-mistralai.
    The environment is validated separately before use.

    Args:
        config: optional RAG configuration.

    Returns:
        MistralAIEmbeddings: embedding client.
    """
    config = config or load_config()
    return MistralAIEmbeddings(model=config.embedding_model)


def get_chat_model(config: RagConfig | None = None) -> ChatMistralAI:
    """
    Instantiate the Mistral chat model.

    Args:
        config: optional RAG configuration.

    Returns:
        ChatMistralAI: chat model client.
    """
    config = config or load_config()
    return ChatMistralAI(
        model_name=config.chat_model,
        temperature=0,
    )


@lru_cache(maxsize=1)
def load_vectorstore() -> FAISS:
    """
    Load the locally generated LangChain FAISS vectorstore.

    Note:
        `allow_dangerous_deserialization=True` is required by LangChain because
        the local docstore metadata is stored in a pickle file. This must only be
        used for this project's own locally generated vectorstore.

    Returns:
        FAISS: loaded vectorstore.
    """
    config = load_config()
    validate_environment(config)

    embeddings = get_embeddings(config)

    return FAISS.load_local(
        folder_path=str(config.vectorstore_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def _validate_question(question: str) -> str:
    """
    Validate and normalize the user question.

    Args:
        question: raw question.

    Returns:
        str: stripped question.

    Raises:
        TypeError: if question is not a string.
        ValueError: if question is empty.
    """
    if not isinstance(question, str):
        raise TypeError(f"question must be a string, got: {type(question).__name__}")

    normalized = question.strip()
    if not normalized:
        raise ValueError("question must not be empty")

    return normalized


def _validate_top_k(top_k: int | None, config: RagConfig) -> int:
    """
    Validate and normalize top_k.

    Args:
        top_k: requested number of retrieved chunks.
        config: RAG configuration.

    Returns:
        int: validated top_k.

    Raises:
        TypeError: if top_k is not an integer.
        ValueError: if top_k is lower than 1.
    """
    if top_k is None:
        return config.default_top_k

    if not isinstance(top_k, int):
        raise TypeError(f"top_k must be an integer, got: {type(top_k).__name__}")

    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got: {top_k}")

    return top_k


def retrieve_relevant_documents(
    question: str,
    top_k: int | None = None,
) -> list[tuple[Document, float | None]]:
    """
    Retrieve relevant FAISS chunks for a question.

    This function is intentionally public-ish because it is useful for later
    tests, manual inspection, and evaluation dataset construction.

    Args:
        question: user question.
        top_k: number of chunks to retrieve.

    Returns:
        list[tuple[Document, float | None]]: retrieved documents with FAISS score
        when available.
    """
    config = load_config()
    validate_environment(config)

    normalized_question = _validate_question(question)
    normalized_top_k = _validate_top_k(top_k, config)

    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search_with_score(
        query=normalized_question,
        k=normalized_top_k,
    )

    normalized_results: list[tuple[Document, float | None]] = []
    for document, score in results:
        normalized_results.append((document, _safe_float(score)))

    return normalized_results


def _safe_float(value: Any) -> float | None:
    """
    Convert numeric-like values to JSON-serializable floats.

    Args:
        value: score-like value.

    Returns:
        float | None: converted score, or None if unavailable.
    """
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_value(value: Any) -> str:
    """
    Convert metadata values into clean display strings.

    Args:
        value: raw metadata value.

    Returns:
        str: clean string, or empty string.
    """
    if value is None:
        return ""

    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""

    return text


def _metadata_value(metadata: dict[str, Any], *keys: str) -> str:
    """
    Read the first non-empty metadata value from possible keys.

    Args:
        metadata: document metadata.
        *keys: candidate metadata keys.

    Returns:
        str: first non-empty value, or empty string.
    """
    for key in keys:
        value = _clean_value(metadata.get(key))
        if value:
            return value

    return ""


def _truncate(text: str, max_chars: int) -> str:
    """
    Truncate text safely for prompts or previews.

    Args:
        text: raw text.
        max_chars: maximum output length.

    Returns:
        str: truncated text.
    """
    normalized = " ".join((text or "").split())
    if len(normalized) <= max_chars:
        return normalized

    return normalized[: max_chars - 3].rstrip() + "..."


def format_sources(
    retrieved_documents: list[tuple[Document, float | None]],
) -> list[dict[str, Any]]:
    """
    Format retrieved documents into source metadata dictionaries.

    Args:
        retrieved_documents: retrieved documents with scores.

    Returns:
        list[dict[str, Any]]: source metadata suitable for CLI display,
        tests, and evaluation traces.
    """
    sources: list[dict[str, Any]] = []

    for rank, (document, score) in enumerate(retrieved_documents, start=1):
        metadata = dict(document.metadata or {})

        first_timing = _metadata_value(
            metadata,
            "first_timing_begin",
            "firstTimingBegin",
            "first_timing",
            "start_date",
        )
        last_timing = _metadata_value(
            metadata,
            "last_timing_begin",
            "lastTimingBegin",
            "last_timing",
            "end_date",
        )

        sources.append(
            {
                "rank": rank,
                "score": score,
                "event_id": _metadata_value(metadata, "event_id", "uid", "id"),
                "title": _metadata_value(metadata, "title", "title_fr", "name"),
                "venue_name": _metadata_value(
                    metadata,
                    "venue_name",
                    "location_name",
                    "place_name",
                    "location",
                ),
                "address": _metadata_value(metadata, "address", "location_address"),
                "city": _metadata_value(metadata, "city", "location_city"),
                "first_timing_begin": first_timing,
                "last_timing_begin": last_timing,
                "date_range": _format_date_range(first_timing, last_timing),
                "conditions": _metadata_value(metadata, "conditions"),
                "registration_url": _metadata_value(
                    metadata,
                    "registration_url",
                    "registration",
                    "registration_link",
                ),
                "latitude": _metadata_value(metadata, "latitude", "lat"),
                "longitude": _metadata_value(metadata, "longitude", "lon", "lng"),
                "origin_agenda": _metadata_value(
                    metadata,
                    "origin_agenda",
                    "agenda",
                    "agenda_slug",
                ),
                "chunk_id": _metadata_value(metadata, "chunk_id"),
                "chunk_index": metadata.get("chunk_index"),
                "chunk_text_length": metadata.get("chunk_text_length"),
                "chunk_preview": _truncate(
                    document.page_content,
                    CHUNK_PREVIEW_CHARS,
                ),
            }
        )

    return sources


def _format_date_range(first_timing: str, last_timing: str) -> str:
    """
    Format a compact date range from metadata timing fields.

    Args:
        first_timing: first timing string.
        last_timing: last timing string.

    Returns:
        str: readable date range.
    """
    if first_timing and last_timing and first_timing != last_timing:
        return f"{first_timing} → {last_timing}"

    return first_timing or last_timing


def format_context(
    retrieved_documents: list[tuple[Document, float | None]],
) -> str:
    """
    Format retrieved documents into a strict prompt context.

    Args:
        retrieved_documents: retrieved documents with scores.

    Returns:
        str: numbered source context for the prompt.
    """
    if not retrieved_documents:
        return ""

    sources = format_sources(retrieved_documents)
    blocks: list[str] = []

    for source, (document, _score) in zip(sources, retrieved_documents, strict=True):
        title = source.get("title") or "Non renseigné"
        venue_name = source.get("venue_name") or "Non renseigné"
        address = source.get("address") or "Non renseignée"
        city = source.get("city") or "Non renseignée"
        date_range = source.get("date_range") or "Non renseignées"
        conditions = source.get("conditions") or "Non renseignées"
        registration_url = source.get("registration_url") or "Non renseignée"
        event_id = source.get("event_id") or "Non renseigné"

        content = _truncate(document.page_content, MAX_CONTEXT_CHARS_PER_CHUNK)

        blocks.append(
            "\n".join(
                [
                    f"[Source {source['rank']}]",
                    f"event_id: {event_id}",
                    f"Titre: {title}",
                    f"Lieu: {venue_name}",
                    f"Adresse: {address}",
                    f"Ville: {city}",
                    f"Dates: {date_range}",
                    f"Conditions / prix: {conditions}",
                    f"Inscription: {registration_url}",
                    "Contenu:",
                    content,
                ]
            )
        )

    return "\n\n".join(blocks)


def build_prompt() -> ChatPromptTemplate:
    """
    Build the strict grounding prompt for event recommendations.

    Returns:
        ChatPromptTemplate: LangChain chat prompt.
    """
    system_message = """
Tu es le chatbot RAG de Puls-Events, spécialisé dans les événements culturels à Paris.

Règles obligatoires :
- Réponds uniquement à partir du CONTEXTE fourni.
- N'invente jamais de titre d'événement, date, lieu, adresse, prix, condition d'accès, lien d'inscription ou information pratique.
- Ne dis pas qu'un événement existe si son titre ou ses informations ne sont pas présents dans le CONTEXTE.
- Si le CONTEXTE ne contient pas assez d'information pour répondre correctement, dis-le clairement.
- Réponds en français par défaut, sauf si la question demande explicitement une autre langue.
- Quand tu recommandes des événements, cite autant que possible : titre, dates, lieu, adresse/ville, conditions ou inscription si disponibles.
- Si plusieurs sources parlent du même événement, consolide l'information sans dupliquer inutilement.
- Si la question demande un nombre précis de recommandations, respecte ce nombre seulement si le CONTEXTE contient assez d'événements pertinents.
- Ne mentionne pas les numéros de sources dans la réponse finale, sauf si c'est utile pour clarifier.
""".strip()

    human_message = """
QUESTION UTILISATEUR :
{question}

CONTEXTE RETROUVÉ :
{context}

RÉPONSE :
""".strip()

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", human_message),
        ]
    )


def _extract_message_text(response: Any) -> str:
    """
    Extract plain text from a LangChain chat model response.

    Args:
        response: raw chat model response.

    Returns:
        str: response text.
    """
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))

        return "\n".join(part.strip() for part in parts if part.strip()).strip()

    return str(content).strip()


def answer_question(question: str, top_k: int | None = None) -> dict[str, Any]:
    """
    Answer a user question using retrieved OpenAgenda event context.

    Args:
        question: user question.
        top_k: number of FAISS chunks to retrieve. If None, uses RAG_TOP_K.

    Returns:
        dict[str, Any]: structured RAG answer with contexts and sources.
    """
    config = load_config()
    validate_environment(config)

    normalized_question = _validate_question(question)
    normalized_top_k = _validate_top_k(top_k, config)

    retrieved_documents = retrieve_relevant_documents(
        question=normalized_question,
        top_k=normalized_top_k,
    )

    contexts = [document.page_content for document, _score in retrieved_documents]
    sources = format_sources(retrieved_documents)

    if not retrieved_documents:
        return {
            "question": normalized_question,
            "answer": (
                "Je n'ai pas trouvé d'événements pertinents dans la base vectorielle "
                "pour répondre à cette question sans inventer d'information."
            ),
            "contexts": contexts,
            "sources": sources,
            "model": config.chat_model,
            "embedding_model": config.embedding_model,
            "top_k": normalized_top_k,
        }

    context_text = format_context(retrieved_documents)
    prompt = build_prompt()
    chat_model = get_chat_model(config)

    chain = prompt | chat_model
    response = chain.invoke(
        {
            "question": normalized_question,
            "context": context_text,
        }
    )

    answer = _extract_message_text(response)

    return {
        "question": normalized_question,
        "answer": answer,
        "contexts": contexts,
        "sources": sources,
        "model": config.chat_model,
        "embedding_model": config.embedding_model,
        "top_k": normalized_top_k,
    }


__all__ = [
    "RagConfig",
    "answer_question",
    "build_prompt",
    "format_context",
    "format_sources",
    "get_chat_model",
    "get_embeddings",
    "load_config",
    "load_vectorstore",
    "retrieve_relevant_documents",
    "validate_environment",
]