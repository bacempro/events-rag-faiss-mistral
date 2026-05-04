"""Validate that the local AI/RAG Python environment is ready.

Run from the project root after activating the Conda environment:

    conda activate local-ai-rag
    python scripts/check_environment.py

The script checks:
- Python version
- required imports
- package versions where available
- FAISS CPU index creation/search
- optional Mistral API key presence without making a network call
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


MIN_PYTHON = (3, 10)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PackageCheck:
    pip_name: str
    import_name: str
    required: bool = True


PACKAGE_CHECKS = [
    PackageCheck("numpy", "numpy"),
    PackageCheck("pandas", "pandas"),
    PackageCheck("requests", "requests"),
    PackageCheck("python-dotenv", "dotenv"),
    PackageCheck("pydantic", "pydantic"),
    PackageCheck("langchain", "langchain"),
    PackageCheck("langchain-core", "langchain_core"),
    PackageCheck("langchain-community", "langchain_community"),
    PackageCheck("langchain-mistralai", "langchain_mistralai"),
    PackageCheck("mistralai", "mistralai"),
    PackageCheck("faiss-cpu", "faiss"),
    PackageCheck("sentence-transformers", "sentence_transformers"),
    PackageCheck("pytest", "pytest"),
    PackageCheck("ruff", "ruff"),
    PackageCheck("ipykernel", "ipykernel"),
]


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def check_python_version() -> bool:
    print_header("Python")
    current = sys.version_info[:3]
    print(f"Executable : {sys.executable}")
    print(f"Version    : {platform.python_version()}")
    print(f"Platform   : {platform.platform()}")

    if current < MIN_PYTHON:
        print(f"[FAIL] Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} is required.")
        return False

    print("[OK] Python version is compatible.")
    return True


def check_imports() -> bool:
    print_header("Package imports")
    ok = True

    for package in PACKAGE_CHECKS:
        try:
            importlib.import_module(package.import_name)
            try:
                installed_version = version(package.pip_name)
            except PackageNotFoundError:
                installed_version = "version metadata unavailable"
            print(f"[OK] {package.import_name:<28} {installed_version}")
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            ok = False if package.required else ok
            status = "FAIL" if package.required else "WARN"
            print(f"[{status}] {package.import_name:<28} {exc.__class__.__name__}: {exc}")

    return ok


def check_faiss_cpu() -> bool:
    print_header("FAISS CPU smoke test")
    try:
        import faiss
        import numpy as np

        dimension = 4
        vectors = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype="float32",
        )
        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")

        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        distances, indices = index.search(query, k=1)

        print(f"FAISS version : {getattr(faiss, '__version__', 'unknown')}")
        print(f"Index size    : {index.ntotal}")
        print(f"Nearest index : {indices[0][0]}")
        print(f"Distance      : {distances[0][0]:.4f}")

        if index.ntotal != 3 or int(indices[0][0]) != 0:
            print("[FAIL] FAISS returned an unexpected search result.")
            return False

        print("[OK] FAISS CPU index creation and search work.")
        return True
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"[FAIL] FAISS smoke test failed: {exc.__class__.__name__}: {exc}")
        return False


def check_env_file() -> bool:
    print_header("Environment variables")
    env_file = PROJECT_ROOT / ".env"
    env_example = PROJECT_ROOT / ".env.example"

    try:
        from dotenv import load_dotenv

        if env_file.exists():
            load_dotenv(env_file)
            print(f"[OK] Loaded {env_file.relative_to(PROJECT_ROOT)}")
        elif env_example.exists():
            print("[WARN] .env file not found. This is acceptable for Step 1.")
            print("       Copy .env.example to .env before calling the Mistral API in later steps.")
        else:
            print("[WARN] Neither .env nor .env.example exists.")

        api_key = os.getenv("MISTRAL_API_KEY", "")
        if api_key:
            print("[OK] MISTRAL_API_KEY is set. No API call was made.")
        else:
            print("[WARN] MISTRAL_API_KEY is not set. Required later for Mistral API calls.")
        return True
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"[FAIL] Environment variable check failed: {exc.__class__.__name__}: {exc}")
        return False


def main() -> int:
    checks = [
        check_python_version(),
        check_imports(),
        check_faiss_cpu(),
        check_env_file(),
    ]

    print_header("Summary")
    if all(checks):
        print("[OK] Environment is ready for Step 2.")
        return 0

    print("[FAIL] Environment is not ready. Review failed checks above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
