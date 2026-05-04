"""Create and validate a reusable Conda environment for local AI/RAG work.

Default environment name: local-ai-rag

Run from the project root with Miniconda/Anaconda available:

    python scripts/setup_environment.py

Then activate it:

    conda activate local-ai-rag

The script:
- verifies that Conda is available
- creates the Conda environment if missing
- installs requirements.txt into that environment with pip
- registers a Jupyter kernel
- runs scripts/check_environment.py inside the environment
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_NAME = "local-ai-rag"
DEFAULT_PYTHON = "3.11"


def run_command(command: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    print(f"\n$ {' '.join(command)}")
    completed = subprocess.run(command, cwd=cwd, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def capture_command(command: list[str], *, cwd: Path = PROJECT_ROOT) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{completed.returncode}: {' '.join(command)}\n{completed.stderr}"
        )
    return completed.stdout


def ensure_conda_available() -> str:
    conda_executable = shutil.which("conda")
    if conda_executable is None:
        raise RuntimeError(
            "Conda executable was not found in PATH. Open an Anaconda Prompt/Miniconda shell "
            "or initialize Conda before running this script."
        )
    return conda_executable


def conda_env_exists(env_name: str) -> bool:
    output = capture_command(["conda", "env", "list"])
    for line in output.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if parts and parts[0] == env_name:
            return True
    return False


def create_env(env_name: str, python_version: str, force_recreate: bool) -> None:
    if conda_env_exists(env_name):
        if not force_recreate:
            print(f"\n[OK] Conda environment already exists: {env_name}")
            return
        run_command(["conda", "env", "remove", "--name", env_name, "--yes"])

    run_command([
        "conda",
        "create",
        "--name",
        env_name,
        f"python={python_version}",
        "pip",
        "--yes",
    ])


def install_requirements(env_name: str) -> None:
    requirements = PROJECT_ROOT / "requirements.txt"
    if not requirements.exists():
        raise FileNotFoundError(f"Missing dependency file: {requirements}")

    run_command([
        "conda",
        "run",
        "--name",
        env_name,
        "python",
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
    ])

    run_command([
        "conda",
        "run",
        "--name",
        env_name,
        "python",
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements),
    ])


def register_kernel(env_name: str) -> None:
    run_command([
        "conda",
        "run",
        "--name",
        env_name,
        "python",
        "-m",
        "ipykernel",
        "install",
        "--user",
        "--name",
        env_name,
        "--display-name",
        f"Python ({env_name})",
    ])


def run_validation(env_name: str) -> None:
    run_command([
        "conda",
        "run",
        "--name",
        env_name,
        "python",
        str(PROJECT_ROOT / "scripts" / "check_environment.py"),
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and validate a reusable Conda environment for local AI/RAG work."
    )
    parser.add_argument(
        "--env-name",
        default=DEFAULT_ENV_NAME,
        help=f"Conda environment name. Default: {DEFAULT_ENV_NAME}",
    )
    parser.add_argument(
        "--python",
        default=DEFAULT_PYTHON,
        help=f"Python version to use when creating the environment. Default: {DEFAULT_PYTHON}",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Delete and recreate the environment if it already exists.",
    )
    parser.add_argument(
        "--skip-kernel",
        action="store_true",
        help="Skip Jupyter kernel registration.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        print(f"Project root : {PROJECT_ROOT}")
        print(f"Env name     : {args.env_name}")
        print(f"Python       : {args.python}")

        conda_executable = ensure_conda_available()
        print(f"Conda        : {conda_executable}")

        create_env(args.env_name, args.python, args.force_recreate)
        install_requirements(args.env_name)

        if not args.skip_kernel:
            register_kernel(args.env_name)

        run_validation(args.env_name)

        print("\n[OK] Setup completed successfully.")
        print(f"Activate the environment with: conda activate {args.env_name}")
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level CLI diagnostic
        print(f"\n[FAIL] Setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
