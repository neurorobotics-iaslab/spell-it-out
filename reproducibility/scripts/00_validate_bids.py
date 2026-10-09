"""Run the official BIDS Validator on a local dataset.

The wrapper prefers an installed ``bids-validator`` executable. If it is not
available, it uses the official Deno package, pinned by default for
reproducibility. Install Deno from https://deno.com/ or with conda-forge.

Examples
--------
python scripts/00_validate_bids.py
python scripts/00_validate_bids.py data/spellitout
python scripts/00_validate_bids.py -- --verbose
python scripts/00_validate_bids.py data/spellitout -- --verbose
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from spellitout import SpellItOutDataset

DEFAULT_VALIDATOR_VERSION = "3.0.2"
REQUIRED_DESCRIPTION_FIELDS = ("Name", "BIDSVersion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=None,
        help=f"BIDS dataset root (default: inferred from SpellItOutDataset).",
    )
    parser.add_argument(
        "--validator-command",
        type=Path,
        default=None,
        help="Explicit path to an installed bids-validator executable.",
    )
    parser.add_argument(
        "--validator-version",
        default=DEFAULT_VALIDATOR_VERSION,
        help=(
            "Official JSR validator version used through Deno "
            f"(default: {DEFAULT_VALIDATOR_VERSION})."
        ),
    )
    parser.add_argument(
        "validator_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to bids-validator; place them after --.",
    )
    return parser.parse_args()


def preflight_dataset(dataset: Path) -> None:
    """Fail early on root-level problems with clearer local messages."""
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")

    description_path = dataset / "dataset_description.json"
    if not description_path.is_file():
        raise FileNotFoundError(
            f"Required BIDS file is missing: {description_path}"
        )
    try:
        description = json.loads(description_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {description_path}: {error}") from error
    missing = [field for field in REQUIRED_DESCRIPTION_FIELDS if not description.get(field)]
    if missing:
        raise ValueError(
            f"{description_path} is missing required fields: {', '.join(missing)}"
        )

    split_task_directories = [
        name for name in ("overt", "covert") if (dataset / name).is_dir()
    ]
    if split_task_directories:
        joined = ", ".join(split_task_directories)
        raise ValueError(
            "Non-BIDS task directories remain at the dataset root: "
            f"{joined}. Move task files below the common sub-*/ses-*/eeg hierarchy."
        )


def validator_command(args: argparse.Namespace, dataset: Path) -> list[str]:
    """Resolve the official installed or Deno-hosted validator command."""
    if args.validator_command is not None:
        executable = args.validator_command.expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"Validator executable not found: {executable}")
        return [str(executable), str(dataset)]

    installed = shutil.which("bids-validator")
    if installed:
        return [installed, str(dataset)]

    deno = shutil.which("deno")
    if deno is None and os.name == "nt":
        standard_deno = Path.home() / ".deno" / "bin" / "deno.exe"
        if standard_deno.is_file():
            deno = str(standard_deno)
    if deno:
        package = f"jsr:@bids/validator@{args.validator_version}"
        return [deno, "run", "-ERWN", package, str(dataset)]

    raise RuntimeError(
        "Neither bids-validator nor Deno is installed. Install Deno "
        "(https://docs.deno.com/runtime/getting_started/installation/) or provide "
        "--validator-command PATH. The official validator can then be run as "
        "`deno run -ERWN jsr:@bids/validator <dataset>`."
    )


def main(args: argparse.Namespace) -> int:
    if args.dataset is None:
        try:
            dataset = SpellItOutDataset().root
        except RuntimeError as error:
            raise RuntimeError(
                "Could not infer dataset root from SpellItOutDataset. "
                "Please provide the dataset path explicitly."
            ) from error
    else:
        dataset = args.dataset.expanduser().resolve()
    preflight_dataset(dataset)
    command = validator_command(args, dataset)
    extra = list(args.validator_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    command.extend(extra)
    print("Running:", subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main(parse_args()))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
