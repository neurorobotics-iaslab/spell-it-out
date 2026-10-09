"""Guard the separation between paper scripts, processing and analysis."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paper_code_does_not_import_core_processing_modules():
    moved = {
        "preprocessing", "review_qc", "ica", "publishing", "syllable_dataset",
        "blink_rate", "erd_ers", "plotting", "utils",
    }
    for folder in ("scripts", "processing", "analysis"):
        for path in (ROOT / folder).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    assert module not in {f"spellitout.{name}" for name in moved}, path


def test_configuration_copy_exists():
    assert (ROOT / "configs" / "syllable_dataset_default.json").is_file()
