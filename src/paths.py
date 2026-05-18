"""Project root — all relative paths in CSVs resolve from here."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_path(filepath: str | Path) -> Path:
    path = Path(filepath)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
