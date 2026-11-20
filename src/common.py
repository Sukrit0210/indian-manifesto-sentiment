"""File locations are relative to this project, so commands work from any directory."""

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REGISTRY = ROOT / "data" / "sources.csv"
OUTPUTS = ROOT / "outputs"
CACHE = ROOT / ".cache"
MODELS = Path(os.environ.get("MANIFESTO_MODELS", CACHE / "models"))


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows, fields=None):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
