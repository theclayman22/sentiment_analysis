"""Export analysis results to CSV."""

from pathlib import Path
from typing import Iterable
import csv


def export_csv(path: Path, rows: Iterable[dict]) -> None:
    """Write iterable of dictionaries to CSV."""

    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
