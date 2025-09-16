"""Load text data from files."""

from pathlib import Path
from typing import List


def load_texts(path: Path) -> List[str]:
    """Return lines of text from a file."""

    return path.read_text(encoding="utf-8").splitlines()
