"""Text preprocessing utilities."""

import re


def clean_text(text: str) -> str:
    """Lowercase and remove non-word characters."""

    text = text.lower()
    return re.sub(r"[^\w\s]", "", text)
