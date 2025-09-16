"""Handle API keys for external services."""

import os
from itertools import cycle
from typing import Iterator


class APIManager:
    """Rotate through a list of API keys."""

    def __init__(self, keys: list[str] | None = None) -> None:
        self._keys = keys or []
        self._iterator: Iterator[str] | None = None

    def load_from_env(self, env_var: str) -> None:
        """Load keys from comma separated environment variable."""

        value = os.getenv(env_var, "")
        self._keys = [k.strip() for k in value.split(",") if k.strip()]
        self._iterator = None

    def get_key(self) -> str | None:
        """Return the next API key or None if not configured."""

        if not self._keys:
            return None
        if self._iterator is None:
            self._iterator = cycle(self._keys)
        return next(self._iterator)
