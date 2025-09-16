"""Base analyzer abstraction."""

from abc import ABC, abstractmethod
from typing import Any


class BaseAnalyzer(ABC):
    """Abstract base for sentiment analyzers."""

    @abstractmethod
    def analyze(self, text: str) -> Any:
        """Analyze text and return a result."""

        raise NotImplementedError
