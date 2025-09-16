"""Application settings."""

from dataclasses import dataclass


@dataclass
class Settings:
    """Basic configuration for the app."""

    app_name: str = "Sentiment Analysis Toolkit"
    version: str = "0.1.0"
    supported_languages: tuple[str, ...] = ("en", "de")


def get_settings() -> Settings:
    """Return application settings."""

    return Settings()
