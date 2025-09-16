"""API key management with fallback and rate limiting helpers."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from config.settings import APIConfig, Settings


class APIManager:
    """Verwaltet API-Schlüssel mit Fallback und Rate Limiting"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._rate_limits: Dict[str, Dict[str, Any]] = {}
        self._api_rate_limits = self._build_api_rate_limits()
        self._default_rate_limit = max(
            (model.rate_limit for model in Settings.MODELS.values()), default=60
        )

    def get_api_config(self, api_type: str) -> APIConfig:
        """Holt API-Konfiguration mit Fallback-Logik"""
        config = Settings.get_api_config(api_type)

        # Prüfe Primary Key
        if config.primary_key:
            if self._is_key_available(api_type, "primary"):
                return config

        # Fallback auf Secondary Key
        if config.fallback_key:
            if self._is_key_available(api_type, "fallback"):
                self.logger.info("Switching to fallback API key for %s", api_type)
                return APIConfig(
                    primary_key=config.fallback_key,
                    fallback_key=None,
                    base_url=config.base_url,
                    timeout=config.timeout,
                )

        raise RuntimeError(f"No available API keys for {api_type}")

    def _is_key_available(self, api_type: str, key_type: str) -> bool:
        """Prüft, ob ein API-Schlüssel verfügbar ist (Rate Limiting)"""
        key_id = f"{api_type}_{key_type}"

        if key_id not in self._rate_limits:
            self._rate_limits[key_id] = {
                "requests": 0,
                "reset_time": time.time() + 60,  # Reset nach 1 Minute
                "max_requests": self._get_rate_limit_for_api(api_type),
            }

        rate_limit = self._rate_limits[key_id]
        current_time = time.time()

        # Reset Rate Limit wenn Zeit abgelaufen
        if current_time >= rate_limit["reset_time"]:
            rate_limit["requests"] = 0
            rate_limit["reset_time"] = current_time + 60

        # Prüfe ob unter Limit
        if rate_limit["requests"] < rate_limit["max_requests"]:
            rate_limit["requests"] += 1
            return True

        return False

    def wait_for_rate_limit(self, api_type: str, key_type: str = "primary") -> None:
        """Wartet bis Rate Limit zurückgesetzt wird"""
        key_id = f"{api_type}_{key_type}"
        if key_id in self._rate_limits:
            wait_time = self._rate_limits[key_id]["reset_time"] - time.time()
            if wait_time > 0:
                self.logger.info(
                    "Rate limit reached for %s (%s), waiting %.1f seconds",
                    api_type,
                    key_type,
                    wait_time,
                )
                time.sleep(wait_time)

    def _build_api_rate_limits(self) -> Dict[str, int]:
        """Create a mapping of API types to their configured rate limits."""
        limits: Dict[str, int] = {}
        for model in Settings.MODELS.values():
            limits[model.api_type] = max(limits.get(model.api_type, 0), model.rate_limit)
        return limits

    def _get_rate_limit_for_api(self, api_type: str) -> int:
        """Return the configured rate limit for ``api_type``."""
        return self._api_rate_limits.get(api_type, self._default_rate_limit)
