"""
API-Schlüssel Management mit Fallback-Logik
"""

import time
from typing import Dict, Any
import logging
from config.settings import Settings, APIConfig


class APIManager:
    """Verwaltet API-Schlüssel mit Fallback und Rate Limiting"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._rate_limits: Dict[str, Dict[str, Any]] = {}

    def get_api_config(self, api_type: str) -> APIConfig:
        """Holt API-Konfiguration mit Fallback-Logik"""
        config = Settings.get_api_config(api_type)

        # Prüfe Primary Key
        if self._is_key_available(api_type, "primary"):
            return config

        # Fallback auf Secondary Key
        if config.fallback_key and self._is_key_available(api_type, "fallback"):
            self.logger.info(f"Switching to fallback API key for {api_type}")
            return APIConfig(
                primary_key=config.fallback_key,
                fallback_key=None,
                base_url=config.base_url,
                timeout=config.timeout
            )

        raise Exception(f"No available API keys for {api_type}")

    def _is_key_available(self, api_type: str, key_type: str) -> bool:
        """Prüft, ob ein API-Schlüssel verfügbar ist (Rate Limiting)"""
        key_id = f"{api_type}_{key_type}"

        if key_id not in self._rate_limits:
            self._rate_limits[key_id] = {
                "requests": 0,
                "reset_time": time.time() + 60,  # Reset nach 1 Minute
                "max_requests": Settings.MODELS.get(api_type, Settings.MODELS["vader"]).rate_limit
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

    def wait_for_rate_limit(self, api_type: str) -> None:
        """Wartet bis Rate Limit zurückgesetzt wird"""
        key_id = f"{api_type}_primary"
        if key_id in self._rate_limits:
            wait_time = self._rate_limits[key_id]["reset_time"] - time.time()
            if wait_time > 0:
                self.logger.info(f"Rate limit reached for {api_type}, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
