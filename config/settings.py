"""
Zentrale Konfiguration für das Sentiment Analysis Toolkit
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import streamlit as st


@dataclass
class ModelConfig:
    """Konfiguration für einzelne Modelle"""
    name: str
    display_name: str
    api_type: str
    supports_valence: bool = True
    supports_ekman: bool = True
    supports_emotion_arc: bool = True
    max_tokens: int = 4000
    rate_limit: int = 60  # Requests per minute


@dataclass
class APIConfig:
    """API Konfiguration"""
    primary_key: str
    fallback_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30


class Settings:
    """Zentrale Einstellungen"""

    # Verfügbare Modelle
    MODELS = {
        "gpt-5-nano": ModelConfig(
            name="gpt-5-nano",
            display_name="OpenAI GPT-5 Nano",
            api_type="openai_reasoning",
            max_tokens=8000,
            rate_limit=100
        ),
        "deepseek-chat": ModelConfig(
            name="deepseek-chat",
            display_name="DeepSeek Chat",
            api_type="deepseek",
            max_tokens=4000,
            rate_limit=60
        ),
        "facebook/bart-large": ModelConfig(
            name="facebook/bart-large",
            display_name="BART Large (HuggingFace)",
            api_type="huggingface",
            supports_emotion_arc=False,
            max_tokens=1024,
            rate_limit=100
        ),
        "FacebookAI/roberta-base": ModelConfig(
            name="FacebookAI/roberta-base",
            display_name="RoBERTa Base (HuggingFace)",
            api_type="huggingface",
            supports_emotion_arc=False,
            max_tokens=512,
            rate_limit=100
        ),
        "siebert/sentiment-roberta-large-english": ModelConfig(
            name="siebert/sentiment-roberta-large-english",
            display_name="SiEBERT (HuggingFace)",
            api_type="huggingface",
            supports_ekman=False,
            supports_emotion_arc=True,
            max_tokens=512,
            rate_limit=100
        ),
        "vader": ModelConfig(
            name="vader",
            display_name="VADER",
            api_type="vader",
            supports_ekman=False,
            supports_emotion_arc=True,
            max_tokens=10000,
            rate_limit=1000
        )
    }

    # API Konfigurationen
    @staticmethod
    def get_api_config(api_type: str) -> APIConfig:
        """Holt API-Konfiguration aus Streamlit Secrets"""
        if api_type == "openai_reasoning":
            return APIConfig(
                primary_key=st.secrets.get("OPENAI_API_KEY_01", ""),
                fallback_key=st.secrets.get("OPENAI_API_KEY_02", ""),
                base_url="https://api.openai.com/v1"
            )
        elif api_type == "deepseek":
            return APIConfig(
                primary_key=st.secrets.get("DEEPSEEK_API_KEY_01", ""),
                fallback_key=st.secrets.get("DEEPSEEK_API_KEY_02", ""),
                base_url="https://api.deepseek.com"
            )
        elif api_type == "huggingface":
            return APIConfig(
                primary_key=st.secrets.get("HUGGING_FACE_API_KEY", ""),
                fallback_key=st.secrets.get("HUGGING_FACE_API_KEY_02", ""),
                base_url="https://api-inference.huggingface.co"
            )
        else:
            return APIConfig(primary_key="")

    # Analyse-Typen
    ANALYSIS_TYPES = {
        "valence": "Valence (Positiv/Negativ/Neutral)",
        "ekman": "Ekman-Emotionen",
        "emotion_arc": "Emotion Arc"
    }

    # Batch-Verarbeitung
    MAX_BATCH_SIZE = 100
    DEFAULT_BATCH_SIZE = 20

    # Export-Formate
    EXPORT_FORMATS = ["CSV", "Excel", "JSON"]

    # UI-Einstellungen
    DEFAULT_LANGUAGE = "DE"
    SUPPORTED_LANGUAGES = ["DE", "EN"]

    # Performance
    MAX_WORKERS = 5
    REQUEST_TIMEOUT = 30
    RETRY_ATTEMPTS = 3
