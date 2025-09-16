"""
Text-Vorverarbeitung und -Utilities
"""

import re
from typing import List, Optional, Tuple
import logging

try:  # pragma: no cover - Optional dependency
    import nltk
    from nltk.tokenize import sent_tokenize
except ModuleNotFoundError:  # pragma: no cover - Fallback ohne NLTK
    nltk = None  # type: ignore[assignment]

    def sent_tokenize(text: str) -> List[str]:  # type: ignore[override]
        return [segment.strip() for segment in re.split(r"[.!?]+", text) if segment.strip()]


class TextProcessor:
    """Verarbeitet und bereinigt Texte für die Analyse"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._ensure_nltk_data()

    def _ensure_nltk_data(self) -> None:
        """Stellt sicher, dass NLTK-Daten verfügbar sind"""
        if nltk is None:
            self.logger.debug("NLTK nicht verfügbar, überspringe Download")
            return
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception as exc:  # pragma: no cover - Download Fehler protokollieren
                self.logger.debug("Failed to download 'punkt': %s", exc)

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except Exception as exc:  # pragma: no cover - Download Fehler protokollieren
                self.logger.debug("Failed to download 'punkt_tab': %s", exc)

    def clean_text(self, text: str) -> str:
        """Bereinigt Text für die Analyse"""
        if not text:
            return ""

        # Entferne übermäßige Whitespaces
        text = re.sub(r'\s+', ' ', text.strip())

        # Entferne HTML-Tags
        text = re.sub(r'<[^>]+>', '', text)

        # Entferne URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Entferne E-Mail-Adressen
        text = re.sub(r'\S+@\S+', '', text)

        return text.strip()

    def split_into_sentences(self, text: str) -> List[str]:
        """Teilt Text in Sätze auf"""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        try:
            sentences = sent_tokenize(cleaned)
            return [self.clean_text(sent) for sent in sentences if sent.strip()]
        except Exception:
            # Fallback: Split by periods
            sentences = cleaned.split('.')
            return [self.clean_text(sent) for sent in sentences if sent.strip()]

    def chunk_text(self, text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
        """Teilt langen Text in Chunks mit Überlappung"""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        words = cleaned.split()

        if len(words) <= max_tokens:
            return [cleaned]

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)

            if end >= len(words):
                break

            start = max(0, end - overlap)

        return chunks

    def validate_text(
        self, text: str, min_length: int = 3, max_length: int = 10000
    ) -> Tuple[bool, Optional[str]]:
        """Validiert Text für die Analyse"""
        if not text or not text.strip():
            return False, "Text ist leer"

        cleaned_text = self.clean_text(text)

        if len(cleaned_text) < min_length:
            return False, f"Text zu kurz (minimum {min_length} Zeichen)"

        if len(cleaned_text) > max_length:
            return False, f"Text zu lang (maximum {max_length} Zeichen)"

        return True, None

    def extract_segments_for_arc(self, text: str, n_segments: int = 20) -> List[str]:
        """Extrahiert Segmente für Emotion Arc Analyse"""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        words = cleaned.split()
        total_words = len(words)

        if total_words < n_segments:
            return [cleaned]

        # Dynamische Segmentgröße mit Überlappung
        segment_size = max(20, total_words // n_segments)
        overlap = segment_size // 2

        segments = []
        start = 0

        while start < total_words and len(segments) < n_segments:
            end = min(start + segment_size, total_words)
            segment = ' '.join(words[start:end])
            segments.append(segment)
            start += max(1, segment_size - overlap)

        return segments[:n_segments]
