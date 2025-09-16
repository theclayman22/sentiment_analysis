"""Utilities for loading and validating text data from different file formats."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

try:  # pragma: no cover - optional dependency
    import chardet
except ImportError:  # pragma: no cover - fallback when chardet is unavailable
    chardet = None  # type: ignore


class DataLoader:
    """Load and process text inputs coming from user supplied files."""

    #: File extensions that are supported by :meth:`load_from_file`.
    supported_formats: Tuple[str, ...] = (".csv", ".txt", ".xlsx")

    def load_from_file(self, uploaded_file) -> Tuple[List[str], Optional[str]]:
        """Load texts from a user uploaded file.

        Parameters
        ----------
        uploaded_file:
            A file-like object as provided by Streamlit's ``file_uploader`` or a
            similar API. The object must expose a ``name`` attribute and the
            standard ``read``/``seek`` methods.

        Returns
        -------
        Tuple[List[str], Optional[str]]
            A two-element tuple containing the extracted texts and an optional
            error message. When an unrecoverable error occurs the list of texts
            will be empty and the error message will describe the failure.
        """

        if uploaded_file is None:
            return [], "Keine Datei übergeben"

        try:
            extension = self._extract_extension(uploaded_file)
            loader_map = {
                ".csv": self._load_csv,
                ".txt": self._load_txt,
                ".xlsx": self._load_excel,
            }

            if extension not in loader_map:
                return [], f"Nicht unterstütztes Dateiformat: {extension or '?'}"

            texts, error = loader_map[extension](uploaded_file)
            return texts, error
        except Exception as exc:  # pragma: no cover - defensive programming
            return [], f"Fehler beim Laden der Datei: {exc}"

    def _load_csv(self, uploaded_file) -> Tuple[List[str], Optional[str]]:
        """Load and parse the contents of a CSV file."""

        try:
            raw_data = uploaded_file.read()
            encoding = self._detect_encoding(raw_data)
            uploaded_file.seek(0)

            df = pd.read_csv(uploaded_file, encoding=encoding)
            return self._extract_texts_from_dataframe(df, "CSV")
        except pd.errors.EmptyDataError:
            return [], "CSV-Datei ist leer"
        except Exception as exc:
            return [], f"Fehler beim CSV-Import: {exc}"
        finally:
            uploaded_file.seek(0)

    def _load_txt(self, uploaded_file) -> Tuple[List[str], Optional[str]]:
        """Load texts from a plain text file."""

        try:
            raw_data = uploaded_file.read()
            encoding = self._detect_encoding(raw_data)
            text_content = raw_data.decode(encoding)
            lines = [line.strip() for line in text_content.splitlines() if line.strip()]
            return lines, None
        except UnicodeDecodeError as exc:
            return [], f"Fehler beim TXT-Import: {exc}"
        except Exception as exc:
            return [], f"Fehler beim TXT-Import: {exc}"
        finally:
            uploaded_file.seek(0)

    def _load_excel(self, uploaded_file) -> Tuple[List[str], Optional[str]]:
        """Load texts from an Excel workbook."""

        try:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
            return self._extract_texts_from_dataframe(df, "Excel")
        except FileNotFoundError as exc:
            return [], f"Fehler beim Excel-Import: {exc}"
        except Exception as exc:
            return [], f"Fehler beim Excel-Import: {exc}"
        finally:
            uploaded_file.seek(0)

    def _extract_texts_from_dataframe(self, df: pd.DataFrame, label: str) -> Tuple[List[str], Optional[str]]:
        """Extract the most plausible text column from a dataframe."""

        if df.empty:
            return [], f"Keine Daten in {label}-Datei gefunden"

        text_column = self._find_text_column(df)
        if not text_column:
            return [], f"Keine Text-Spalte in {label} gefunden"

        texts = (
            df[text_column]
            .dropna()
            .astype(str)
            .map(str.strip)
            .loc[lambda s: s.str.len() > 0]
            .tolist()
        )
        return texts, None

    def _find_text_column(self, df: pd.DataFrame) -> Optional[str]:
        """Return the most likely text column in ``df``."""

        if df.empty:
            return None

        text_indicators = [
            "text",
            "content",
            "message",
            "comment",
            "review",
            "description",
        ]

        for column in df.columns:
            if column.lower() in text_indicators:
                return column

        for column in df.columns:
            lower = column.lower()
            if any(indicator in lower for indicator in text_indicators):
                return column

        string_like = df.select_dtypes(include=["object", "string"])
        if not string_like.empty:
            avg_lengths = (
                string_like.astype(str)
                .apply(lambda col: col.str.len().mean())
                .dropna()
            )
            if not avg_lengths.empty:
                return avg_lengths.idxmax()

        return None

    def validate_texts(self, texts: List[str]) -> Tuple[List[str], List[str]]:
        """Validate and normalise user provided texts."""

        valid_texts: List[str] = []
        errors: List[str] = []

        for index, text in enumerate(texts, start=1):
            if not text or len(text.strip()) < 3:
                errors.append(f"Text {index}: Zu kurz (< 3 Zeichen)")
                continue

            if len(text) > 10_000:
                errors.append(f"Text {index}: Zu lang (> 10.000 Zeichen)")
                continue

            valid_texts.append(text.strip())

        return valid_texts, errors

    def _detect_encoding(self, data: bytes) -> str:
        """Detect the most appropriate text encoding for ``data``."""

        if not data:
            return "utf-8"

        if chardet is None:
            return "utf-8"

        detection = chardet.detect(data)
        encoding = detection.get("encoding") if detection else None
        return encoding or "utf-8"

    def _extract_extension(self, uploaded_file) -> str:
        """Extract the lowercase file extension including the leading dot."""

        name = getattr(uploaded_file, "name", "") or ""
        return Path(name).suffix.lower()


class _NamedBytesIO(io.BytesIO):
    """Helper to emulate Streamlit's uploaded file objects in tests."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def load_texts(path: Path) -> List[str]:
    """Load texts from ``path`` using :class:`DataLoader`.

    This helper retains backwards compatibility with earlier versions of the
    project that exposed a simple ``load_texts`` function.
    """

    loader = DataLoader()
    with path.open("rb") as file_handle:
        buffer = _NamedBytesIO(file_handle.read(), path.name)
    texts, error = loader.load_from_file(buffer)
    if error:
        raise ValueError(error)
    return texts
