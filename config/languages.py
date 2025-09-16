"""
Mehrsprachige Texte für die Benutzeroberfläche
"""

TRANSLATIONS = {
    "DE": {
        # Navigation
        "title": "🧠 Sentiment Analysis Toolkit",
        "subtitle": "Professionelle Sentiment- und Emotionsanalyse",

        # Sidebar
        "analysis_type": "Analyse-Typ",
        "model_selection": "Modell-Auswahl",
        "benchmark_mode": "Benchmark-Modus",
        "benchmark_description": "Vergleicht alle verfügbaren Modelle",
        "single_model": "Einzelnes Modell",
        "language_settings": "Sprach-Einstellungen",

        # Input
        "input_method": "Eingabe-Methode",
        "single_text": "Einzelner Text",
        "batch_upload": "Batch-Upload",
        "text_input": "Text eingeben",
        "text_placeholder": "Geben Sie hier Ihren Text ein...",
        "file_upload": "Datei hochladen",
        "file_types": "Unterstützte Formate: CSV, TXT",

        # Analysis
        "analyze_button": "Analysieren",
        "analyzing": "Analysiere...",
        "analysis_complete": "Analyse abgeschlossen",
        "text_count": "Anzahl Texte",
        "processing_time": "Verarbeitungszeit",

        # Results
        "results": "Ergebnisse",
        "valence_results": "Valence-Ergebnisse",
        "ekman_results": "Ekman-Emotionen",
        "emotion_arc_results": "Emotion Arc",
        "benchmark_results": "Benchmark-Ergebnisse",

        # Export
        "export": "Export",
        "export_csv": "Als CSV exportieren",
        "export_excel": "Als Excel exportieren",
        "download": "Herunterladen",

        # Emotions
        "positive": "Positiv",
        "negative": "Negativ",
        "neutral": "Neutral",
        "joy": "Freude",
        "surprise": "Überraschung",
        "fear": "Angst",
        "anger": "Wut",
        "disgust": "Ekel",
        "sadness": "Trauer",
        "contempt": "Verachtung",

        # Errors
        "error_no_text": "Bitte geben Sie einen Text ein.",
        "error_no_file": "Bitte laden Sie eine Datei hoch.",
        "error_api_key": "API-Schlüssel fehlt oder ungültig.",
        "error_processing": "Fehler bei der Verarbeitung.",
        "error_file_format": "Ungültiges Dateiformat.",

        # Arc Analysis
        "arc_pattern": "Erkanntes Muster",
        "arc_confidence": "Konfidenz",
        "arc_features": "Arc-Eigenschaften",
        "key_moments": "Schlüsselmomente",
        "story_progression": "Handlungsverlauf"
    },

    "EN": {
        # Navigation
        "title": "🧠 Sentiment Analysis Toolkit",
        "subtitle": "Professional Sentiment and Emotion Analysis",

        # Sidebar
        "analysis_type": "Analysis Type",
        "model_selection": "Model Selection",
        "benchmark_mode": "Benchmark Mode",
        "benchmark_description": "Compares all available models",
        "single_model": "Single Model",
        "language_settings": "Language Settings",

        # Input
        "input_method": "Input Method",
        "single_text": "Single Text",
        "batch_upload": "Batch Upload",
        "text_input": "Enter Text",
        "text_placeholder": "Enter your text here...",
        "file_upload": "Upload File",
        "file_types": "Supported formats: CSV, TXT",

        # Analysis
        "analyze_button": "Analyze",
        "analyzing": "Analyzing...",
        "analysis_complete": "Analysis Complete",
        "text_count": "Number of Texts",
        "processing_time": "Processing Time",

        # Results
        "results": "Results",
        "valence_results": "Valence Results",
        "ekman_results": "Ekman Emotions",
        "emotion_arc_results": "Emotion Arc",
        "benchmark_results": "Benchmark Results",

        # Export
        "export": "Export",
        "export_csv": "Export as CSV",
        "export_excel": "Export as Excel",
        "download": "Download",

        # Emotions
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
        "joy": "Joy",
        "surprise": "Surprise",
        "fear": "Fear",
        "anger": "Anger",
        "disgust": "Disgust",
        "sadness": "Sadness",
        "contempt": "Contempt",

        # Errors
        "error_no_text": "Please enter a text.",
        "error_no_file": "Please upload a file.",
        "error_api_key": "API key missing or invalid.",
        "error_processing": "Error during processing.",
        "error_file_format": "Invalid file format.",

        # Arc Analysis
        "arc_pattern": "Detected Pattern",
        "arc_confidence": "Confidence",
        "arc_features": "Arc Features",
        "key_moments": "Key Moments",
        "story_progression": "Story Progression"
    }
}


def get_text(key: str, language: str = "DE") -> str:
    """Holt übersetzten Text"""
    return TRANSLATIONS.get(language, {}).get(key, key)
