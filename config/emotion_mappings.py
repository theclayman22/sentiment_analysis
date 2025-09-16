"""
Emotionen-Mappings und Synonyme für Ekman-Emotionen
"""

# Ekman-Emotionen mit Synonymen
EKMAN_EMOTIONS = {
    "joy": {
        "name": "Freude",
        "name_en": "Joy",
        "synonyms": [
            "happiness", "delight", "pleasure", "bliss", "cheerfulness",
            "contentment", "satisfaction", "elation", "euphoria", "glee",
            "jubilation", "mirth", "amusement", "enjoyment", "gladness"
        ]
    },
    "surprise": {
        "name": "Überraschung",
        "name_en": "Surprise",
        "synonyms": [
            "astonishment", "amazement", "wonder", "bewilderment", "shock",
            "stupefaction", "perplexity", "confusion", "disbelief", "awe"
        ]
    },
    "fear": {
        "name": "Angst",
        "name_en": "Fear",
        "synonyms": [
            "anxiety", "terror", "panic", "fright", "dread", "horror",
            "apprehension", "nervousness", "worry", "concern", "alarm",
            "trepidation", "unease", "phobia", "scared"
        ]
    },
    "anger": {
        "name": "Wut",
        "name_en": "Anger",
        "synonyms": [
            "rage", "fury", "wrath", "ire", "annoyance", "irritation",
            "frustration", "indignation", "hostility", "resentment",
            "outrage", "exasperation", "vexation", "mad", "furious"
        ]
    },
    "disgust": {
        "name": "Ekel",
        "name_en": "Disgust",
        "synonyms": [
            "revulsion", "repulsion", "nausea", "loathing", "abhorrence",
            "aversion", "distaste", "repugnance", "disdain",
            "revulsion", "sickening", "disgusting", "revolting"
        ]
    },
    "sadness": {
        "name": "Trauer",
        "name_en": "Sadness",
        "synonyms": [
            "sorrow", "grief", "melancholy", "depression", "despair",
            "dejection", "despondency", "gloom", "misery", "unhappiness",
            "heartbreak", "anguish", "woe", "mourning", "regret"
        ]
    },
    "contempt": {
        "name": "Verachtung",
        "name_en": "Contempt",
        "synonyms": [
            "scorn", "disdain", "derision", "condescension", "arrogance",
            "superiority", "dismissal", "mockery", "ridicule", "sneering"
        ]
    }
}

# Valence-Kategorien mit Synonymen
VALENCE_CATEGORIES = {
    "positive": {
        "name": "Positiv",
        "name_en": "Positive",
        "synonyms": [
            "good", "great", "excellent", "wonderful", "fantastic",
            "amazing", "awesome", "brilliant", "outstanding", "superb",
            "marvelous", "terrific", "fabulous", "splendid", "magnificent"
        ]
    },
    "negative": {
        "name": "Negativ", 
        "name_en": "Negative",
        "synonyms": [
            "bad", "terrible", "awful", "horrible", "dreadful",
            "disgusting", "appalling", "atrocious", "abysmal", "deplorable",
            "ghastly", "hideous", "revolting", "repulsive", "vile"
        ]
    },
    "neutral": {
        "name": "Neutral",
        "name_en": "Neutral",
        "synonyms": [
            "okay", "alright", "fine", "acceptable", "adequate",
            "average", "moderate", "fair", "reasonable", "standard"
        ]
    }
}


def get_all_emotion_terms(emotion_key: str) -> list:
    """Gibt alle Begriffe (Hauptemotion + Synonyme) für eine Emotion zurück"""
    if emotion_key in EKMAN_EMOTIONS:
        return [emotion_key] + EKMAN_EMOTIONS[emotion_key]["synonyms"]
    elif emotion_key in VALENCE_CATEGORIES:
        return [emotion_key] + VALENCE_CATEGORIES[emotion_key]["synonyms"]
    else:
        return [emotion_key]


def get_emotion_display_name(emotion_key: str, language: str = "DE") -> str:
    """Gibt den Anzeigenamen für eine Emotion zurück"""
    if emotion_key in EKMAN_EMOTIONS:
        return EKMAN_EMOTIONS[emotion_key]["name" if language == "DE" else "name_en"]
    elif emotion_key in VALENCE_CATEGORIES:
        return VALENCE_CATEGORIES[emotion_key]["name" if language == "DE" else "name_en"]
    else:
        return emotion_key.title()
