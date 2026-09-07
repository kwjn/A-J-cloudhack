"""Definitions and English meanings for supported SgSL intents."""


INTENTS = {
    "ALLERGY": {
        "intent_id": "ALLERGY",
        "english": "I have a food allergy.",
        "critical": True,
    },
    "VEGETARIAN": {
        "intent_id": "VEGETARIAN",
        "english": "Is this vegetarian?",
        "critical": False,
    },
    "CONTAINS_PORK": {
        "intent_id": "CONTAINS_PORK",
        "english": "Does this contain pork?",
        "critical": False,
    },
    "HOW_MUCH": {
        "intent_id": "HOW_MUCH",
        "english": "How much is this?",
        "critical": False,
    },
    "WRONG_ORDER": {
        "intent_id": "WRONG_ORDER",
        "english": "This is not what I ordered.",
        "critical": False,
    },
    "NOT_RECEIVED": {
        "intent_id": "NOT_RECEIVED",
        "english": "I haven't received my food yet.",
        "critical": False,
    },
    "PLEASE_REPEAT": {
        "intent_id": "PLEASE_REPEAT",
        "english": "Please repeat that.",
        "critical": False,
    },
    "PACK": {
        "intent_id": "PACK",
        "english": "Please pack this.",
        "critical": False,
    },
}
