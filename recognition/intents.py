"""Definitions and English meanings for supported SgSL intents."""


INTENTS = {
    "ALLERGY": {
        "intent_id": "ALLERGY",
        "english": "I am allergic to this.",
        "critical": True,
    },
    "VEGETARIAN": {
        "intent_id": "VEGETARIAN",
        "english": "Is this vegetarian?",
        "critical": False,
    },
    "NO_SPICY": {
        "intent_id": "NO_SPICY",
        "english": "Please make it not spicy.",
        "critical": False,
    },
    "WRONG_ORDER": {
        "intent_id": "WRONG_ORDER",
        "english": "This is not what I ordered.",
        "critical": False,
    },
    "HAVENT_RECEIVED_ORDER": {
        "intent_id": "HAVENT_RECEIVED_ORDER",
        "english": "I haven't received my order yet.",
        "critical": False,
    },
}
