"""Mock Streamlit interface for two-way hawker-centre communication."""

import streamlit as st

from services.elevenlabs_service import (
    VoiceConfigurationError,
    VoiceOutputError,
    get_voice_id,
    speak_text,
)


MOCK_INTENTS = {
    "ALLERGY": {
        "text": "I have a food allergy.",
        "critical": True,
    },
    "VEGETARIAN": {
        "text": "Is this vegetarian?",
        "critical": False,
    },
    "CONTAINS_PORK": {
        "text": "Does this contain pork?",
        "critical": False,
    },
    "HOW_MUCH": {
        "text": "How much is this?",
        "critical": False,
    },
    "WRONG_ORDER": {
        "text": "This is not what I ordered.",
        "critical": False,
    },
    "NOT_RECEIVED": {
        "text": "I haven't received my food yet.",
        "critical": False,
    },
    "PLEASE_REPEAT": {
        "text": "Please repeat that.",
        "critical": False,
    },
    "TAKEAWAY": {
        "text": "Takeaway, please.",
        "critical": False,
    },
}

# Frontend demonstration behavior only. This is not a classifier threshold.
MOCK_REPEAT_THRESHOLD = 0.50


def make_mock_result(intent: str, confidence: float) -> dict:
    """Return a mock result using the future recogniser's result contract."""
    intent_details = MOCK_INTENTS[intent]
    return {
        "intent": intent,
        "text": intent_details["text"],
        "confidence": confidence,
        "critical": intent_details["critical"],
    }


def initialize_state() -> None:
    """Set the initial UI state once per browser session."""
    if "signing_status" not in st.session_state:
        st.session_state.signing_status = "READY"
    if "recognition_result" not in st.session_state:
        st.session_state.recognition_result = None
    if "speech_audio" not in st.session_state:
        st.session_state.speech_audio = None
    if "voice_error" not in st.session_state:
        st.session_state.voice_error = None


def status_badge(status: str) -> str:
    """Return accessible badge markup for a known signing status."""
    css_class = "status-ready"
    if status == "RECOGNISED":
        css_class = "status-success"
    elif status == "RECOGNISING":
        css_class = "status-working"
    elif status == "PLEASE REPEAT":
        css_class = "status-repeat"
    return f'<span class="status-badge {css_class}">{status}</span>'


@st.cache_data(show_spinner=False)
def cached_speech(text: str, voice_id: str) -> bytes:
    """Cache generated audio by its spoken text and configured voice."""
    return speak_text(text, voice_id=voice_id)


st.set_page_config(
    page_title="Hawker Hands",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17211b;
        --muted: #5e6b63;
        --cream: #fffdf7;
        --green: #0b6847;
        --green-soft: #e7f5ee;
        --amber-soft: #fff3d5;
        --border: #dce5df;
    }
    .stApp { background: #f4f1e8; color: var(--ink); }
    .block-container { max-width: 1120px; padding: 1.6rem 2rem 2rem; }
    h1 { color: var(--ink); font-size: clamp(2rem, 4vw, 3.15rem) !important; }
    h2 { color: var(--ink); font-size: clamp(1.45rem, 2.5vw, 2rem) !important; }
    .app-kicker {
        color: var(--green); font-size: .82rem; font-weight: 800;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: .25rem;
    }
    .app-subtitle { color: var(--muted); font-size: 1.08rem; margin: -.4rem 0 1.4rem; }
    .message-card {
        min-height: 155px; display: flex; align-items: center;
        background: var(--cream); border: 2px solid var(--border);
        border-radius: 20px; padding: 1.5rem 1.7rem; margin: .8rem 0 .65rem;
        box-shadow: 0 5px 18px rgba(35, 54, 43, .06);
    }
    .message-text {
        color: var(--ink); font-size: clamp(2rem, 4vw, 3.25rem);
        font-weight: 750; line-height: 1.12; letter-spacing: -.025em;
    }
    .placeholder-text {
        color: var(--muted); font-size: clamp(1.55rem, 3vw, 2.35rem);
        font-weight: 600; line-height: 1.2;
    }
    .status-badge {
        display: inline-block; border-radius: 999px; padding: .38rem .78rem;
        font-size: .8rem; font-weight: 850; letter-spacing: .08em;
    }
    .status-ready { background: #edf0ee; color: #3f4c44; }
    .status-success { background: var(--green-soft); color: var(--green); }
    .status-working { background: #e8f0ff; color: #2459a6; }
    .status-repeat { background: var(--amber-soft); color: #745300; }
    .confidence { color: var(--muted); font-size: 1rem; font-weight: 650; }
    .critical-note {
        background: var(--amber-soft); border-left: 5px solid #d79c14;
        border-radius: 10px; color: #5e470d; font-weight: 720;
        margin: .8rem 0; padding: .72rem .9rem;
    }
    .direction-note { color: var(--muted); margin-top: -.45rem; }
    div.stButton > button { min-height: 3rem; font-size: 1rem; font-weight: 700; }
    [data-testid="stExpander"] {
        background: rgba(255,255,255,.55); border: 1px solid var(--border);
        border-radius: 14px;
    }
    @media (max-width: 700px) {
        .block-container { padding: 1rem; }
        .message-card { min-height: 130px; padding: 1.1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

initialize_state()

st.markdown('<div class="app-kicker">Hawker Hands</div>', unsafe_allow_html=True)
st.title("A clearer conversation, both ways")
st.markdown(
    '<div class="app-subtitle">A simple communication aid for Deaf customers and hawkers.</div>',
    unsafe_allow_html=True,
)

signing_column, reply_column = st.columns(2, gap="large")

with signing_column:
    st.header("Signing → Hawker")
    st.markdown(
        '<div class="direction-note">Your signed message, shown in English</div>',
        unsafe_allow_html=True,
    )
    st.markdown(status_badge(st.session_state.signing_status), unsafe_allow_html=True)

    result = st.session_state.recognition_result
    if st.session_state.signing_status == "PLEASE REPEAT":
        signing_message = "Please repeat the sign."
    elif result is not None:
        signing_message = result["text"]
    else:
        signing_message = "Your recognised message will appear here."

    message_class = "message-text" if result is not None else "placeholder-text"
    st.markdown(
        f'<div class="message-card"><div class="{message_class}">{signing_message}</div></div>',
        unsafe_allow_html=True,
    )

    if result is not None:
        st.markdown(
            f'<div class="confidence">Confidence: {result["confidence"]:.0%}</div>',
            unsafe_allow_html=True,
        )
        if result["critical"] and st.session_state.signing_status == "RECOGNISED":
            st.markdown(
                '<div class="critical-note">Important dietary/safety message</div>',
                unsafe_allow_html=True,
            )

    can_speak = result is not None and st.session_state.signing_status == "RECOGNISED"
    if st.button(
        "Speak to Hawker",
        disabled=not can_speak,
        use_container_width=True,
        type="primary" if can_speak else "secondary",
        help=(
            "Generate voice output for the displayed message."
            if can_speak
            else "Recognise a message before using voice output."
        ),
    ):
        st.session_state.voice_error = None
        try:
            with st.spinner("Preparing voice output…"):
                voice_id = get_voice_id()
                st.session_state.speech_audio = cached_speech(
                    result["text"],
                    voice_id,
                )
        except VoiceConfigurationError as error:
            st.session_state.speech_audio = None
            st.session_state.voice_error = str(error)
        except (VoiceOutputError, ValueError):
            st.session_state.speech_audio = None
            st.session_state.voice_error = (
                "ElevenLabs could not generate audio. Check the connection and configuration."
            )

    if st.session_state.voice_error:
        st.warning("Voice output unavailable. Please use the displayed message.")
    elif st.session_state.speech_audio is not None:
        st.audio(
            st.session_state.speech_audio,
            format="audio/mpeg",
            autoplay=True,
        )

with reply_column:
    st.header("Hawker → You")
    st.markdown(
        '<div class="direction-note">The hawker’s spoken reply, shown as text</div>',
        unsafe_allow_html=True,
    )
    st.markdown(status_badge("READY"), unsafe_allow_html=True)
    st.markdown(
        '<div class="message-card"><div class="placeholder-text">The hawker\'s spoken reply will appear here.</div></div>',
        unsafe_allow_html=True,
    )
    st.button(
        "Record Hawker Reply",
        disabled=True,
        use_container_width=True,
        help="Speech-to-text will be connected next.",
    )
    st.caption("Speech-to-text will be connected next.")

st.divider()

with st.expander("Development / demo test", expanded=False):
    st.caption("Mock controls for testing the interface before recognition is connected.")
    control_a, control_b = st.columns([1.4, 1])
    with control_a:
        selected_intent = st.selectbox("Intent", tuple(MOCK_INTENTS), index=7)
    with control_b:
        mock_confidence = st.slider(
            "Confidence test",
            min_value=0.0,
            max_value=1.0,
            value=0.93,
            step=0.01,
            help="Frontend demo value only; this is not a classifier threshold.",
        )

    if st.button("Simulate Recognition", type="primary", use_container_width=True):
        st.session_state.signing_status = "RECOGNISING"
        mock_result = make_mock_result(selected_intent, mock_confidence)
        st.session_state.recognition_result = mock_result
        st.session_state.speech_audio = None
        st.session_state.voice_error = None
        st.session_state.signing_status = (
            "PLEASE REPEAT"
            if mock_confidence < MOCK_REPEAT_THRESHOLD
            else "RECOGNISED"
        )
        st.rerun()

    if st.session_state.voice_error:
        st.caption(f"Voice setup: {st.session_state.voice_error}")
