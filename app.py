"""Streamlit interface for two-way hawker-centre communication."""

import streamlit as st

from services.elevenlabs_service import (
    VoiceConfigurationError,
    VoiceOutputError,
    get_voice_id,
    speak_text,
)
from services.recognition_bridge import read_latest_prediction


def initialize_state() -> None:
    """Set the initial UI state once per browser session."""
    if "speech_audio" not in st.session_state:
        st.session_state.speech_audio = None
    if "voice_error" not in st.session_state:
        st.session_state.voice_error = None
    if "real_recognition_result" not in st.session_state:
        st.session_state.real_recognition_result = None
    if "last_prediction_id" not in st.session_state:
        st.session_state.last_prediction_id = None


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


def load_new_prediction() -> bool:
    """Consume a newly published real prediction without replaying duplicates."""
    latest_prediction = read_latest_prediction()
    if (
        latest_prediction is not None
        and latest_prediction["prediction_id"] != st.session_state.last_prediction_id
    ):
        st.session_state.real_recognition_result = latest_prediction
        st.session_state.last_prediction_id = latest_prediction["prediction_id"]
        st.session_state.speech_audio = None
        st.session_state.voice_error = None
        return True
    return False


@st.fragment(run_every=0.75)
def poll_for_prediction() -> None:
    """Poll silently and refresh the full page only for a new prediction."""
    if load_new_prediction():
        st.rerun()


def render_signing_panel() -> None:
    """Render the latest real recognition result."""
    result = st.session_state.real_recognition_result
    display_status = "RECOGNISED" if result is not None else "READY"

    st.header("Signing → Hawker")
    st.markdown(
        '<div class="direction-note">Your signed message, shown in English</div>',
        unsafe_allow_html=True,
    )
    st.markdown(status_badge(display_status), unsafe_allow_html=True)

    signing_message = result["text"] if result else "Waiting for a sign..."
    message_class = "message-text" if result else "placeholder-text"
    st.markdown(
        f'<div class="message-card"><div class="{message_class}">{signing_message}</div></div>',
        unsafe_allow_html=True,
    )

    if result is not None:
        st.markdown(
            f'<div class="confidence">Confidence: {result["confidence"]:.0%}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Intent: {result['intent']}")
        if result["critical"]:
            st.markdown(
                '<div class="critical-note">Important dietary/safety message</div>',
                unsafe_allow_html=True,
            )

    can_speak = result is not None
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
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCaptionContainer"] p {
        color: var(--ink);
    }
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
    div.stButton > button {
        min-height: 3rem; font-size: 1rem; font-weight: 700;
        color: var(--ink); background: var(--cream); border-color: #aebbb3;
    }
    div.stButton > button[kind="primary"] {
        color: #ffffff; background: var(--green); border-color: var(--green);
    }
    div.stButton > button:disabled {
        color: #5d6861 !important; background: #e4e5df !important;
        border-color: #b9c0bb !important; opacity: 1 !important;
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

poll_for_prediction()

signing_column, reply_column = st.columns(2, gap="large")

with signing_column:
    render_signing_panel()

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
