"""Streamlit interface for two-way hawker-centre communication."""

import hashlib
import os
import re
import threading
import time
from uuid import uuid4

import av
from dotenv import load_dotenv
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from recognition.live_predictor import LiveSignPredictor
from services.elevenlabs_service import (
    SpeechTranscriptionError,
    VoiceConfigurationError,
    VoiceOutputError,
    get_voice_id,
    speak_text,
    transcribe_audio,
)

HAWKER_REPLY_MAP = {
    "YES": ["yes", "can", "okay", "ok", "sure", "have"],
    "NO": ["no", "cannot", "can't", "dont have", "don't have", "not available"],
    "PLEASE_REPEAT": ["repeat", "say again", "again please", "pardon"],
}

SIGN_VIDEO_MAP = {
    "YES": "assets/signs/yes.mp4",
    "NO": "assets/signs/no.mp4",
    "PLEASE_REPEAT": "assets/signs/please_repeat.mp4",
}

STUN_SERVER_URL = "stun:stun.l.google.com:19302"


def build_rtc_configuration() -> dict:
    """Build browser ICE configuration without hardcoding TURN credentials."""
    load_dotenv()
    ice_servers = [{"urls": [STUN_SERVER_URL]}]
    turn_url = os.getenv("TURN_URL", "").strip()
    turn_username = os.getenv("TURN_USERNAME", "").strip()
    turn_credential = os.getenv("TURN_CREDENTIAL", "").strip()

    if turn_url and turn_username and turn_credential:
        ice_servers.append(
            {
                "urls": [turn_url],
                "username": turn_username,
                "credential": turn_credential,
            }
        )
    return {"iceServers": ice_servers}


class LatestPredictionStore:
    """Thread-safe in-memory handoff from WebRTC to the Streamlit thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = {}
        self._recognition_status = {
            "signer_state": "STARTING",
            "recording": False,
            "message": "Starting camera…",
        }

    def publish(self, prediction: dict) -> None:
        latest = dict(prediction)
        latest.setdefault("prediction_id", uuid4().hex)
        with self._lock:
            self._latest = latest

    def read(self) -> dict | None:
        with self._lock:
            return dict(self._latest) if self._latest else None

    def publish_status(
        self,
        signer_state: str,
        recording: bool,
        message: str,
    ) -> None:
        status = {
            "signer_state": signer_state,
            "recording": recording,
            "message": message,
        }
        with self._lock:
            self._recognition_status = status

    def read_status(self) -> dict:
        with self._lock:
            return dict(self._recognition_status)


class SignVideoProcessor(VideoProcessorBase):
    """Run the cached live predictor on frames from the browser camera."""

    def __init__(
        self,
        predictor: LiveSignPredictor,
        prediction_store: LatestPredictionStore,
    ):
        self._predictor = predictor
        self._prediction_store = prediction_store
        self._predictor_lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        with self._predictor_lock:
            annotated_frame, prediction = self._predictor.process_frame(image)
            signer_state = (
                self._predictor.session_result.state.value
                if self._predictor.session_result is not None
                else "STARTING"
            )
            recording = self._predictor.recording
            status_message = self._predictor.status_message
        self._prediction_store.publish_status(
            signer_state,
            recording,
            status_message,
        )
        if prediction is not None:
            self._prediction_store.publish(prediction)
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    def start_recording(self) -> bool:
        """Apply the predictor's start/stop behavior under its frame lock."""
        with self._predictor_lock:
            prediction = self._predictor.start_recording()
            recording = self._predictor.recording
            signer_state = (
                self._predictor.session_result.state.value
                if self._predictor.session_result is not None
                else "STARTING"
            )
            status_message = self._predictor.status_message
        self._prediction_store.publish_status(
            signer_state,
            recording,
            status_message,
        )
        if prediction is not None:
            self._prediction_store.publish(prediction)
        return recording

    def discard_recording(self) -> None:
        """Apply the existing R behavior safely between video frames."""
        with self._predictor_lock:
            self._predictor.discard_recording()
            signer_state = (
                self._predictor.session_result.state.value
                if self._predictor.session_result is not None
                else "STARTING"
            )
            status_message = self._predictor.status_message
        self._prediction_store.publish_status(
            signer_state,
            False,
            status_message,
        )

    def is_recording(self) -> bool:
        with self._predictor_lock:
            return self._predictor.recording


@st.cache_resource(show_spinner="Loading sign recognition…")
def get_live_predictor(session_key: str) -> LiveSignPredictor:
    """Create one expensive recognition pipeline per browser session."""
    return LiveSignPredictor(
        control_hint="Use Start Recording and Discard below",
    )


@st.cache_resource
def get_prediction_store(session_key: str) -> LatestPredictionStore:
    """Create one WebRTC-to-Streamlit handoff per browser session."""
    return LatestPredictionStore()


def map_hawker_reply(text: str) -> str | None:
    """Map the hawker's transcript to a supported reply intent."""
    normalized = text.lower().strip()

    for intent in ("NO", "PLEASE_REPEAT", "YES"):
        for phrase in HAWKER_REPLY_MAP[intent]:
            pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
            if re.search(pattern, normalized):
                return intent

    return None


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
    if "recognition_session_key" not in st.session_state:
        st.session_state.recognition_session_key = uuid4().hex
    if "recognition_camera_status" not in st.session_state:
        st.session_state.recognition_camera_status = {
            "signer_state": "STARTING",
            "recording": False,
            "message": "Start the camera to begin.",
        }
    if "hawker_transcript" not in st.session_state:
        st.session_state.hawker_transcript = None
    if "hawker_audio_hash" not in st.session_state:
        st.session_state.hawker_audio_hash = None
    if "stt_error" not in st.session_state:
        st.session_state.stt_error = None
    if "hawker_reply_intent" not in st.session_state:
        st.session_state.hawker_reply_intent = None


def status_badge(status: str) -> str:
    """Return accessible badge markup for a known signing status."""
    css_class = "status-ready"
    if status == "RECOGNISED":
        css_class = "status-success"
    elif status in ("RECOGNISING", "RECORDING"):
        css_class = "status-working"
    elif status == "PLEASE REPEAT":
        css_class = "status-repeat"
    return f'<span class="status-badge {css_class}">{status}</span>'


@st.cache_data(show_spinner=False)
def cached_speech(text: str, voice_id: str) -> bytes:
    """Cache generated audio by its spoken text and configured voice."""
    return speak_text(text, voice_id=voice_id)


def load_recognition_updates(prediction_store: LatestPredictionStore) -> bool:
    """Consume changed WebRTC prediction or capture state."""
    changed = False
    latest_prediction = prediction_store.read()
    if (
        latest_prediction is not None
        and latest_prediction["prediction_id"] != st.session_state.last_prediction_id
    ):
        st.session_state.real_recognition_result = latest_prediction
        st.session_state.last_prediction_id = latest_prediction["prediction_id"]
        st.session_state.speech_audio = None
        st.session_state.voice_error = None
        changed = True

    latest_status = prediction_store.read_status()
    if latest_status != st.session_state.recognition_camera_status:
        st.session_state.recognition_camera_status = latest_status
        changed = True
    return changed


@st.fragment(run_every=0.75)
def poll_for_prediction(prediction_store: LatestPredictionStore) -> None:
    """Refresh only when a WebRTC prediction or capture state changes."""
    if load_recognition_updates(prediction_store):
        st.rerun()


def render_signing_panel(
    predictor: LiveSignPredictor,
    prediction_store: LatestPredictionStore,
) -> None:
    """Render the browser camera and latest real recognition result."""
    result = st.session_state.real_recognition_result

    st.header("Signing → Hawker")
    st.markdown(
        '<div class="direction-note">Your signed message, shown in English</div>',
        unsafe_allow_html=True,
    )

    webrtc_context = webrtc_streamer(
        key="sgsl-inline-camera",
        video_processor_factory=lambda: SignVideoProcessor(
            predictor,
            prediction_store,
        ),
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=build_rtc_configuration(),
        async_processing=True,
    )
    video_processor = webrtc_context.video_processor
    camera_status = st.session_state.recognition_camera_status
    signer_state = camera_status["signer_state"]
    recording = camera_status["recording"]
    recognition_message = camera_status["message"]
    st.markdown(
        status_badge("RECORDING" if recording else "READY"),
        unsafe_allow_html=True,
    )
    if signer_state == "LOCKED":
        st.caption("Signer locked. Ready to record.")
    elif video_processor is None:
        st.caption("Start the camera and allow browser camera access.")
    else:
        st.caption("Hold still and face the camera while signer lock is acquired.")
    if recognition_message.startswith("REJECTED:"):
        st.warning(recognition_message)

    capture_column, discard_column = st.columns(2)
    with capture_column:
        if st.button(
            "Stop Recording" if recording else "Start Recording",
            disabled=(
                video_processor is None
                or (not recording and signer_state != "LOCKED")
            ),
            use_container_width=True,
        ):
            if not recording:
                countdown = st.empty()
                for number in (3, 2, 1):
                    countdown.markdown(
                        f"### Recording starts in {number}…"
                    )
                    time.sleep(1)
                countdown.empty()
            video_processor.start_recording()
            st.rerun()
    with discard_column:
        if st.button(
            "Discard",
            disabled=video_processor is None or not recording,
            use_container_width=True,
        ):
            video_processor.discard_recording()
            st.rerun()

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
recognition_session_key = st.session_state.recognition_session_key
live_predictor = get_live_predictor(recognition_session_key)
prediction_store = get_prediction_store(recognition_session_key)

st.markdown('<div class="app-kicker">Hawker Hands</div>', unsafe_allow_html=True)
st.title("A clearer conversation, both ways")
st.markdown(
    '<div class="app-subtitle">A simple communication aid for Deaf customers and hawkers.</div>',
    unsafe_allow_html=True,
)

poll_for_prediction(prediction_store)

signing_column, reply_column = st.columns(2, gap="large")

with signing_column:
    render_signing_panel(live_predictor, prediction_store)

with reply_column:
    st.header("Hawker → You")
    st.markdown(
        '<div class="direction-note">The hawker’s spoken reply, shown as text</div>',
        unsafe_allow_html=True,
    )
    st.markdown(status_badge("READY"), unsafe_allow_html=True)

    if st.session_state.hawker_transcript:
        st.markdown(
            f'<div class="message-card"><div>{st.session_state.hawker_transcript}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="message-card"><div class="placeholder-text">The hawker\'s spoken reply will appear here.</div></div>',
            unsafe_allow_html=True,
        )

    if st.session_state.stt_error:
        st.warning(st.session_state.stt_error)

    if st.session_state.hawker_reply_intent:
        sign_video = SIGN_VIDEO_MAP.get(
            st.session_state.hawker_reply_intent
        )

        if sign_video:
            st.caption("SgSL response")
            st.video(sign_video, autoplay=True)

    hawker_audio = st.audio_input("Record Hawker Reply")

    if hawker_audio is not None:
        audio_bytes = hawker_audio.getvalue()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()

        if audio_hash != st.session_state.hawker_audio_hash:
            st.session_state.hawker_audio_hash = audio_hash
            st.session_state.stt_error = None

            try:
                with st.spinner("Transcribing..."):
                    st.session_state.hawker_transcript = transcribe_audio(audio_bytes)
                    st.session_state.hawker_reply_intent = map_hawker_reply(
                        st.session_state.hawker_transcript
                    )
            except (VoiceConfigurationError, SpeechTranscriptionError):
                st.session_state.hawker_transcript = None
                st.session_state.hawker_reply_intent = None
                st.session_state.stt_error = (
                    "Could not transcribe the hawker's reply. Please try again."
                )
