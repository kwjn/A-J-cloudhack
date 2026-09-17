"""Streamlit interface for two-way hawker-centre communication."""

import hashlib
import html
import os
from pathlib import Path
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
    "YES": [
        "yes", "can", "okay", "ok", "sure", "have", "i have it",
        "we have it", "i can", "we can",
    ],
    "NO": [
        "no", "cannot", "can't", "dont have", "don't have", "not available",
        "not okay", "not ok", "do not have", "i do not have it",
        "i don't have it", "we do not have it", "we don't have it",
        "i cannot", "we cannot", "i can't", "we can't",
    ],
    "PLEASE_REPEAT": ["repeat", "say again", "again please", "pardon"],
}

APP_DIR = Path(__file__).resolve().parent
SIGN_VIDEO_MAP = {
    "YES": str(APP_DIR / "assets/signs/yes.mp4"),
    "NO": str(APP_DIR / "assets/signs/no.mp4"),
    "PLEASE_REPEAT": str(APP_DIR / "assets/signs/please_repeat.mp4"),
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
                signer_state, recording, status_message,
            )
            if prediction is not None:
                self._prediction_store.publish(prediction)
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    def start_recording(self) -> bool:
        """Start only if capture is currently stopped."""
        return self._set_recording(True)

    def stop_recording(self) -> bool:
        """Stop only if capture is currently running, including after auto-stop."""
        return self._set_recording(False)

    def _set_recording(self, requested: bool) -> bool:
        """Check and change capture state atomically with frame processing."""
        with self._predictor_lock:
            prediction = None
            if self._predictor.recording != requested:
                prediction = self._predictor.start_recording()
            recording = self._predictor.recording
            signer_state = (
                self._predictor.session_result.state.value
                if self._predictor.session_result is not None
                else "STARTING"
            )
            status_message = self._predictor.status_message
            self._prediction_store.publish_status(
                signer_state, recording, status_message,
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
        control_hint="Use Start recording and Stop recording below",
    )


@st.cache_resource
def get_prediction_store(session_key: str) -> LatestPredictionStore:
    """Create one WebRTC-to-Streamlit handoff per browser session."""
    return LatestPredictionStore()


def map_hawker_reply(text: str) -> str | None:
    """Map complete supported replies; leave ambiguous speech as text only."""
    normalized = text.casefold().replace("\u2019", "'").replace("\u2018", "'")
    normalized = re.sub(r"[^\w\s']", " ", normalized)
    normalized = " ".join(normalized.split())
    for intent, phrases in HAWKER_REPLY_MAP.items():
        if normalized in phrases:
            return intent
    # Allow polite wrappers, but never infer affirmation from a word embedded
    # in a longer sentence (e.g. 'not okay' or 'can you check?').
    normalized = re.sub(r"^(?:please|sorry)\s+", "", normalized)
    normalized = re.sub(r"\s+(?:please|thank you|thanks)$", "", normalized)
    matches = {
        intent for intent, phrases in HAWKER_REPLY_MAP.items()
        if normalized in phrases
    }
    return matches.pop() if len(matches) == 1 else None


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
            "message": "Preparing the camera…",
        }
    if "recording_started_at" not in st.session_state:
        st.session_state.recording_started_at = None
    if "hawker_transcript" not in st.session_state:
        st.session_state.hawker_transcript = None
    if "hawker_audio_hash" not in st.session_state:
        st.session_state.hawker_audio_hash = None
    if "hawker_failed_audio_hash" not in st.session_state:
        st.session_state.hawker_failed_audio_hash = None
    if "stt_error" not in st.session_state:
        st.session_state.stt_error = None
    if "hawker_reply_intent" not in st.session_state:
        st.session_state.hawker_reply_intent = None


def status_badge(status: str) -> str:
    """Return accessible badge markup for a known interface status."""
    normalized_status = status.upper()
    css_class = "status-ready"
    if normalized_status in ("RESULT", "RECOGNISED"):
        css_class = "status-success"
    elif normalized_status in ("COUNTDOWN", "PROCESSING"):
        css_class = "status-working"
    elif normalized_status == "RECORDING":
        css_class = "status-recording"
    elif normalized_status in ("ERROR", "PLEASE REPEAT"):
        css_class = "status-error"
    return (
        f'<span class="status-badge {css_class}">'
        f'{normalized_status.title()}</span>'
    )


def recording_control_states(
    camera_available: bool,
    signer_state: str,
    recording: bool,
) -> tuple[bool, bool]:
    """Return whether the Start and Stop controls should be disabled."""
    start_disabled = (
        not camera_available or recording or signer_state != "LOCKED"
    )
    stop_disabled = not camera_available or not recording
    return start_disabled, stop_disabled


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


@st.fragment(run_every=0.5)
def render_recording_indicator(recording: bool) -> None:
    """Keep the elapsed recording time visible without refreshing the camera."""
    if not recording:
        return
    started_at = st.session_state.recording_started_at or time.monotonic()
    elapsed = max(0.0, time.monotonic() - started_at)
    st.markdown(
        '<div class="recording-notice"><span></span>'
        f'Recording sign <strong>{elapsed:0.1f}s</strong></div>',
        unsafe_allow_html=True,
    )
    st.caption("Complete your sign naturally, then select Stop recording.")


def render_signer_panel(
    predictor: LiveSignPredictor,
    prediction_store: LatestPredictionStore,
) -> None:
    """Render the browser camera and manual sign recording controls."""
    st.header("You")
    st.markdown(
        '<div class="direction-note">Your live signing camera</div>',
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
        desired_playing_state=True,
        media_toggle_controls=False,
        video_html_attrs={"autoPlay": True, "controls": False, "muted": True},
    )
    video_processor = webrtc_context.video_processor
    camera_status = st.session_state.recognition_camera_status
    signer_state = camera_status["signer_state"]
    recording = camera_status["recording"]
    recognition_message = camera_status["message"]
    recognition_error = recognition_message.startswith("REJECTED:")

    if recognition_error:
        camera_display_status = "ERROR"
    elif recording:
        camera_display_status = "RECORDING"
    elif video_processor is None or signer_state != "LOCKED":
        camera_display_status = "PROCESSING"
    else:
        camera_display_status = "READY"

    status_placeholder = st.empty()
    status_placeholder.markdown(
        status_badge(camera_display_status), unsafe_allow_html=True
    )
    if recognition_error:
        st.warning("We could not read that sign. Please record it again.")
    elif signer_state == "LOCKED":
        st.caption("You are in frame and ready to sign.")
    elif video_processor is None:
        st.caption("Allow camera access in your browser to show the live preview.")
    else:
        st.caption("Face the camera and hold still briefly while we find you.")

    render_recording_indicator(recording)

    start_disabled, stop_disabled = recording_control_states(
        video_processor is not None,
        signer_state,
        recording,
    )
    start_column, stop_column = st.columns(2)
    with start_column:
        if st.button(
            "Start recording",
            disabled=start_disabled,
            use_container_width=True,
            type="primary",
        ):
            st.session_state.real_recognition_result = None
            st.session_state.speech_audio = None
            countdown = st.empty()
            for number in (3, 2, 1):
                status_placeholder.markdown(
                    status_badge("COUNTDOWN"), unsafe_allow_html=True
                )
                countdown.markdown(
                    f'<div class="countdown"><span>{number}</span>'
                    '<small>Get ready to sign</small></div>',
                    unsafe_allow_html=True,
                )
                time.sleep(1)
            countdown.empty()
            if video_processor.start_recording():
                if st.session_state.recording_started_at is None:
                    st.session_state.recording_started_at = time.monotonic()
            else:
                st.session_state.recording_started_at = None
            st.rerun()
    with stop_column:
        if st.button(
            "Stop recording",
            disabled=stop_disabled,
            use_container_width=True,
            type="secondary",
        ):
            status_placeholder.markdown(
                status_badge("PROCESSING"), unsafe_allow_html=True
            )
            video_processor.stop_recording()
            st.session_state.recording_started_at = None
            st.rerun()


def render_translation_result() -> None:
    """Render the newest deduplicated SgSL translation and voice output."""
    result = st.session_state.real_recognition_result
    result_status = "RESULT" if result else "READY"
    translated_text = (
        html.escape(str(result["text"]))
        if result
        else "Your translated message will appear here after you record a sign."
    )
    message_class = "message-text" if result else "placeholder-text"

    st.markdown('<div class="section-label">Translation</div>', unsafe_allow_html=True)
    st.markdown(status_badge(result_status), unsafe_allow_html=True)
    st.markdown(
        '<div class="translation-card">'
        f'<div class="{message_class}">{translated_text}</div></div>',
        unsafe_allow_html=True,
    )
    if result:
        intent = html.escape(str(result["intent"]).replace("_", " ").title())
        st.markdown(
            '<div class="result-meta">'
            f'<span>{intent}</span>'
            f'<span>Confidence {float(result["confidence"]):.0%}</span></div>',
            unsafe_allow_html=True,
        )
        if result["intent"] == "ALLERGY" or result["critical"]:
            st.markdown(
                '<div class="critical-note"><strong>Food allergy alert</strong>'
                '<span>Please take extra care when preparing this order.</span></div>',
                unsafe_allow_html=True,
            )

    if st.button(
        "Play translation aloud",
        disabled=result is None,
        use_container_width=True,
        type="secondary",
        help="Recognise a message first." if result is None else None,
    ):
        st.session_state.voice_error = None
        try:
            with st.spinner("Preparing audio…"):
                voice_id = get_voice_id()
                st.session_state.speech_audio = cached_speech(
                    result["text"], voice_id
                )
        except VoiceConfigurationError:
            st.session_state.speech_audio = None
            st.session_state.voice_error = (
                "Voice playback is not configured. You can still show the written translation."
            )
        except (VoiceOutputError, ValueError):
            st.session_state.speech_audio = None
            st.session_state.voice_error = (
                "Voice playback is unavailable right now. Please use the written translation."
            )

    if st.session_state.voice_error:
        st.warning(st.session_state.voice_error)
    elif st.session_state.speech_audio is not None:
        st.audio(st.session_state.speech_audio, format="audio/mpeg", autoplay=True)


def render_staff_panel() -> None:
    """Render existing staff transcription and mapped response videos."""
    transcript = st.session_state.hawker_transcript
    reply_intent = st.session_state.hawker_reply_intent
    sign_video = SIGN_VIDEO_MAP.get(reply_intent) if reply_intent else None

    st.header("Staff")
    st.markdown(
        '<div class="direction-note">Staff response</div>',
        unsafe_allow_html=True,
    )
    if sign_video:
        st.video(sign_video, autoplay=True)
    else:
        placeholder = (
            "No signed response is available for this reply."
            if transcript
            else "Staff response will appear here"
        )
        st.markdown(
            '<div class="video-placeholder"><div>'
            f'{html.escape(placeholder)}</div></div>',
            unsafe_allow_html=True,
        )

    staff_status = (
        "ERROR"
        if st.session_state.stt_error
        else "RESULT"
        if transcript
        else "READY"
    )
    st.markdown(status_badge(staff_status), unsafe_allow_html=True)
    if transcript:
        st.markdown(
            '<div class="transcript-card"><div class="result-label">Staff said</div>'
            f'<div>{html.escape(transcript)}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Record the staff member's reply to display it as text.")

    if st.session_state.stt_error:
        st.warning(st.session_state.stt_error)

    hawker_audio = st.audio_input("Record staff reply")
    if hawker_audio is None:
        return
    audio_bytes = hawker_audio.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    if audio_hash == st.session_state.hawker_audio_hash:
        return

    if audio_hash == st.session_state.hawker_failed_audio_hash:
        if not st.button("Retry transcription", key="retry_transcription"):
            return

    st.session_state.stt_error = None
    try:
        with st.spinner("Turning speech into text…"):
            st.session_state.hawker_transcript = transcribe_audio(audio_bytes)
            st.session_state.hawker_reply_intent = map_hawker_reply(
                st.session_state.hawker_transcript
            )
        st.session_state.hawker_audio_hash = audio_hash
        st.session_state.hawker_failed_audio_hash = None
        st.rerun()
    except VoiceConfigurationError:
        st.session_state.hawker_transcript = None
        st.session_state.hawker_reply_intent = None
        st.session_state.stt_error = (
            "Speech transcription is not configured. Please add the ElevenLabs API key and try again."
        )
    except SpeechTranscriptionError:
        st.session_state.hawker_transcript = None
        st.session_state.hawker_reply_intent = None
        st.session_state.stt_error = (
            "We couldn't understand that recording. Please try again."
        )
    st.session_state.hawker_failed_audio_hash = audio_hash
    st.rerun()


st.set_page_config(
    page_title="Clear Conversation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Accessibility-focused visual system
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #67645f;
        --surface: #fffaf0;
        --background: #eee7d9;
        --accent: #416b8b;
        --accent-soft: #e7eef3;
        --border: #d7cdbd;
        --danger: #b42318;
        --danger-soft: #fff1f0;
    }
    #MainMenu, footer, [data-testid="stHeader"],
    [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;
    }
    .stApp { background: var(--background); color: var(--ink); }
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCaptionContainer"] p {
        color: var(--ink);
    }
    .block-container {
        max-width: 1240px;
        padding: 1.7rem 2rem 3rem;
    }
    .app-header {
        margin-bottom: 1.8rem;
        max-width: 760px;
    }
    .app-name {
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 760;
        letter-spacing: -.035em;
        line-height: 1.08;
        margin: 0 0 .55rem;
    }
    .app-subtitle {
        color: var(--muted);
        font-size: 1.08rem;
        line-height: 1.55;
        margin: 0;
    }
    h2 {
        color: var(--ink);
        font-size: clamp(1.3rem, 2.2vw, 1.65rem) !important;
        letter-spacing: -.015em;
        margin-bottom: .2rem !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 22px rgba(56, 47, 36, .07);
        padding: .35rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        height: 100%;
    }
    .message-card {
        min-height: 148px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        background: #fbfcfe;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.35rem 1.45rem;
        margin: .9rem 0 .65rem;
    }
    .message-text {
        color: var(--ink);
        font-size: clamp(1.8rem, 3.2vw, 2.55rem);
        font-weight: 720;
        line-height: 1.18;
        letter-spacing: -.025em;
    }
    .placeholder-text {
        color: var(--muted);
        font-size: 1.15rem;
        font-weight: 520;
        line-height: 1.45;
    }
    .result-label {
        color: var(--muted);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .08em;
        margin-bottom: .55rem;
        text-transform: uppercase;
    }
    .result-meta {
        color: var(--muted);
        display: flex;
        font-size: .9rem;
        gap: 1rem;
        justify-content: space-between;
        margin-bottom: .7rem;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: .35rem .7rem;
        font-size: .78rem;
        font-weight: 750;
        letter-spacing: .025em;
    }
    .status-ready { background: #eef0f3; color: #344054; }
    .status-success, .status-working {
        background: var(--accent-soft);
        color: #174ea6;
    }
    .status-recording, .status-error {
        background: var(--danger-soft);
        color: var(--danger);
    }
    .recording-notice {
        align-items: center;
        background: var(--danger-soft);
        border-radius: 10px;
        color: #7a271a;
        display: flex;
        font-size: .92rem;
        font-weight: 650;
        gap: .65rem;
        margin: .65rem 0;
        padding: .7rem .8rem;
    }
    .recording-notice span {
        background: #d92d20;
        border-radius: 50%;
        display: inline-block;
        height: .65rem;
        width: .65rem;
    }
    .countdown {
        color: var(--accent);
        display: flex;
        flex-direction: column;
        font-weight: 780;
        padding: .8rem 0;
        text-align: center;
    }
    .countdown span { font-size: 4rem; line-height: 1; }
    .countdown small {
        color: var(--muted);
        font-size: .9rem;
        font-weight: 600;
        margin-top: .45rem;
    }
    .critical-note {
        background: var(--danger-soft);
        border-left: 4px solid var(--danger);
        border-radius: 10px;
        color: #7a271a;
        margin: .8rem 0;
        padding: .8rem .9rem;
    }
    .critical-note span {
        display: block;
        margin-top: .2rem;
    }
    .section-label {
        color: var(--ink);
        font-size: 1.28rem;
        font-weight: 760;
        margin: 1.6rem 0 .55rem;
    }
    .translation-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: 0 8px 22px rgba(56, 47, 36, .07);
        margin: .65rem 0 .6rem;
        min-height: 130px;
        padding: 1.5rem 1.7rem;
        display: flex;
        align-items: center;
    }
    .video-placeholder {
        align-items: center;
        aspect-ratio: 16 / 9;
        background: #f4eddf;
        border: 1px solid var(--border);
        border-radius: 13px;
        color: var(--muted);
        display: flex;
        justify-content: center;
        margin-bottom: .8rem;
        min-height: 260px;
        padding: 2rem;
        text-align: center;
    }
    .transcript-card {
        background: #f8f1e5;
        border: 1px solid var(--border);
        border-radius: 12px;
        color: var(--ink);
        font-size: 1.15rem;
        line-height: 1.45;
        margin: .7rem 0;
        min-height: 92px;
        padding: 1rem 1.1rem;
    }
    .direction-note {
        color: var(--muted);
        font-size: .95rem;
        margin: -.2rem 0 .75rem;
    }
    div.stButton > button {
        min-height: 2.85rem;
        border-radius: 10px;
        font-size: .96rem;
        font-weight: 680;
    }
    div.stButton > button[kind="primary"] {
        color: #ffffff;
        background: var(--accent);
        border-color: var(--accent);
    }
    div.stButton > button[kind="secondary"] {
        color: var(--ink);
        background: var(--surface);
        border-color: var(--border);
    }
    div.stButton > button:disabled {
        color: #667085 !important;
        background: #f2f4f7 !important;
        border-color: #d0d5dd !important;
        opacity: 1 !important;
    }
    [data-testid="stVideo"] video,
    video {
        border-radius: 12px;
        aspect-ratio: 16 / 9;
        object-fit: cover;
    }
    [data-testid="stAudioInput"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

initialize_state()
recognition_session_key = st.session_state.recognition_session_key
live_predictor = get_live_predictor(recognition_session_key)
prediction_store = get_prediction_store(recognition_session_key)

st.markdown(
    '<div class="app-header"><div class="app-name">Clear Conversation</div>'
    '<p class="app-subtitle">A simple two-way communication aid for signers '
    'and staff.</p></div>',
    unsafe_allow_html=True,
)

poll_for_prediction(prediction_store)

signing_column, reply_column = st.columns(2, gap="large")

with signing_column:
    with st.container(border=True):
        render_signer_panel(live_predictor, prediction_store)

with reply_column:
    with st.container(border=True):
        render_staff_panel()

render_translation_result()
