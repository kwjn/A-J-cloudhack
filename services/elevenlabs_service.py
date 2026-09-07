"""Small, reusable ElevenLabs text-to-speech integration."""

import os
from typing import Optional

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs


TTS_MODEL_ID = "eleven_flash_v2_5"
TTS_OUTPUT_FORMAT = "mp3_44100_128"


class VoiceConfigurationError(RuntimeError):
    """Raised when required local ElevenLabs configuration is missing."""


class VoiceOutputError(RuntimeError):
    """Raised when ElevenLabs cannot produce usable audio."""


def get_voice_id() -> str:
    """Return the configured ElevenLabs voice ID."""
    load_dotenv()
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
    if not voice_id:
        raise VoiceConfigurationError(
            "Set ELEVENLABS_VOICE_ID in the local .env file."
        )
    return voice_id


def speak_text(text: str, voice_id: Optional[str] = None) -> bytes:
    """Generate in-memory MP3 audio for arbitrary English text."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text to speak must not be empty.")

    load_dotenv()
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise VoiceConfigurationError(
            "Set ELEVENLABS_API_KEY in the local .env file."
        )

    selected_voice_id = voice_id.strip() if voice_id else get_voice_id()
    if not selected_voice_id:
        raise VoiceConfigurationError(
            "Set ELEVENLABS_VOICE_ID in the local .env file."
        )

    try:
        audio_chunks = ElevenLabs(api_key=api_key).text_to_speech.convert(
            voice_id=selected_voice_id,
            text=text.strip(),
            model_id=TTS_MODEL_ID,
            output_format=TTS_OUTPUT_FORMAT,
        )
        audio_bytes = b"".join(audio_chunks)
    except Exception as error:
        # Do not pass SDK/network details to the UI because they may contain
        # request metadata. The original exception remains available as cause.
        raise VoiceOutputError("ElevenLabs could not generate audio.") from error

    if not audio_bytes:
        raise VoiceOutputError("ElevenLabs returned no audio.")
    return audio_bytes
