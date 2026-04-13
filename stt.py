"""
Speech-to-Text module using Groq Whisper Large V3 API.

Hardware constraint: local Whisper Large V3 requires 10GB+ VRAM and produces
15-30s latency on standard hardware (e.g., Apple M-series or GTX 1660),
making real-time voice interaction impossible. Groq's LPU inference engine
delivers <1s latency with identical WER accuracy, making it the correct
production choice for a voice-first agent.

API Reference: https://console.groq.com/docs/speech-text
"""

import logging
import os

from groq import Groq
from config import (
    GROQ_API_KEY,
    GROQ_STT_MODEL,
    ALLOWED_AUDIO_EXTENSIONS,
    MAX_AUDIO_SIZE_BYTES,
    MAX_AUDIO_SIZE_MB,
    API_TIMEOUT_SECONDS,
)

logger = logging.getLogger("voice-agent.stt")

# Initialize Groq client once at module level to reuse connection pooling
_client = Groq(api_key=GROQ_API_KEY, timeout=API_TIMEOUT_SECONDS)


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file to text using Groq Whisper Large V3.

    Accepts .wav, .mp3, and .m4a files. The Groq API handles all
    audio preprocessing (resampling, normalization) server-side,
    so no local ffmpeg dependency is required.

    Args:
        audio_path: Absolute or relative path to the audio file.

    Returns:
        Transcribed text string. Returns an error message string
        (not an exception) on failure to support graceful UI degradation.

    Example:
        >>> text = transcribe_audio("recording.wav")
        >>> print(text)
        "Create a Python file called hello.py with a main function"
    """
    # ── Validate file exists ─────────────────────────────────────
    if not audio_path or not os.path.isfile(audio_path):
        logger.warning("Audio file not found: %s", audio_path)
        return f"[STT ERROR] Audio file not found: {audio_path}"

    # ── Validate file extension ──────────────────────────────────
    _, ext = os.path.splitext(audio_path)
    if ext.lower() not in ALLOWED_AUDIO_EXTENSIONS:
        return (
            f"[STT ERROR] Unsupported audio format '{ext}'. "
            f"Accepted: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )

    # ── Validate file size (Groq limit: 25MB) ────────────────────
    file_size = os.path.getsize(audio_path)
    if file_size > MAX_AUDIO_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return (
            f"[STT ERROR] Audio file too large ({size_mb:.1f}MB). "
            f"Groq API limit is {MAX_AUDIO_SIZE_MB}MB. "
            f"Please use a shorter recording."
        )

    if file_size == 0:
        return "[STT ERROR] Audio file is empty (0 bytes)."

    # ── Call Groq Whisper API ────────────────────────────────────
    try:
        logger.info("Transcribing audio: %s (%.1f KB)", os.path.basename(audio_path), file_size / 1024)

        with open(audio_path, "rb") as audio_file:
            transcription = _client.audio.transcriptions.create(
                model=GROQ_STT_MODEL,
                file=audio_file,
                response_format="text",
                language="en",
            )

        # Groq returns the text directly when response_format="text"
        result = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()

        if not result:
            return "[STT ERROR] Transcription returned empty. Audio may be silent or too short."

        logger.info("Transcription successful: %d chars", len(result))
        return result

    except Exception as e:
        error_msg = str(e)
        logger.error("Groq transcription failed: %s", error_msg)

        if "429" in error_msg or "rate" in error_msg.lower():
            return "[STT ERROR] Groq rate limit hit. Please wait a moment and try again."
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            return "[STT ERROR] Groq API timed out after 30s. Check your connection."
        if "401" in error_msg or "auth" in error_msg.lower():
            return "[STT ERROR] Groq authentication failed. Check your GROQ_API_KEY."

        return f"[STT ERROR] Groq transcription failed: {error_msg}"
