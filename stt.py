"""
Speech-to-Text module using Groq Whisper Large V3 API.

Hardware constraint: local Whisper Large V3 requires 10GB+ VRAM and produces
15-30s latency on standard hardware (e.g., Apple M-series or GTX 1660),
making real-time voice interaction impossible. Groq's LPU inference engine
delivers <1s latency with identical WER accuracy, making it the correct
production choice for a voice-first agent.

API Reference: https://console.groq.com/docs/speech-text
"""

import os
from groq import Groq
from config import GROQ_API_KEY, GROQ_STT_MODEL, ALLOWED_AUDIO_EXTENSIONS


# Initialize Groq client once at module level to reuse connection pooling
_client = Groq(api_key=GROQ_API_KEY)


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
        return f"[STT ERROR] Audio file not found: {audio_path}"

    # ── Validate file extension ──────────────────────────────────
    _, ext = os.path.splitext(audio_path)
    if ext.lower() not in ALLOWED_AUDIO_EXTENSIONS:
        return (
            f"[STT ERROR] Unsupported audio format '{ext}'. "
            f"Accepted: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )

    # ── Call Groq Whisper API ────────────────────────────────────
    try:
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

        return result

    except Exception as e:
        return f"[STT ERROR] Groq transcription failed: {str(e)}"
