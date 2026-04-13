"""
Configuration module for Voice-Controlled Local AI Agent.

Single source of truth for all environment variables, API keys,
and project-wide settings. Validates required configuration on
import and raises clear errors for missing values.
"""

import os
import sys
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


def _require_env(var_name: str) -> str:
    """
    Retrieve a required environment variable or exit with a clear error.

    Args:
        var_name: Name of the environment variable to retrieve.

    Returns:
        The value of the environment variable.

    Raises:
        SystemExit: If the variable is not set or is empty.
    """
    value = os.getenv(var_name)
    if not value:
        print(
            f"\n[CONFIG ERROR] Required environment variable '{var_name}' is not set.\n"
            f"  → Copy .env.example to .env and fill in your values:\n"
            f"    cp .env.example .env\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return value


# ── API Keys ────────────────────────────────────────────────────────
GROQ_API_KEY: str = _require_env("GROQ_API_KEY")
GEMINI_API_KEY: str = _require_env("GEMINI_API_KEY")

# ── Model Configuration ─────────────────────────────────────────────
GROQ_STT_MODEL: str = "whisper-large-v3"
GEMINI_FLASH_MODEL: str = "gemini-2.5-flash"
GEMINI_PRO_MODEL: str = "gemini-2.5-pro"

# ── Application Settings ────────────────────────────────────────────
OUTPUT_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MAX_CHAT_HISTORY: int = 5
ALLOWED_AUDIO_EXTENSIONS: tuple = (".wav", ".mp3", ".m4a")
GRADIO_SERVER_PORT: int = 7860

# ── Ensure output directory exists ──────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
