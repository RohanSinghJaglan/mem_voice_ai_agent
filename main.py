"""
Voice-Controlled Local AI Agent — Gradio Interface.

Main application entry point. Provides a web UI for voice input (microphone
or file upload), real-time transcription via Groq Whisper, intent classification
via Gemini Flash, and tool execution with optional Human-in-the-Loop confirmation
for destructive operations (file creation, code generation).

Usage:
    python main.py
    → Open http://localhost:7860 in your browser.
"""

import json
import logging

import gradio as gr
from stt import transcribe_audio
from intent import classify_intent
from tools import execute_intent
from config import GRADIO_SERVER_PORT, OUTPUT_DIR

logger = logging.getLogger("voice-agent.ui")

# ── Confidence threshold for low-confidence warning ─────────────────
LOW_CONFIDENCE_THRESHOLD = 0.5


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


def process_audio(audio_path: str, human_in_loop: bool, pending_state: dict):
    """
    Full pipeline: Audio → STT → Intent Classification → Tool Execution.

    Implements graceful degradation at every stage — if STT fails, the
    pipeline stops with a clear error. If intent classification fails,
    it defaults to chat. If tool execution fails, it returns an error
    message without crashing the UI.

    Args:
        audio_path: Path to the uploaded/recorded audio file.
        human_in_loop: Whether to require confirmation before file operations.
        pending_state: gr.State dict for storing pending intent data.

    Returns:
        Tuple of (transcription, intent_display, action_message, result_output,
                  confirm_button_visible, pending_state).
    """
    # ── Guard: No audio provided ─────────────────────────────────
    if not audio_path:
        return (
            "No audio provided. Please record or upload an audio file.",
            "", "", "",
            gr.update(visible=False),
            pending_state,
        )

    # ── Stage 1: Speech-to-Text ──────────────────────────────────
    try:
        transcription = transcribe_audio(audio_path)
    except Exception as e:
        logger.error("STT exception: %s", str(e))
        transcription = f"[STT ERROR] Could not transcribe audio: {str(e)}"

    if transcription.startswith("[STT ERROR]"):
        return (
            transcription,
            "STT failed — cannot proceed with intent classification.",
            "Pipeline stopped at STT stage.",
            "",
            gr.update(visible=False),
            pending_state,
        )

    # ── Stage 2: Intent Classification ───────────────────────────
    try:
        intent_data = classify_intent(transcription)
    except (ValueError, Exception) as e:
        logger.warning("Intent classification failed, defaulting to chat: %s", str(e)[:100])
        intent_data = {
            "intent": "chat",
            "parameters": {},
            "confidence": 0.0,
            "reasoning": f"Intent classification failed ({str(e)[:100]}). Defaulting to chat.",
        }

    # ── Low confidence warning ───────────────────────────────────
    confidence = intent_data.get("confidence", 0.0)
    confidence_warning = ""
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        confidence_warning = f"⚠ Low confidence ({confidence:.0%}) — result may be inaccurate.\n"

    intent_display = (
        f"{confidence_warning}"
        f"Intent: {intent_data['intent']}\n"
        f"Confidence: {confidence:.0%}\n"
        f"Reasoning: {intent_data['reasoning']}\n"
        f"Parameters: {json.dumps(intent_data.get('parameters', {}), indent=2)}"
    )

    # ── Stage 3: Execution (with optional Human-in-the-Loop) ────
    intent = intent_data.get("intent", "chat")
    requires_confirmation = intent in ("create_file", "write_code")

    if human_in_loop and requires_confirmation:
        pending_state = {
            "intent_data": intent_data,
            "original_text": transcription,
        }

        action_msg = (
            f"⚠ Human-in-the-Loop: Intent '{intent}' requires confirmation.\n"
            f"Review the intent details and click 'Confirm Execute' to proceed.\n"
            f"Parameters: {json.dumps(intent_data.get('parameters', {}), indent=2)}"
        )

        return (
            transcription,
            intent_display,
            action_msg,
            "Awaiting confirmation...",
            gr.update(visible=True),
            pending_state,
        )

    # Execute immediately
    try:
        result_msg, output_content = execute_intent(intent_data, transcription)
    except Exception as e:
        logger.error("Execution error: %s", str(e))
        result_msg = f"✗ Execution error: {str(e)}"
        output_content = ""

    return (
        transcription,
        intent_display,
        result_msg,
        output_content,
        gr.update(visible=False),
        pending_state,
    )


def confirm_execution(pending_state: dict):
    """
    Execute a previously stored intent after Human-in-the-Loop confirmation.

    Called when the user clicks 'Confirm Execute' after reviewing the
    classified intent and its parameters.

    Args:
        pending_state: Dictionary containing intent_data and original_text.

    Returns:
        Tuple of (action_message, result_output, confirm_button_visible, cleared_state).
    """
    if not pending_state or "intent_data" not in pending_state:
        return (
            "No pending action to confirm.",
            "",
            gr.update(visible=False),
            {},
        )

    try:
        intent_data = pending_state["intent_data"]
        original_text = pending_state["original_text"]
        result_msg, output_content = execute_intent(intent_data, original_text)
    except Exception as e:
        logger.error("Confirmed execution error: %s", str(e))
        result_msg = f"✗ Execution error: {str(e)}"
        output_content = ""

    return (
        f"✓ Confirmed and executed.\n{result_msg}",
        output_content,
        gr.update(visible=False),
        {},
    )


# ═══════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════


def build_ui() -> gr.Blocks:
    """
    Build the Gradio Blocks interface for the voice agent.

    Returns:
        Configured gr.Blocks application ready to launch.
    """
    with gr.Blocks(
        title="Voice-Controlled AI Agent",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
            .main-header { text-align: center; margin-bottom: 8px; }
            .model-info { text-align: center; color: #6b7280; font-size: 0.9em; margin-bottom: 16px; }
            .confirm-btn { margin-top: 8px; }
        """,
    ) as app:

        # ── Header ──────────────────────────────────────────────
        gr.Markdown(
            "# 🎙️ Voice-Controlled Local AI Agent",
            elem_classes="main-header",
        )
        gr.Markdown(
            "**STT:** Groq Whisper Large V3 &nbsp;|&nbsp; "
            "**Intent:** Gemini 2.5 Flash &nbsp;|&nbsp; "
            "**Code Gen:** Gemini 2.5 Pro &nbsp;|&nbsp; "
            "**Sandbox:** `output/`",
            elem_classes="model-info",
        )

        # ── State for Human-in-the-Loop ─────────────────────────
        pending_state = gr.State({})

        # ── Input Section ────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### 📥 Input")
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or upload audio",
            )
            with gr.Row():
                hitl_checkbox = gr.Checkbox(
                    label="🔒 Human-in-the-Loop confirmation (for file writes)",
                    value=True,
                    scale=3,
                )
                process_btn = gr.Button(
                    "🚀 Process Audio",
                    variant="primary",
                    scale=1,
                )

        # ── Output Section (2-column grid) ───────────────────────
        with gr.Group():
            gr.Markdown("### 📤 Output")
            with gr.Row():
                transcription_box = gr.Textbox(
                    label="📝 Transcription",
                    lines=3,
                    interactive=False,
                )
                intent_box = gr.Textbox(
                    label="🎯 Intent Classification",
                    lines=3,
                    interactive=False,
                )
            with gr.Row():
                action_box = gr.Textbox(
                    label="⚡ Action Taken",
                    lines=3,
                    interactive=False,
                )
                result_box = gr.Textbox(
                    label="📄 Result Output",
                    lines=8,
                    interactive=False,
                )

        # ── Confirm Button (hidden by default) ──────────────────
        confirm_btn = gr.Button(
            "✅ Confirm Execute",
            variant="secondary",
            visible=False,
            elem_classes="confirm-btn",
        )

        # ── Wire up events ──────────────────────────────────────
        process_btn.click(
            fn=process_audio,
            inputs=[audio_input, hitl_checkbox, pending_state],
            outputs=[
                transcription_box,
                intent_box,
                action_box,
                result_box,
                confirm_btn,
                pending_state,
            ],
        )

        confirm_btn.click(
            fn=confirm_execution,
            inputs=[pending_state],
            outputs=[
                action_box,
                result_box,
                confirm_btn,
                pending_state,
            ],
        )

        # ── Command Reference Table ─────────────────────────────
        gr.Markdown("---")
        gr.Markdown(
            """### 📋 Supported Commands

| Intent | Example Phrase | Action |
|--------|---------------|--------|
| `create_file` | "Create a text file called notes.txt with hello world" | Creates file in `output/` |
| `write_code` | "Write a Python function that reverses a string" | Generates code via Gemini Pro → `output/` |
| `summarize` | "Summarize: Machine learning is a subset of AI..." | Returns structured bullet-point summary |
| `chat` | "What is the difference between RAM and ROM?" | Conversational response (5-turn memory) |
"""
        )

        gr.Markdown(
            f"**Output directory:** `{OUTPUT_DIR}`\n\n"
            "All generated files are sandboxed to this directory via `safe_path()`."
        )

    return app


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=GRADIO_SERVER_PORT,
        share=False,
        show_error=True,
    )
