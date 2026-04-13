"""
Tool execution module for the Voice-Controlled AI Agent.

Handles all file operations, code generation, summarization, and chat.
SAFETY: Every file write is sandboxed to the output/ directory via safe_path()
to prevent path traversal attacks. No file is ever written outside output/.
"""

import os
import re
import google.generativeai as genai
from config import (
    GEMINI_API_KEY,
    GEMINI_PRO_MODEL,
    GEMINI_FLASH_MODEL,
    OUTPUT_DIR,
    MAX_CHAT_HISTORY,
)


# ── Configure Google AI SDK ─────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)

# ── Chat history stored in-memory (last N turns) ────────────────────
_chat_history: list[dict] = []


def safe_path(filename: str) -> str:
    """
    Construct a safe file path within the output/ directory.

    Strips path traversal sequences (../, ..\\ , leading /) and joins
    the sanitized filename with the OUTPUT_DIR. This is the ONLY function
    that should be used to construct file paths for writing.

    Args:
        filename: Raw filename from user input.

    Returns:
        Absolute path within the output/ directory.

    Example:
        >>> safe_path("../../etc/passwd")
        '/project/output/etc/passwd'  # traversal stripped
        >>> safe_path("hello.py")
        '/project/output/hello.py'
    """
    # Strip leading slashes and path traversal
    sanitized = filename.replace("\\", "/")
    sanitized = re.sub(r"\.\./", "", sanitized)
    sanitized = re.sub(r"\.\.", "", sanitized)
    sanitized = sanitized.lstrip("/")

    # Remove any remaining dangerous patterns
    sanitized = sanitized.replace("\x00", "")  # null bytes

    if not sanitized:
        sanitized = "untitled.txt"

    full_path = os.path.join(OUTPUT_DIR, sanitized)

    # Final safety check: resolved path must be within OUTPUT_DIR
    resolved = os.path.realpath(full_path)
    output_resolved = os.path.realpath(OUTPUT_DIR)
    if not resolved.startswith(output_resolved):
        # If somehow still escaping, force into output root
        full_path = os.path.join(OUTPUT_DIR, os.path.basename(sanitized))

    return full_path


def create_file(filename: str, content: str) -> tuple:
    """
    Create a text file with the given content in the output/ directory.

    Args:
        filename: Name of the file to create (sanitized via safe_path).
        content: Text content to write to the file.

    Returns:
        Tuple of (result_message: str, file_content: str).

    Example:
        >>> result, content = create_file("notes.txt", "Hello World")
        >>> print(result)
        "✓ File created: output/notes.txt (11 bytes)"
    """
    try:
        filepath = safe_path(filename)

        # Create subdirectories if filename includes a path
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        rel_path = os.path.relpath(filepath)
        size = os.path.getsize(filepath)
        return (
            f"✓ File created: {rel_path} ({size} bytes)",
            content,
        )

    except Exception as e:
        return (f"✗ Failed to create file '{filename}': {str(e)}", "")


def _strip_code_blocks(text: str) -> str:
    """
    Remove markdown code fences from LLM-generated code.

    LLMs often wrap code in ```python ... ``` blocks. This extracts
    the raw code content for writing to files.

    Args:
        text: Raw LLM response that may contain code fences.

    Returns:
        Cleaned code string without markdown formatting.
    """
    # Match ```language\n...code...\n``` pattern
    pattern = r"^```(?:\w+)?\s*\n(.*?)\n```\s*$"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)

    # Fallback: just strip leading/trailing fences
    cleaned = re.sub(r"^```\w*\s*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned


def write_code(filename: str, description: str, language: str) -> tuple:
    """
    Generate code using Gemini 1.5 Pro and save it to the output/ directory.

    Uses the Pro model (not Flash) because code generation requires deeper
    reasoning for producing quality docstrings, type hints, and error handling.
    Latency is acceptable here since Human-in-the-Loop confirmation gives
    the model time to generate.

    Args:
        filename: Target filename for the generated code.
        description: Natural language description of what the code should do.
        language: Programming language (python, javascript, etc.).

    Returns:
        Tuple of (result_message: str, generated_code: str).

    Example:
        >>> result, code = write_code("sort.py", "quicksort algorithm", "python")
        >>> print(result)
        "✓ Code generated: output/sort.py (45 lines)"
    """
    if not filename:
        ext_map = {
            "python": ".py", "javascript": ".js", "typescript": ".ts",
            "java": ".java", "c": ".c", "cpp": ".cpp", "go": ".go",
            "rust": ".rs", "ruby": ".rb", "html": ".html", "css": ".css",
        }
        ext = ext_map.get(language.lower(), ".txt")
        filename = f"generated{ext}"

    code_prompt = f"""Generate production-quality {language} code for the following task:

TASK: {description}

REQUIREMENTS — You MUST include ALL of these:
1. Comprehensive docstrings explaining purpose, args, returns, and edge cases
2. Type hints on all function signatures (where the language supports them)
3. Proper error handling with try/except blocks and meaningful error messages
4. Example usage in comments at the bottom of the file
5. Follow {language} community style conventions (PEP 8 for Python, etc.)

OUTPUT: Return ONLY the raw code. No markdown fences, no explanations, no preamble.
Start directly with the code (imports, shebang, or module docstring)."""

    try:
        model = genai.GenerativeModel(GEMINI_PRO_MODEL)

        response = model.generate_content(
            code_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,  # Low temp for deterministic, correct code
                max_output_tokens=4096,
            ),
        )

        code = _strip_code_blocks(response.text)

        # Write to file
        filepath = safe_path(filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)

        rel_path = os.path.relpath(filepath)
        line_count = code.count("\n") + 1
        return (
            f"✓ Code generated: {rel_path} ({line_count} lines, {language})",
            code,
        )

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return ("✗ Gemini API quota exceeded. Please wait and retry.", "")
        return (f"✗ Code generation failed: {error_msg}", "")


def summarize_text(content: str) -> tuple:
    """
    Summarize text using Gemini 1.5 Flash.

    Flash is used here (not Pro) because summarization benefits from
    speed over deep reasoning, and Flash produces structured bullet
    points reliably.

    Args:
        content: Text to summarize.

    Returns:
        Tuple of (result_message: str, summary: str).

    Example:
        >>> result, summary = summarize_text("Machine learning is...")
        >>> print(summary)
        "## Summary\\n- ML is a subset of AI\\n- ..."
    """
    if not content or not content.strip():
        return ("✗ No content provided to summarize.", "")

    summary_prompt = f"""Provide a clear, structured summary of the following text.

FORMAT your response as:
## Summary
- Key point 1
- Key point 2
- Key point 3
...

## Key Takeaways
- Takeaway 1
- Takeaway 2

Keep it concise but comprehensive. Use bullet points for clarity.

TEXT TO SUMMARIZE:
{content}"""

    try:
        model = genai.GenerativeModel(GEMINI_FLASH_MODEL)

        response = model.generate_content(
            summary_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )

        summary = response.text.strip()
        return ("✓ Text summarized successfully.", summary)

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return ("✗ Gemini API quota exceeded. Please wait and retry.", "")
        return (f"✗ Summarization failed: {error_msg}", "")


def chat_response(text: str, history: list = None) -> tuple:
    """
    Generate a conversational response using Gemini 1.5 Flash.

    Maintains the last 5 turns of conversation context for coherent
    multi-turn dialogue. History is stored in-memory and resets on
    server restart.

    Args:
        text: User's chat message.
        history: Optional external history list. If None, uses internal history.

    Returns:
        Tuple of (result_message: str, response_text: str).

    Example:
        >>> result, reply = chat_response("What is the difference between RAM and ROM?")
        >>> print(reply)
        "RAM is volatile memory used for..."
    """
    global _chat_history

    if history is not None:
        _chat_history = history

    # Build context from recent history
    context_parts = []
    for turn in _chat_history[-MAX_CHAT_HISTORY:]:
        context_parts.append(f"User: {turn['user']}")
        context_parts.append(f"Assistant: {turn['assistant']}")

    history_text = "\n".join(context_parts)

    chat_prompt = f"""You are a helpful AI assistant in a voice-controlled agent.
Be concise, accurate, and conversational. If the user asks a technical question,
provide a clear explanation with examples when relevant.

{"CONVERSATION HISTORY:" + chr(10) + history_text + chr(10) if history_text else ""}
USER: {text}

Respond naturally and helpfully."""

    try:
        model = genai.GenerativeModel(GEMINI_FLASH_MODEL)

        response = model.generate_content(
            chat_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,  # Higher temp for natural conversation
                max_output_tokens=1024,
            ),
        )

        reply = response.text.strip()

        # Store in history
        _chat_history.append({"user": text, "assistant": reply})

        # Trim history to max size
        if len(_chat_history) > MAX_CHAT_HISTORY:
            _chat_history = _chat_history[-MAX_CHAT_HISTORY:]

        return ("✓ Chat response generated.", reply)

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return ("✗ Gemini API quota exceeded. Please wait and retry.", "")
        return (f"✗ Chat failed: {error_msg}", "")


def execute_intent(intent_data: dict, original_text: str) -> tuple:
    """
    Route an intent to the correct tool function and execute it.

    This is the main dispatcher that maps classified intents to their
    corresponding tool functions. It extracts parameters from the
    intent data and calls the appropriate function.

    Args:
        intent_data: Dictionary from classify_intent() with intent, parameters, etc.
        original_text: The original transcribed text (used as fallback content).

    Returns:
        Tuple of (result_message: str, output_content: str).

    Example:
        >>> intent = {"intent": "create_file", "parameters": {"filename": "test.txt", "content": "hello"}}
        >>> result, output = execute_intent(intent, "create a file called test.txt with hello")
    """
    intent = intent_data.get("intent", "chat")
    params = intent_data.get("parameters", {})

    try:
        if intent == "create_file":
            filename = params.get("filename", "untitled.txt")
            content = params.get("content", original_text)
            return create_file(filename, content)

        elif intent == "write_code":
            filename = params.get("filename", "")
            description = params.get("description", original_text)
            language = params.get("language", "python")
            return write_code(filename, description, language)

        elif intent == "summarize":
            content = params.get("content", original_text)
            return summarize_text(content)

        elif intent == "chat":
            return chat_response(original_text)

        else:
            # Unknown intent — fall back to chat
            return chat_response(original_text)

    except Exception as e:
        return (f"✗ Tool execution failed: {str(e)}", "")
