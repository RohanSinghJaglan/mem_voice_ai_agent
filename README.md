# 🎙️ Voice-Controlled Local AI Agent

A production-grade voice-controlled AI agent that transcribes spoken commands, classifies intent, and executes actions — all through a clean Gradio web interface. Built for the Mem0 internship assignment.

## Demo

> 🎥 [Video demo link — to be added after recording]

---

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌────────────┐     ┌────────────┐
│  Mic/Upload  │────▶│  Groq Whisper     │────▶│  Gemini 1.5      │────▶│   Tools    │────▶│ Gradio UI  │
│  (Audio)     │     │  Large V3 (STT)  │     │  Flash (Intent)  │     │ (Execute)  │     │ (Display)  │
└──────────────┘     └──────────────────┘     └──────────────────┘     └────────────┘     └────────────┘
                          ~0.8s                     ~0.4s                                        
                       Groq LPU API            Vertex AI API          create_file              
                                                                      write_code → Gemini Pro  
                                                                      summarize → Gemini Flash 
                                                                      chat → Gemini Flash      
```

### Pipeline Flow

1. **Audio Input** → User records via microphone or uploads a file (.wav, .mp3, .m4a)
2. **STT** → Groq Whisper Large V3 transcribes audio to text (<1s latency)
3. **Intent Classification** → Gemini 1.5 Flash parses text into structured intent + parameters
4. **Human-in-the-Loop** → Optional confirmation step for destructive operations (file writes)
5. **Tool Execution** → Routed to the correct tool function based on classified intent
6. **Output** → Results displayed in the Gradio UI; files written to sandboxed `output/` directory

---

## Hardware & Model Decisions

### Why Groq API for STT (not local Whisper)

Running Whisper Large V3 locally is impractical for a real-time voice agent:

| Factor | Local Whisper Large V3 | Groq API |
|--------|----------------------|----------|
| **VRAM Required** | 10GB+ (fp16) | 0 (cloud) |
| **Latency** | 15-30s on M1/M2 MacBook | <1s |
| **CPU Fallback** | 45-90s (unusable) | N/A |
| **Accuracy (WER)** | Identical | Identical |
| **Setup** | ffmpeg + torch + model download (~3GB) | API key only |

**Verdict:** Local Whisper makes real-time voice interaction impossible on standard hardware. The Groq LPU delivers the same Whisper Large V3 model at 50-100x faster inference with zero local compute. For a voice-first UX, API latency IS the product.

### Why Gemini 1.5 Flash for Intent Classification

- **Latency:** ~400ms for structured JSON output — critical for responsive voice UX
- **JSON reliability:** Flash consistently produces well-formed JSON with the right prompt engineering
- **Cost:** Vertex AI free tier provides $300 in credits; Flash is ~10x cheaper than Pro per token
- **Accuracy:** Intent classification is a constrained task (4 categories) — Flash's reasoning is sufficient

### Why Gemini 1.5 Pro for Code Generation

- **Deeper reasoning:** Code generation requires understanding of best practices, edge cases, type systems
- **Quality bar:** Pro produces significantly better docstrings, error handling, and idiomatic code
- **Latency acceptable:** Human-in-the-Loop confirmation means the user is already pausing to review — Pro's 2-4s latency is invisible in this flow
- **When it matters:** Pro is ONLY invoked for `write_code` intent, not for every request

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/keys) (free tier available)
- A Google Cloud project with Vertex AI API enabled
- A GCP service account JSON key (or `gcloud auth application-default login`)

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/RohanSinghJaglan/mem_local_ai_model.git
cd mem_local_ai_model

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your actual keys:
#   GROQ_API_KEY=gsk_...
#   GCP_PROJECT_ID=your-project-id
#   GOOGLE_APPLICATION_CREDENTIALS=./service-account.json

# 5. Place your GCP service account key
# Download from: GCP Console → IAM → Service Accounts → Keys
# Save as service-account.json in the project root

# 6. Launch the agent
python main.py
# → Open http://localhost:7860 in your browser
```

### Alternative: Use Application Default Credentials (no service account file)

```bash
gcloud auth application-default login
# Then remove GOOGLE_APPLICATION_CREDENTIALS from .env
```

---

## Supported Intents

| Intent | Example Phrase | Action | Model Used |
|--------|---------------|--------|------------|
| `create_file` | "Create a text file called notes.txt with hello world" | Writes file to `output/notes.txt` | — |
| `write_code` | "Write a Python function that reverses a string" | Generates code → `output/generated.py` | Gemini 1.5 Pro |
| `summarize` | "Summarize: Machine learning is a subset of AI that focuses on learning from data" | Returns bullet-point summary | Gemini 1.5 Flash |
| `chat` | "What is the difference between RAM and ROM?" | Conversational response with 5-turn memory | Gemini 1.5 Flash |

---

## Bonus Features Implemented

### 🔒 Human-in-the-Loop Confirmation
Destructive operations (`create_file`, `write_code`) require explicit user confirmation when the checkbox is enabled. The classified intent and parameters are displayed for review before any file is written.

### 🛡️ Graceful Degradation
Every pipeline stage has independent error handling:
- **STT fails?** → Clear error message, pipeline stops cleanly
- **Intent classification fails?** → Falls back to `chat` intent automatically
- **Tool execution fails?** → Returns error string, UI never crashes
- **Quota exceeded?** → Specific message with retry guidance

### 🔗 Compound Command Detection
The intent classifier detects multi-action commands (e.g., "create a file and then write code in it") and sets a `compound` flag in parameters, classifying based on the primary action.

### 💬 Session Chat History
The `chat` intent maintains the last 5 conversation turns in-memory, enabling coherent multi-turn dialogue within a session. History resets on server restart.

### 📁 Path Traversal Protection
All file operations use `safe_path()` which strips `../`, null bytes, and leading slashes, then verifies the resolved path is within `output/`. No file can ever be written outside the sandbox.

---

## Challenges & Solutions

### 1. JSON Parsing from LLM Responses
**Challenge:** Gemini models frequently wrap JSON responses in markdown code fences (` ```json ... ``` `) despite explicit prompt instructions not to.

**Solution:** Built `_strip_markdown_json()` in `intent.py` that uses regex to strip code fences before JSON parsing. This handles both ` ```json ` and bare ` ``` ` fencing patterns. The function is called on every LLM response before `json.loads()`.

### 2. Vertex AI Authentication Complexity
**Challenge:** GCP authentication has three modes (service account JSON, Application Default Credentials, workload identity) and each fails differently — sometimes silently returning empty responses.

**Solution:** `config.py` validates the service account file exists at startup and prints a specific warning with the `gcloud auth application-default login` fallback command. The intent and tools modules surface quota errors (`429`) separately from auth errors.

### 3. Code Block Stripping for Generated Code
**Challenge:** When Gemini Pro generates code, it wraps the output in ` ```python ... ``` ` blocks. Writing this directly to a file creates invalid syntax with the fence markers included.

**Solution:** `_strip_code_blocks()` in `tools.py` uses a two-pass approach — first trying to match the full ` ```lang\n...\n``` ` pattern, then falling back to stripping leading/trailing fences. This handles edge cases like nested code blocks in generated code.

### 4. Gradio State Management for Human-in-the-Loop
**Challenge:** Gradio's event-driven model doesn't natively support "pause and wait for confirmation" workflows. The `process_audio` function needs to return partial results, store state, and resume later.

**Solution:** Used `gr.State` to persist the classified intent between button clicks. The `process_audio` function stores `intent_data` in state and shows the confirm button; `confirm_execution` reads from state and executes. The confirm button's visibility is toggled via `gr.update(visible=...)`.

---

## Project Structure

```
.
├── main.py              # Gradio UI and pipeline orchestration
├── stt.py               # Groq Whisper Large V3 transcription
├── intent.py            # Gemini Flash intent classification
├── tools.py             # Tool execution (file ops, code gen, chat)
├── config.py            # Environment config and validation
├── output/              # Sandboxed directory for generated files
│   └── .gitkeep
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Git exclusions
└── README.md            # This file
```

---

## License

MIT
