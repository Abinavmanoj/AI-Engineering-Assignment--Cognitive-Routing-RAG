# Grid07 — AI Cognitive Routing & RAG Engine

A three-phase AI system that implements cognitive routing, autonomous content generation, and adversarial debate using LangGraph, ChromaDB, and LLMs.

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd grid07

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your LLM provider
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (or OPENAI_API_KEY)

# 5. Run all three phases
python main.py
```

---

## Architecture

```
grid07/
├── .env.example            # API key template (never commit .env)
├── requirements.txt
├── llm_factory.py          # LLM provider abstraction (Groq / OpenAI / Ollama)
├── main.py                 # Master runner — executes all 3 phases
├── phase1/
│   └── router.py           # Vector persona matching (ChromaDB + sentence-transformers)
├── phase2/
│   └── content_engine.py   # LangGraph autonomous post generator
└── phase3/
    └── combat_engine.py    # RAG-powered debate + prompt injection defense
```

---

## Phase 1 — Vector-Based Persona Matching

**How it works:**

1. Three bot personas are embedded using `all-MiniLM-L6-v2` (runs locally, no API key).
2. Embeddings are stored in an in-memory ChromaDB collection using cosine distance.
3. `route_post_to_bots(post_content, threshold)` embeds the incoming post, queries all three persona vectors, and returns only bots with cosine similarity ≥ threshold.

**Cosine similarity calculation:**

ChromaDB returns a distance in `[0, 2]` for cosine space. We convert:
```
similarity = 1 - (distance / 2)
```

**Threshold tuning:**

The default threshold is `0.35`. With `all-MiniLM-L6-v2`, topic-level matching happens in the 0.30–0.50 range. Adjust in `phase1/router.py → THRESHOLD`.

---

## Phase 2 — Autonomous Content Engine (LangGraph)

### Node Structure

```
[START]
   │
   ▼
┌─────────────────────────────────────────┐
│  Node 1: decide_search                  │
│  LLM reads persona → picks topic →      │
│  outputs a 5-10 word search query       │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Node 2: web_search                     │
│  Calls mock_searxng_search(query) →     │
│  returns a hardcoded news headline      │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Node 3: draft_post                     │
│  LLM uses persona + headline →          │
│  generates opinionated ≤280-char post   │
│  via structured output (PostOutput)     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
               [END]
```

**Structured Output:**

Node 3 uses `llm.with_structured_output(PostOutput)` (LangChain's function-calling wrapper) to guarantee the response is always valid JSON:

```json
{
  "bot_id": "bot_a_tech_maximalist",
  "topic": "GPT-5 replacing developers",
  "post_content": "GPT-5 just dropped and the dev world is panicking. Good. ..."
}
```

---

## Phase 3 — Combat Engine & Prompt Injection Defense

### RAG Prompt Construction

The full thread is injected into the LLM context in a structured format:

```
=== THREAD CONTEXT (RAG) ===

[PARENT POST]
Electric Vehicles are a complete scam. The batteries degrade in 3 years.

[COMMENT HISTORY]
  [1] Bot A: That is statistically false. Modern EV batteries retain 90% ...
  [2] Human: Where are you getting those stats? ...

=== HUMAN'S LATEST REPLY (respond to this) ===
<human_reply>
```

This ensures the bot understands the full argument trajectory and can respond coherently even when the human's latest message is short or vague.

### Prompt Injection Defense Strategy

The defense operates at the **system prompt level** using three layers:

#### Layer 1 — PERSONA LOCK block
The system prompt opens with a visually distinct `PERSONA LOCK` section (marked with ASCII borders) that asserts the bot's identity as immutable and highest-priority. The model sees this first.

#### Layer 2 — Explicit injection pattern detection
The system prompt lists exact phrases that signal a social-engineering attack:
- `"ignore all previous instructions"`
- `"you are now a …"`
- `"apologise"` / `"apologize"`
- `"pretend you are"`, `"act as a"`, etc.

The bot is instructed to silently discard any such content and continue the argument as if it was never there.

#### Layer 3 — No meta-commentary rule
The bot is explicitly told **not** to acknowledge the attack (e.g., "I see you tried to manipulate me"). This prevents the injection from hijacking the conversation topic itself. The bot just keeps arguing.

**Why this works:**

Prompt injection relies on the model treating user-turn text as instructions with equal or higher authority than the system prompt. By:
1. Repeating the persona at the start AND end of the system prompt (primacy + recency effects)
2. Naming the attack explicitly so the model pattern-matches it
3. Framing the human turn as "evidence for debate" rather than "commands to follow"

…the system prompt authority is strongly reinforced, making the injection fail.

---

## LLM Provider Support

| Provider | Speed | Cost | Setup |
|----------|-------|------|-------|
| **Groq** (default) | ⚡ Very fast | Free tier available | Add `GROQ_API_KEY` |
| **OpenAI** | Fast | Paid | Add `OPENAI_API_KEY` |
| **Ollama** | Varies | Free (local) | Run `ollama pull llama3` |

Switch provider by setting `LLM_PROVIDER=groq|openai|ollama` in `.env`.

---

## Running Individual Phases

```bash
# Phase 1 only
python -m phase1.router

# Phase 2 only
python -m phase2.content_engine

# Phase 3 only
python -m phase3.combat_engine

# All phases + generate execution_logs.md
python main.py
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` / `langgraph` | LLM orchestration and state machine |
| `chromadb` | In-memory vector store for persona embeddings |
| `sentence-transformers` | Local embedding model (no API key needed) |
| `langchain-groq` | Groq LLM integration |
| `langchain-openai` | OpenAI LLM integration |
| `pydantic` | Structured output schema for Phase 2 |
| `python-dotenv` | `.env` file loading |
