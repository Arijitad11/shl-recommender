# SHL Assessment Recommender

Conversational agent that takes a hiring manager from a vague intent to a grounded shortlist of SHL assessments through dialogue.

## Architecture

```
┌─────────────┐    POST /chat     ┌────────────────────────────────────────┐
│   Evaluator │ ───────────────►  │             FastAPI Service             │
│  (or user)  │ ◄─────────────── │                                        │
└─────────────┘  {reply, recs,   │  ┌──────────┐  ┌─────────────────────┐ │
                  eoc}            │  │  Agent   │  │ CatalogRetriever    │ │
                                  │  │ (agent.py│  │ (semantic + keyword)│ │
                                  │  └──────────┘  └─────────────────────┘ │
                                  │       │                   │             │
                                  │       ▼                   ▼             │
                                  │  LLM (Gemini/            catalog.json   │
                                  │   OpenAI/Groq/           (scraper.py)  │
                                  │   Anthropic)                            │
                                  └────────────────────────────────────────┘
```

### Key components

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, lifespan startup, `/health` and `/chat` endpoints |
| `agent.py` | Intent classification, prompt assembly, LLM call, JSON parsing, URL validation |
| `retriever.py` | Semantic (sentence-transformers + FAISS) or keyword (TF-IDF) retrieval |
| `scraper.py` | Live SHL catalog scraper + static fallback merge |
| `catalog_data.py` | Curated static catalog (≥40 Individual Test Solutions) |
| `eval.py` | 10-trace evaluation harness (Recall@10, hard evals, behavior probes) |

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your LLM API key

The service supports **Gemini (default/free)**, OpenAI, Groq, and Anthropic.

```bash
# Gemini (recommended free tier – https://aistudio.google.com)
export GEMINI_API_KEY=your_key_here

# OR OpenAI
export OPENAI_API_KEY=sk-...

# OR Groq (free tier – https://console.groq.com)
export GROQ_API_KEY=gsk_...

# OR Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

Optional overrides:
```bash
export MODEL_PROVIDER=gemini          # gemini | openai | groq | anthropic
export GEMINI_MODEL=gemini-2.0-flash  # default
export OPENAI_MODEL=gpt-4o-mini       # default
export GROQ_MODEL=llama-3.1-8b-instant
```

### 3. (Optional) Rebuild catalog

```bash
python scraper.py    # attempts live scrape; falls back to static data
```

### 4. Run the service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test

```bash
# Health
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a Java developer who works with stakeholders"},
      {"role": "assistant", "content": "Sure. What is the seniority level?"},
      {"role": "user", "content": "Mid-level, around 4 years experience"}
    ]
  }'
```

---

## Running the evaluator

```bash
# Against local server (make sure it is running)
python eval.py

# Against custom traces file
python eval.py --traces my_traces.json

# Against deployed service
python eval.py --base-url https://your-service.onrender.com
```

---

## Deployment (free tier)

### Render

1. Push this folder to a GitHub repo.
2. Create a **Web Service** on [render.com](https://render.com).
3. Set **Build command**: `pip install -r requirements.txt && python scraper.py`
4. Set **Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add your `GEMINI_API_KEY` (or equivalent) as an environment variable.

### Fly.io

```bash
fly launch
fly secrets set GEMINI_API_KEY=your_key
fly deploy
```

### Railway / Hugging Face Spaces

Both support standard Dockerfile deployments — use the provided `Dockerfile`.

---

## API specification

### `GET /health`

```json
{"status": "ok"}
```
HTTP 200. First call may take up to 2 minutes on cold-start deployments.

### `POST /chat`

**Request**
```json
{
  "messages": [
    {"role": "user",      "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is the seniority level?"},
    {"role": "user",      "content": "Mid-level, around 4 years"}
  ]
}
```

Constraints:
- Last message must have `role: "user"`.
- Maximum 8 messages total (≈ 4 turns each side).
- Each `content` field is capped at 4 000 characters.

**Response**
```json
{
  "reply": "Got it. Here are 5 assessments for a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r",       "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

`recommendations` is `[]` when the agent is clarifying or refusing.  
`end_of_conversation` is `true` only when the agent considers the task complete.

---

## Evaluation metrics

| Metric | Description |
|--------|-------------|
| **Hard eval pass rate** | Schema compliance, catalog-only URLs, ≤8-turn cap |
| **Recall@10** | Fraction of expected assessments appearing in top-10 recommendations |
| **Behavior probe rate** | Binary probes: refuses off-topic, no rec on vague query, honors refinement, etc. |

---

## Design notes

### Retrieval strategy
- **Primary**: sentence-transformers `all-MiniLM-L6-v2` + FAISS `IndexFlatIP` (cosine similarity). Fast, ~80 MB model.
- **Fallback**: TF-IDF keyword scoring (zero external deps). Activates if `sentence-transformers` or `faiss` is unavailable.
- Top-15 candidates are injected into the LLM context window; this grounds the agent and eliminates hallucinated products.

### Prompt design
- All catalog items are passed as structured context (name, URL, type, description, tags, levels).
- The LLM is instructed to return **only JSON** matching the response schema — no freetext wrapping.
- A regex+JSON parser strips markdown fences and extracts the first `{...}` block to handle imperfect LLM outputs.

### Hallucination guard
- Every recommendation's URL is validated against the catalog before the response is returned.
- If the URL is wrong but the name matches a catalog item, the correct URL is substituted automatically.
- Items that cannot be matched are silently dropped.

### Turn budget
- Conversations are capped at 8 turns (enforced in Pydantic validation).
- At turn ≥4 the system prompt gains a pressure clause urging the agent to commit to a shortlist.

### What didn't work
- Direct live scraping (SHL returns 403 to headless clients). Solution: curated static catalog + best-effort scrape merge.
- Asking the LLM to produce a comma-separated list of names (too fragile for the URL validation step). Solution: structured JSON output with explicit URL fields.
