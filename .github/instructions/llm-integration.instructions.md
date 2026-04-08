---
description: "Use when working on Ollama integration, prompt engineering, LLM narrative generation, or parsing LLM responses. Covers the Ollama HTTP client, prompt templates, and narrative parsing in src/llm/."
applyTo: "src/llm/**,tests/test_llm/**"
---

# LLM Integration Instructions

## Ollama Client (`src/llm/ollama_client.py`)
- Communicate with Ollama via its HTTP REST API (`/api/generate` or `/api/chat`)
- Read `OLLAMA_BASE_URL` and `OLLAMA_MODEL` from `config/settings.py` — never hardcode
- Use `httpx.AsyncClient` with a configurable timeout (default 120s — LLMs are slow)
- Implement a `generate(prompt: str) -> str` async function as the single entry point
- Handle HTTP errors and timeouts explicitly; raise a custom `OllamaError` with a safe message
- Log the model name and prompt length at DEBUG; never log the full prompt in production

## Prompt Design (`src/llm/narrative.py`)
- Build the prompt in `build_prompt(prediction_context: dict) -> str`
- The prompt must instruct the model to act as a **data analyst** with the following structure:
  1. A 2–3 sentence executive summary of the predicted salary
  2. **Explicit uncertainty statement** — the narrative MUST tell the user:
     - The point estimate (e.g. "$125 000")
     - The peer-group salary range from the leaf IQR: "Most peers with this profile earn between $110 000 and $140 000"
     - The model's typical error magnitude: "Predictions for this model carry a typical error of roughly ±$31 500 (MAE)"
     This section is mandatory — never omit or soften it.
  3. Key factors driving the result (feature importance insights)
  4. A comparison to regional/industry averages (from context, not hallucinated)
  5. At least one chart specification (described in a `[CHART]` block — see format below)
  6. One actionable recommendation for the candidate
- The `prediction_context` dict passed to `build_prompt` must include:
  - `point_estimate`, `range_low`, `range_high` — from `PredictionResult`
  - `model_mae` — injected from the registry metrics so the LLM can quote it accurately
- Keep the system prompt under 500 tokens; inject dynamic data into a user message
- Use few-shot examples to enforce the output format

## Chart Specification Format
The LLM must include exactly one chart block in its response:
```
[CHART]
type: bar | histogram | scatter | line
title: <chart title>
x_label: <axis label>
y_label: <axis label>
data_key: <key name the dashboard will look up from Supabase>
[/CHART]
```
- `src/visualizations/charts.py` reads this block and generates the actual figure
- Validate the parsed block with a Pydantic model before passing to the chart layer

## Narrative Parsing
- Parse the LLM response with `parse_narrative(raw: str) -> NarrativeResult`
- `NarrativeResult`: `summary: str`, `insights: list[str]`, `chart_spec: ChartSpec`, `recommendation: str`
- Use regex or a simple state machine — do not call the LLM a second time to fix malformed output
- If the `[CHART]` block is missing or malformed, log a WARNING and use a sensible default chart spec

## General Rules
- All Ollama calls are async; do not use `asyncio.run()` inside library code
- Write tests in `tests/test_llm/` with a mocked HTTP client — no real Ollama calls in CI
- Keep prompts versioned as string constants or template files — easy to A/B test later
