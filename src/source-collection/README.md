# Source collection scripts

Scripts to generate source URLs (via an LLM) and scrape their content into the same JSON format used elsewhere in this repo (`data/collected-sources/...`).

## Setup

From the repo root:

```bash
pip install -r requirements.txt
# or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in your environment or in a `.env` file in the repo root (scripts use `python-dotenv`).

**Model:** The `--model` flag accepts any model name supported by the API you call. That includes OpenAI models (e.g. `gpt-4o-mini`, `gpt-4o`) and, when using a custom base URL, any **OpenAI-compatible** endpoint (e.g. other providers or local servers). Use `--openai-base-url` to point to a different API; the same `--model` name is sent in the request.

## Scripts

| Script | Purpose |
|--------|--------|
| **`get_urls.py`** | Call OpenAI (or OpenAI-compatible API) to suggest URLs for each query. Output is a single JSON file (reference format) with `runs` and empty `aggregated_sources`. |
| **`collect_sources_from_urls.py`** | Read a URL JSON (reference format), scrape each URL, and fill `aggregated_sources`. |
| **`collect_sources.py`** | Full pipeline: run `get_urls` then scrape (and optionally patch under-covered queries). One command from queries → scraped JSON. |

## Quick start (all-in-one)

Run the full pipeline. You only need a **query file** (JSONL with `id` and `query` per line). Output goes to `source-collection/.output/` by default (gitignored).

```bash
# From repo root
python src/source-collection/collect_sources.py \
  --queries path/to/queries.jsonl
```

Example query file `queries.jsonl`:

```jsonl
{"id": "q1", "query": "What is the capital of France?"}
{"id": "q2", "query": "How does photosynthesis work?"}
```

Outputs (by default):

- `src/source-collection/.output/urls.json` — generated URLs (reference format)
- `src/source-collection/.output/scraped.json` — same structure with `aggregated_sources` filled
- `src/source-collection/.output/rejected.jsonl` — URLs that failed or were too short

Override paths:

```bash
python src/source-collection/collect_sources.py \
  -q queries.jsonl \
  -u /tmp/urls.json \
  -s /tmp/scraped.json \
  --rejected-out /tmp/rejected.jsonl
```

## Running the steps separately

**1. Generate URLs only (OpenAI)**

```bash
python src/source-collection/get_urls.py \
  --input queries.jsonl \
  --output urls.json \
  --model gpt-4o-mini \
  --max-urls 5
```

**2. Scrape from a URL file**

```bash
python src/source-collection/collect_sources_from_urls.py \
  --input urls.json \
  --output scraped.json \
  --rejected-output rejected.jsonl
```

## Options (high level)

- **`collect_sources.py`**: `--model` (any OpenAI or OpenAI-compatible model name), `--temperature`, `--max-urls`, `--min-success-per-query`, `--patch-rounds`, `--ai-model-name`, `--id-field`, `--query-field`, plus `--openai-base-url` and `--openai-api-key` for custom/OpenAI-compatible endpoints.
- **`get_urls.py`**: `--model`, `--temperature`, `--max-urls`, `--id-field`, `--query-field`, `--ai-model-name`, `--openai-base-url`, `--openai-api-key`.
- **`collect_sources_from_urls.py`**: `--timeout`, `--min-chars`, `--ai-model-name`.

Use `--help` on any script for full options.

Example with an OpenAI-compatible endpoint (e.g. another provider or local server):

```bash
python src/source-collection/collect_sources.py -q queries.jsonl \
  --model their-model-name \
  --openai-base-url https://api.other-provider.com/v1 \
  --openai-api-key YOUR_KEY
```

## Output format

All JSON output matches the schema of `data/collected-sources/gpt-5/GE_Response_Data_5_web_search_scored_deduped_with_avg_ge_freq.json`: top-level array of records with `query_id`, `query_text`, `ai_model_name`, `runs`, and `aggregated_sources`.
