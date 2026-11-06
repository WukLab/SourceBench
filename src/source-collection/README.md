# Source Collection

Collects the sources a system cites for each query. The output is the input of
[`scoring.py`](../content-scoring/scripts/README.md).

## How It Works

1. `get_urls.py` sends each query to the system three times and records the
   URLs cited in each answer, in citation order.
2. `collect_sources_from_urls.py` merges the cited URLs of all runs, drops
   pages that cannot be fetched or have no text, and selects up to five
   sources per query by citation frequency, then average citation rank, then
   text length. The page text of the selected sources is stored for scoring.

`collect_sources.py` runs both steps.

## Usage

```bash
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
python src/source-collection/collect_sources.py \
  --queries data/queries/sourcebench_queries.jsonl \
  --model gpt-5
```

Outputs go to `src/source-collection/.output/` (`urls.json`, `scraped.json`,
`rejected.jsonl`) unless set with `-u`, `-s` and `--rejected-out`.

- `--query-ids 1 2 3` runs a subset of queries.
- `--resume` continues an interrupted run; finished queries are kept.
- `--runs` and `--top-k` set the runs per query (default 3) and the sources
  selected per query (default 5).

Run any script with `--help` for all options.

## Adapting to Another Provider

This implementation uses the OpenAI Responses API with the `web_search` tool.
Only `query_system()` in `get_urls.py` is provider-specific. Replace it (and
`make_client()` if the provider needs another client) with a function that
takes one query and returns:

```python
SystemResponse(
    response_text="the final answer",
    citations=[{"url": "https://...", "title": "..."}],  # in citation order
    raw_response={...},   # the provider's response, stored as-is
    usage={...},          # optional
)
```

The rest of the pipeline stays the same. For a search API that returns a
ranked list of results instead of an answer, return that list as `citations`.

## Output

`scraped.json` is a list of queries. Each query has `runs` (one per call:
answer, raw response, cited URLs) and `aggregated_sources` (one per unique
cited URL, selected sources first). Main fields of an aggregated source:

| Field | Meaning |
|---|---|
| `selected_topk` | `true` for the sources that are scored |
| `text_content` | stored page text (selected sources only) |
| `frequency`, `ge_rank_list`, `avg_ge_rank` | runs citing the URL, its rank in each run, average rank |
| `citation_existence_check` | page fetched with non-empty text; `null` if not checked |
| `retrieved_at`, `text_sha256` | fetch time (UTC) and hash of the stored text |
| `search_result_date`, `search_result_last_updated` | HTTP `Date` and `Last-Modified` headers |
