# Scoring

`scoring.py` scores the selected sources of each query on eight dimensions
with the judge model `qwen-plus-2025-07-28` on DashScope (OpenAI-compatible
endpoint, temperature 0).

## Input

A JSON list of queries with `aggregated_sources`, as written by
`src/source-collection/collect_sources.py` or stored in
`data/collected-sources/`. `.json.gz` files are read directly. Sources with
`selected_topk: true` are scored, up to five per query.

## Usage

```bash
export QWEN_API_KEY=YOUR_DASHSCOPE_API_KEY
python src/content-scoring/scripts/scoring.py \
  --input-file output/gpt-5/sources.json \
  --out-dir output/gpt-5/scored \
  --run-name gpt-5
```

- `--query-ids 2 11 21` scores a subset of queries.
- `--dry-run` writes mock scores without calling the API, to check the input
  and output format.
- `QWEN_BASE_URL` sets another endpoint (default: DashScope international);
  `SOURCEBENCH_JUDGE_MODEL` sets another model name.

The freshness rubric treats January 2026 as the current date.

## Output

- `<run-name>.enriched.json`: the input with `content_vector` (the eight
  scores), `content_vector_justification` and `scoring_meta` added to each
  scored source. This is the input of `src/evaluation/compute_metrics.py`.
- `<run-name>.csv`: one row per scored source. `score_list` is ordered
  semantic_relevance, factual_accuracy, freshness, objectivity_tone,
  layout_ad_density, accountability, transparency, authority.
