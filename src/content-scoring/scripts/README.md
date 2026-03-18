# `scoring.py`

- `scoring.py`: read a query-list JSON file, score selected sources with Qwen, and write an enriched JSON plus a CSV.

## What Problem This Script Solves

Suppose you already have a JSON file where:

- the file is a list of queries
- each query has its text
- each query also has a list of source pages under `aggregated_sources`
- each source already contains extracted page text such as `text_content`

This script scores those source pages on 8 dimensions and writes the scores back out in two formats:

- `enriched.json`: keeps the original nested JSON and adds scores onto each scored source
- `csv`: one row per scored source, easier to inspect in spreadsheets or pandas

## What File Should I Pass To `--input-file`?

Pass the raw query-list JSON.

The input is correct if, after opening the file, it looks roughly like this:

```json
[
  {
    "query_id": 2,
    "query_text": "some user query",
    "aggregated_sources": [
      {
        "agg_source_id": 1,
        "selected_topk": true,
        "url_examples": ["https://example.com/page"],
        "text_content": "page text ...",
        "text_content_length": 27848
      }
    ]
  }
]
```

Do not pass:

- an old flat scoring JSON
- a CSV
- a JSON that already contains only one row per source

## What The Script Writes

If you run:

```bash
python3 scripts/scoring.py \
  --input-file /path/to/input.json \
  --out-dir /path/to/output \
  --run-name my_run
```

the script writes:

- `/path/to/output/my_run.enriched.json`
- `/path/to/output/my_run.csv`

If `--run-name` is omitted, it uses `input_file.stem`.

### Enriched JSON

This keeps the original nested query structure and adds these fields onto each scored source:

- `content_vector`
- `content_vector_justification`
- `scoring_meta`

### CSV

The CSV uses lowercase column names:

```text
query_id,query_text,source_url,text_content_length,score_list
```

`score_list` is ordered as:

```text
semantic_relevance
factual_accuracy
freshness
objectivity_tone
layout_ad_density
accountability
transparency
authority
```

## Minimal Usage

Dry run:

```bash
python3 scripts/scoring.py \
  --input-file /path/to/input.json \
  --out-dir /path/to/output \
  --dry-run
```

Dry run does not call the model API. It writes mock scores, which is useful for checking:

- your input file is accepted
- output paths look right
- CSV / enriched JSON format is correct

Live run:

```bash
python3 scripts/scoring.py \
  --input-file /path/to/input.json \
  --out-dir /path/to/output \
  --run-name my_run \
  --query-ids 2 11 21
```

If `--query-ids` is omitted, the script processes all queries in the file.

## Environment Variables

For live scoring, set:

```bash
export QWEN_API_KEY=...
```

Optional:

```bash
export QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

`QWEN_BASE_URL` falls back to the DashScope-compatible endpoint if you do not set it.

## What The Script Assumes By Default

The script intentionally exposes only a small runtime surface:

- `--input-file`
- `--out-dir`
- `--run-name`
- `--dry-run`
- `--query-ids`

Other behavior is fixed near the top of `scoring.py`, including:

- model name
- timeout
- concurrency
- selected-topk filtering
- max sources per query
