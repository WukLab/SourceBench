# Evaluation scripts

This directory contains the benchmark-side evaluation scripts for SourceBench.

The file names are historical, but the intended workflow is simple:

1. `validate_official_submission.py`
2. `official_submission_backend.py`
3. `official_run.py`
4. `compute_metrics.py`

If you are only running the public benchmark locally, you usually need just:

- `compute_metrics.py`

If you are running official leaderboard evaluation server-side, the full flow is:

- validate submission schema
- intake the submission into an internal queue
- execute the official run on the hidden holdout split
- compute final leaderboard metrics

## Script guide

### `compute_metrics.py`

Purpose:

- convert one or more scored source files into leaderboard-ready metrics
- aggregate overall metrics and per-query-type metrics
- export JSON and CSV artifacts

Use this when:

- you already have judged source files from `scoring.py`
- you want final metrics for local comparison or official reporting

Main input:

- one or more scored JSON files, usually `*.enriched.json`

Main output:

- `leaderboard_data.json`
- `leaderboard_overall.csv`
- `leaderboard_by_query_type.csv`
- query- and source-level CSVs

Typical command:

```bash
python src/evaluation/compute_metrics.py \
  --run MODEL_NAME=path/to/scored.enriched.json \
  --query-metadata data/queries/sourcebench_public_queries_v1.csv \
  --out-dir output/metrics
```

### `validate_official_submission.py`

Purpose:

- validate the schema of an official submission package
- support both `endpoint` and `answer_url_bundle` submissions

Use this when:

- a participant has prepared a submission JSON
- you want a strict yes/no schema check before any execution

Main input:

- one submission JSON file

Main output:

- validation report JSON
- exit code `0` for valid submissions, `1` for invalid submissions

Typical command:

```bash
python src/evaluation/validate_official_submission.py \
  --input leaderboard/examples/endpoint_submission.example.json \
  --output output/validation_report.json
```

### `official_submission_backend.py`

Purpose:

- intake a validated submission into the internal official evaluation queue
- assign a submission id
- write submission status, evaluation request, and redacted copies

This script is best understood as the **official submission intake step**.

Use this when:

- you want to register a participant submission for server-side evaluation
- you want a reproducible internal directory for that submission

Main input:

- one participant submission JSON

Main output:

- a submission directory under `leaderboard/.official_submissions/`
- `validation_report.json`
- `submission_status.json`
- `evaluation_request.json`
- redacted or raw submission copies

Typical command:

```bash
python src/evaluation/official_submission_backend.py \
  --input submission.json
```

### `official_run.py`

Purpose:

- execute the official SourceBench pipeline for a queued submission
- run hidden queries for endpoint submissions, or continue from answer+URL bundles
- run scraping, judging, and final metric computation

This script is best understood as the **official evaluation runner**.

Use this when:

- a submission has already been validated and accepted into the queue
- you want to produce official artifacts for the holdout evaluation

Main input:

- a submission directory produced by `official_submission_backend.py`
- an internal holdout manifest CSV

Main output:

- stage outputs for source collection, scraping, scoring, and metrics
- final official metrics artifacts
- updated submission status and run manifest

Typical command:

```bash
python src/evaluation/official_run.py \
  --submission-dir leaderboard/.official_submissions/SUBMISSION_ID \
  --holdout-manifest path/to/internal_holdout.csv
```

## Recommended mental model

Think about these files as four layers:

- `validate_official_submission.py`: schema validation
- `official_submission_backend.py`: intake and queueing
- `official_run.py`: execution
- `compute_metrics.py`: metric aggregation

## Naming note

If this repository is refactored further, the clearer long-form names would be:

- `validate_submission.py`
- `intake_official_submission.py`
- `run_official_evaluation.py`
- `compute_metrics.py`

The current file names are kept for compatibility with the existing docs and examples.
