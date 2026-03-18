# SourceBench

SourceBench is a benchmark for evaluating the quality of web sources cited by generative engines.
Instead of measuring only final answer quality, SourceBench evaluates whether a system cites sources
that are relevant, accurate, fresh, transparent, accountable, authoritative, and usable.

Resources:

- arXiv: `https://arxiv.org/abs/2602.16942`
- blog: `https://mlsys.wuklab.io/posts/sourcebench/`

This repository is the benchmark codebase. It keeps the public query split, source collection pipeline,
source scoring pipeline, metric computation code, and official submission contract. The public-facing
leaderboard site should be hosted separately.

## What is included

- fixed public query split for open evaluation
- source collection scripts
- source judging scripts
- metric computation scripts
- official submission validation and runner scripts
- split policy and official submission contract

## Repository layout

```text
data/queries/
  sourcebench_public_queries_v1.csv
leaderboard/
  QUERY_SPLIT_POLICY.md
  OFFICIAL_SUBMISSION_CONTRACT.md
  examples/
src/source-collection/
src/content-scoring/scripts/
src/evaluation/
requirements.txt
```

## Public evaluation pipeline

The public pipeline is:

1. Run source collection on the fixed public query set.
2. Scrape and normalize cited sources.
3. Score sources with the fixed judge configuration.
4. Compute leaderboard-ready metrics.

Core scripts:

- `src/source-collection/get_urls.py`
- `src/source-collection/collect_sources_from_urls.py`
- `src/content-scoring/scripts/scoring.py`
- `src/evaluation/compute_metrics.py`

## Official evaluation

Open evaluation can be run locally on the public split.

Official leaderboard evaluation is intended to be run server-side by the SourceBench team using:

- a hidden holdout split
- the fixed judge model and prompts
- the fixed metrics code
- a standardized submission contract

Relevant files:

- `leaderboard/QUERY_SPLIT_POLICY.md`
- `leaderboard/OFFICIAL_SUBMISSION_CONTRACT.md`
- `leaderboard/examples/`
- `src/evaluation/README.md`
- `src/evaluation/validate_official_submission.py`
- `src/evaluation/official_submission_backend.py`
- `src/evaluation/official_run.py`

## Notes on public release

This repository keeps only the public split.

The hidden holdout queries should not be committed here. The benchmark master pool should also stay out of the
public release if it can be used to reconstruct the holdout split. For that reason, `data/queries/queries.csv`
is excluded from this benchmark base.
