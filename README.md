<div align="center">

# SourceBench: A Benchmark for Cited Source Quality in Generative Engines

[![Code License](https://img.shields.io/badge/Code%20License-MIT-blue?style=flat-square)](#)
[![GitHub](https://img.shields.io/badge/GitHub-SourceBench-181717?style=flat-square&logo=github)](https://github.com/WukLab/SourceBench)
[![arXiv](https://img.shields.io/badge/arXiv-2602.16942-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2602.16942)
[![Blog](https://img.shields.io/badge/Blog-SourceBench-f59e0b?style=flat-square)](https://mlsys.wuklab.io/posts/sourcebench/)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-hosted%20separately-8b5cf6?style=flat-square)](#official-evaluation)

SourceBench evaluates whether a generative engine cites **high-quality web sources**, not only whether it produces a fluent final answer.

</div>

---

## Overview

SourceBench focuses on the quality of cited sources along dimensions such as:

- semantic relevance
- factual accuracy
- freshness
- objectivity / tone
- layout / ad density
- accountability
- transparency
- authority

This repository is the **benchmark codebase**. It contains:

- the fixed public query split for open evaluation
- source collection scripts
- source judging scripts
- metric computation scripts
- official submission validation and runner scripts
- split policy and official submission contract

The public leaderboard site should be hosted separately from this repository.

## Repository Layout

```text
data/queries/
  sourcebench_public_queries_v1.csv
leaderboard/
  QUERY_SPLIT_POLICY.md
  OFFICIAL_SUBMISSION_CONTRACT.md
  README.md
  examples/
src/source-collection/
src/content-scoring/scripts/
src/evaluation/
requirements.txt
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the public evaluation pipeline:

```bash
python src/source-collection/get_urls.py \
  --input path/to/public_queries.jsonl \
  --output output/public_urls.json \
  --model YOUR_MODEL_NAME \
  --openai-base-url YOUR_OPENAI_COMPATIBLE_ENDPOINT \
  --openai-api-key YOUR_GE_API_KEY \
  --ai-model-name YOUR_MODEL_NAME
```

```bash
python src/source-collection/collect_sources_from_urls.py \
  --input output/public_urls.json \
  --output output/public_sources.json \
  --rejected-output output/public_rejected.jsonl \
  --ai-model-name YOUR_MODEL_NAME
```

```bash
export QWEN_API_KEY=YOUR_QWEN_API_KEY
python src/content-scoring/scripts/scoring.py \
  --input-file output/public_sources.json \
  --out-dir output/scored \
  --run-name YOUR_MODEL_NAME
```

```bash
python src/evaluation/compute_metrics.py \
  --run YOUR_MODEL_NAME=output/scored/YOUR_MODEL_NAME.enriched.json \
  --query-metadata data/queries/sourcebench_public_queries_v1.csv \
  --out-dir output/metrics
```

## Evaluation Scripts

The evaluation-side scripts are documented in:

- [`src/evaluation/README.md`](src/evaluation/README.md)

In short:

- `validate_official_submission.py`: validate official submission schema
- `official_submission_backend.py`: intake and queue a submission
- `official_run.py`: run hidden official evaluation
- `compute_metrics.py`: aggregate final metrics

## Official Evaluation

Open evaluation can be run locally on the public split.

Official leaderboard evaluation is intended to be run server-side by the SourceBench team using:

- a hidden holdout split
- the fixed judge model and prompts
- the fixed metrics code
- a standardized submission contract

Relevant files:

- [`leaderboard/QUERY_SPLIT_POLICY.md`](leaderboard/QUERY_SPLIT_POLICY.md)
- [`leaderboard/OFFICIAL_SUBMISSION_CONTRACT.md`](leaderboard/OFFICIAL_SUBMISSION_CONTRACT.md)
- [`leaderboard/README.md`](leaderboard/README.md)
- [`leaderboard/examples/`](leaderboard/examples/)

## Public Release Notes

This repository keeps only the public split.

The hidden holdout queries should not be committed here. The benchmark master query pool should also stay out of the public release if it can be used to reconstruct the holdout split. For that reason, `data/queries/queries.csv` is excluded from this benchmark base.

Internal official submission artifacts are also excluded. The ignored directory:

- `leaderboard/.official_submissions/`

is only for server-side submission intake and official evaluation runs.

## Citation

If you use SourceBench in your research, please cite:

```bibtex
@article{sourcebench2026,
  title={SourceBench: Can AI Answers Reference Quality Web Sources?},
  author={Hexi Jin and Stephen Liu and Yuheng Li and Simran Malik and Yiying Zhang},
  journal={arXiv preprint arXiv:2602.16942},
  year={2026}
}
```
