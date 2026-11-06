<div align="center">

# SourceBench: Can AI Answers Reference Quality Web Sources?

[![Code License](https://img.shields.io/badge/Code%20License-MIT-blue?style=flat-square)](LICENSE)
[![Data License](https://img.shields.io/badge/Data-see%20data%2FLICENSE-green?style=flat-square)](data/LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-SourceBench-181717?style=flat-square&logo=github)](https://github.com/WukLab/SourceBench)
[![arXiv](https://img.shields.io/badge/arXiv-2602.16942-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2602.16942)
[![Blog](https://img.shields.io/badge/Blog-SourceBench-f59e0b?style=flat-square)](https://mlsys.wuklab.io/posts/sourcebench/)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Hugging%20Face-8b5cf6?style=flat-square)](https://huggingface.co/spaces/WukLab/SourceBench-Leaderboard)

SourceBench evaluates whether a generative engine cites **high-quality web sources**, not only whether it produces a fluent final answer.

</div>

---

## Overview

SourceBench scores every source a system cites on eight dimensions: semantic
relevance, factual accuracy, freshness, objectivity / tone, layout / ad
density, accountability, transparency, and authority.

Evaluating a system takes three steps: collect the sources it cites, score
them with an LLM judge, and compute metrics. The queries, collected sources,
and scores used in the paper are included in `data/`.

## Repository Layout

```text
LICENSE                          code license (MIT)
requirements.txt
data/
  LICENSE                        data terms
  queries/                       the 100 benchmark queries (.csv, .jsonl)
  collected-sources/             collected sources with stored page text (.json.gz)
  content-scores/                judge scores and justifications
  evaluation-results/            stored analysis outputs (SE/GE overlap, per-query-type scores)
  rank-scores/                   stored rank scores and collection reports
  README.md                      description of the data files
leaderboard/                     leaderboard submission format and examples
src/
  source-collection/             step 1: collect cited sources
  content-scoring/scripts/       step 2: score sources
  evaluation/                    step 3: compute metrics; submission scripts
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Collect Sources

```bash
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
python src/source-collection/collect_sources.py \
  --queries data/queries/sourcebench_queries.jsonl \
  --model gpt-5 \
  --urls-out output/gpt-5/urls.json \
  --scraped-out output/gpt-5/sources.json \
  --rejected-out output/gpt-5/rejected.jsonl
```

Each query is sent to the system three times. The URLs cited in its answers
are merged, pages that cannot be fetched or have no text are dropped, and up to five sources
per query are selected and their page text is stored in
`output/gpt-5/sources.json`.

This is the implementation for OpenAI models with web search. For other
providers, see [`src/source-collection/README.md`](src/source-collection/README.md).

## 2. Score Sources

```bash
export QWEN_API_KEY=YOUR_DASHSCOPE_API_KEY
python src/content-scoring/scripts/scoring.py \
  --input-file output/gpt-5/sources.json \
  --out-dir output/gpt-5/scored \
  --run-name gpt-5
```

This scores the selected sources with `qwen-plus-2025-07-28` and writes
`output/gpt-5/scored/gpt-5.enriched.json`. See
[`src/content-scoring/scripts/README.md`](src/content-scoring/scripts/README.md)
for options.

## 3. Compute Metrics

```bash
python src/evaluation/compute_metrics.py \
  --run gpt-5=output/gpt-5/scored/gpt-5.enriched.json \
  --query-metadata data/queries/sourcebench_queries.csv \
  --exclude-incomplete \
  --out-dir output/gpt-5/metrics
```

Repeat `--run NAME=PATH` to compute several systems together.
`--exclude-incomplete` averages over sources scored on all eight dimensions.
The output directory contains:

| File | Content |
|---|---|
| `leaderboard_overall.csv` | per-system scores: the eight dimensions and the weighted score |
| `leaderboard_by_query_type.csv` | per-system scores by query type |
| `source_metrics.csv`, `query_metrics.csv` | per-source and per-query scores |
| `leaderboard_data.json` | all of the above as JSON |

## Using the Released Data

To get the paper's results from the data in `data/` without collecting
sources again:

```bash
python src/evaluation/reproduce_paper.py
```

This computes metrics from `data/content-scores/` and prints the weighted score
of each system, its weighted score per query type, and the scores of the four
DeepSeek configurations. The metric files are written to `output/paper/`. It
needs no API key and no third-party packages.

To score the stored sources again, pass a file from `data/collected-sources/`
to step 2:

```bash
python src/content-scoring/scripts/scoring.py \
  --input-file data/collected-sources/gpt-5.json.gz \
  --out-dir output/rescored \
  --run-name gpt-5
```

## Leaderboard Score

The scripts in `src/evaluation/` score a system from a submission file in the
[leaderboard](https://huggingface.co/spaces/WukLab/SourceBench-Leaderboard)
format. There are two ways to prepare one (examples in
[`leaderboard/examples/`](leaderboard/examples/)):

- **`endpoint`**: the model name, API base URL and API key. Sources are
  collected with `get_urls.py` as in step 1. For providers other than OpenAI,
  adapt `query_system()` first (see
  [`src/source-collection/README.md`](src/source-collection/README.md)).
- **`answer_url_bundle`**: run your system on the queries in
  `data/queries/sourcebench_queries.jsonl` and record, for each query, the
  `query_id`, the answer (`answer_text`) and the cited URLs (`cited_urls`).
  Repeat a `query_id` to give several runs of the same query.

Then score the submission locally:

```bash
python src/evaluation/official_submission_backend.py \
  --input my_submission.json

python src/evaluation/official_run.py \
  --submission-dir leaderboard/.official_submissions/SUBMISSION_ID \
  --holdout-manifest data/queries/sourcebench_queries.csv
```

The first command checks the file and creates the submission directory; it
prints the directory path. For `endpoint` submissions, add `--store-secrets` so
the API key is kept for the run. The second command collects or scrapes the
sources, scores them, and computes the metrics. Results are written to
`SUBMISSION_DIR/official_run/`; `stage4/metrics/leaderboard_data.json` has the
same format as the leaderboard data. Add `--dry-run-judge` to test the run
without calling the judge API.

## License

The code in this repository is released under the [MIT License](LICENSE).

The data in `data/` is **not** uniformly licensed and is **not** covered by the MIT
license. It mixes material we created, material derived from third-party research
datasets, and third-party web page content whose copyright we do not hold. Terms are
stated per component in [`data/LICENSE`](data/LICENSE), and the data itself is
documented in [`data/README.md`](data/README.md). In summary:

| Component | Terms |
|---|---|
| `data/content-scores/` — evaluator scores and justifications | CC BY 4.0 |
| `data/queries/` — the 100 benchmark queries | Per source dataset: CC BY-SA 4.0 for the 20 HotpotQA-derived queries; the other 80 follow their upstream terms, for non-commercial research use |
| `data/collected-sources/` — frozen page snapshots | Layered. Stored page text is third-party content, copyright retained by the original publishers, redistributed for non-commercial research use only |
| `data/evaluation-results/`, `data/rank-scores/` — stored analysis outputs | CC BY 4.0 |

Rights holders may request removal of any stored page text; see the contact in
[`data/LICENSE`](data/LICENSE).

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
