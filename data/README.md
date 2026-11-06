# SourceBench data

The frozen evaluation snapshot for the SourceBench paper. Rescoring this snapshot with
`src/evaluation/compute_metrics.py` computes the main source-quality metrics
without re-crawling the web.

**Licensing is not uniform across this directory.** See [`LICENSE`](LICENSE) before
redistributing anything here. Short version: evaluator and analysis outputs are
CC BY 4.0, the queries follow the terms of their source datasets (CC BY-SA 4.0 for
HotpotQA, CC BY-NC-SA 3.0 for VACOS NLQ), and stored page text is third-party
content released for non-commercial research use only.

---

## Layout

```
data/
  queries/
    sourcebench_queries.csv      100 benchmark queries (query_ID, query, source)
    sourcebench_queries.jsonl    the same queries as JSONL (id, query, source)
  content-scores/                evaluator outputs, 16 files, one per configuration
  collected-sources/             frozen page snapshots, 15 files, gzipped JSON
  evaluation-results/            stored pipeline outputs (SE/GE overlap, per-query-type metrics)
  rank-scores/                   stored GEO rank scores and collection debug reports
  LICENSE                        per-component data terms
  README.md                      this file
```

## Configurations

Twelve systems are evaluated in the paper's main results, plus four DeepSeek
configurations used in the controlled retrieval-vs-reasoning study.

| File slug | System in the paper |
|---|---|
| `gpt-5` | GPT-5 |
| `gpt-4o` | GPT-4o |
| `gemini-3-pro` | Gemini-3-Pro |
| `gemini-3-flash` | Gemini-3-Flash |
| `gemini-2.5-flash` | Gemini-2.5-Flash |
| `grok-4.1-fast` | Grok-4.1-Fast |
| `claude-sonnet-4.5` | Claude-Sonnet-4.5 |
| `perplexity-sonar-pro` | Perplexity-Sonar-Pro |
| `google-search` | Google Search |
| `exa` | Exa AI Search |
| `tavily` | Tavily AI Search |
| `gensee` | Gensee AI Search |
| `deepseek-chat-tavily` | DS-Chat-Tavily (study only) |
| `deepseek-chat-gensee` | DS-Chat-Gensee (study only) |
| `deepseek-reasoning-tavily` | DS-Reasoning-Tavily (study only) |
| `deepseek-reasoning-gensee` | DS-Reasoning-Gensee (study only) |

## Record counts

| Scope | Scored records | Complete on all 8 dimensions |
|---|---|---|
| All 16 configurations | 4,808 | 4,708 |
| The 12 main systems | 4,508 | 4,431 |
| The 4 DeepSeek study configurations | 300 | 277 |

These are counts of records in `content-scores/`. The paper's figure of 3,996
system–query–source instances is defined in the paper and is **not** the raw record
count in these files; do not expect the two numbers to match.

## Query types

The `source` column in the queries CSV and the `query_type` field in `content-scores`
both hold the **name of the originating dataset**, not the paper's intent label.
`compute_metrics.py` groups by this value, so `leaderboard_by_query_type.csv` is keyed
on dataset names. Mapping to the paper's query types:

| Value in the data | Paper's type | Paper's source name | Primary intent |
|---|---|---|---|
| `VACOS` | Shopping | VA-COS NLQ | Product search and purchase guidance |
| `HotpotQA` | Reasoning | HotpotQA | Multi-hop factual research and lookups |
| `Pinocchios` | Factuality | Pinocchio | Fact-checking and claim verification |
| `DebateQA` | Argumentative | DebateQA | Multi-perspective subjective reasoning |
| `QuoraQuestions` | Social Q&A | Quora | Community-based forum inquiries |

Twenty queries per type. Note `Pinocchios` and `QuoraQuestions` are the labels as
stored; the paper names the datasets Pinocchio and Quora Question Pairs.

## `content-scores/*.json`

A JSON array of records, one per scored source:

| Field | Meaning |
|---|---|
| `query_id` | Matches `query_ID` in the queries file |
| `agg_source_id` | Identifies the source within that query, matching `collected-sources` |
| `url` | The scored page |
| `text_length` | Character length of the page text that was scored |
| `query_type` | Originating dataset; see the mapping above |
| `group1_analysis`, `group2_analysis`, `group3_analysis` | Intermediate evaluator output for the content, metadata, and layout prompt groups |
| `content_vector` | The eight scores, 1–5 |
| `content_vector_justification` | One written justification per dimension |

**`content_vector` keys do not match the paper's metric names.** The mapping:

| Key in the data | Paper metric | Abbrev. | Weight |
|---|---|---|---|
| `semantic_relevance` | Content Relevance | CR | 3 |
| `factual_accuracy` | Factual Accuracy | FA | 3 |
| `objectivity_tone` | Neutrality | NE | 3 |
| `freshness` | Freshness | FR | 2 |
| `transparency` | **Ownership** | OA | 2 |
| `accountability` | **Author Accountability** | AA | 2 |
| `authority` | **Domain Authority** | DA | 2 |
| `layout_ad_density` | Layout Clarity | LC | 2 |

The three bolded rows are the easy ones to misread: `transparency` is Ownership (not a
general transparency score), `accountability` is Author Accountability, and `authority`
is Domain Authority. Weights are those used for the weighted score; the maximum
possible weighted sum is 95, scaled to 100.

Justifications and analyses may quote short excerpts of the query and of the evaluated
page.

## `collected-sources/*.json.gz`

Gzipped JSON arrays, one record per query, holding the frozen page snapshots used for
scoring and the evaluated system's own answer.

| Field | Meaning |
|---|---|
| `query_id`, `query_text` | The benchmark query |
| `ai_model_name` | The evaluated configuration |
| `runs[]` | Per-run collection output, including `response_text`, `cited_sources`, `search_results` |
| `aggregated_sources[]` | Deduplicated candidate sources for the query |

Inside `aggregated_sources[]`: `agg_source_id`, `url_canonical`, `url_examples`,
`frequency`, `ge_rank_list`, `ge_score`, `selected_topk`, `avg_ge_freq`,
`domain_category`, `title`, `text_content` and its several length variants, plus
`search_result_date` and `search_result_last_updated`.

Two cautions:

- **`search_result_date` and `search_result_last_updated` are dates reported by the page
  or the search provider. They are not crawl timestamps.** This release does not record
  retrieval timestamps or content hashes.
- `scoring.py` reads the `.json.gz` files directly.

## `evaluation-results/` and `rank-scores/`

Stored outputs of the analysis pipeline, included so that recomputed numbers can be
diffed against the ones used in the paper rather than only re-derived.

| Path | Contents |
|---|---|
| `evaluation-results/<system>/merged_se_ge_scores.json` | Per-query overlap between a system's cited sources and Google Search results, including the `percentage_ge_sources_not_in_se_sources` figure behind the SE/GE analysis |
| `evaluation-results/se-ge/` | `analyzed_data.json` and `correlations.csv` for the cross-system SE/GE comparison |
| `evaluation-results/query-type-metrics/<system>/<system>_scores.csv` | Per-source scores tagged with query type, plus `total_content_score` and `weighted_total_content_score` |
| `evaluation-results/query-type-metrics/*.csv` | Aggregates: `avg_content_score_per_query_type.csv`, `overall_score_per_llm_query_type.csv`, `query_type_audit.csv` |
| `evaluation-results/content-metrics/content_stats_audit.json` | Content-statistics audit |
| `rank-scores/<system>/geo_scores.json` | GEO rank scores |
| `rank-scores/<system>/*_report.json` | Collection debug reports, including sources not found in the Google Search result set |

**Provenance and caveats.** These files were recovered from the pre-release archive
prepared before the public repository existed, so three things differ from the rest of
`data/`:

1. **Folder names have been normalised** to the slugs used elsewhere in this directory.
   In the archive they appeared as `claude`, `grok`, `grok-4.1-fast-non-reasoning`,
   `perplexity`, `gemini-2.5-flash-preview`, `gemini-3-flash-preview` and
   `gemini-3-pro-preview`.
2. **Superseded duplicates were removed.** The archive held a second, earlier copy of
   five `query-type-metrics` folders (`perplexity-pro`, and underscore-named DeepSeek
   variants such as `deepseek_chat_gensee`). Those copies predate the
   `weighted_total_content_score` column, and `perplexity-pro` reports every
   `query_type` as `Unknown`. Only the current versions are included; their row counts
   match `content-scores/` exactly.
3. **Coverage is narrower than `content-scores/`.** Neither tree includes
   `google-search`, and `rank-scores/` additionally has no `gemini-2.5-flash`. Google
   Search is the baseline the SE/GE comparison is measured against rather than a system
   measured by it.

These outputs are not required to compute the main metrics — `compute_metrics.py`
regenerates the equivalent aggregates from `content-scores/`. They are included so the
specific artifacts behind the SE/GE and per-query-type analyses are inspectable.

## Not included

- **Human annotations.** The annotated benchmark pages, the calibration set, and
  per-annotator ratings are not released. Results that depend on them — the
  evaluator-versus-human agreement table in particular — cannot be recomputed from the
  files here. See the paper's reproducibility appendix.
- **Stored page text for `claude-sonnet-4.5`.** `content-scores/claude-sonnet-4.5.json`
  is present (259 records, with `url` and `text_length`), but there is no
  `collected-sources/claude-sonnet-4.5.json.gz`. Claude-Sonnet-4.5's scores can
  be computed from the stored scores, but its page text is not available for rescoring.

## Computing the scores

`src/evaluation/compute_released_scores.py` computes, from `content-scores/`, the weighted
score of each system, its weighted score per query type, and the scores of the four
DeepSeek configurations. It needs no third-party packages:

```bash
python src/evaluation/compute_released_scores.py
```

The metric files (`leaderboard_overall.csv`, `leaderboard_by_query_type.csv`,
`main_table.csv`, `query_metrics.csv`, `source_metrics.csv`, `leaderboard_data.json`)
are written to `output/paper/`.

## Contact

Corrections, attribution changes, and takedown requests: Stephen Liu <xul052@ucsd.edu>
