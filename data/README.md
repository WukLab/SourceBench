# SourceBench data

The frozen evaluation snapshot for the SourceBench paper. Rescoring this snapshot with
`src/evaluation/compute_metrics.py` reproduces the paper's main source-quality results
without re-crawling the web.

**Licensing is not uniform across this directory.** See [`LICENSE`](LICENSE) before
redistributing anything here. Short version: evaluator outputs are CC BY 4.0, the
queries are CC BY-SA 4.0, and stored page text is third-party content released for
non-commercial research use only.

---

## Layout

```
data/
  queries/
    sourcebench_queries.csv      100 benchmark queries (query_ID, query, source)
    sourcebench_queries.jsonl    the same queries as JSONL (id, query, source)
  content-scores/                evaluator outputs, 16 files, one per configuration
  collected-sources/             frozen page snapshots, 15 files, gzipped JSON
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
on dataset names. Mapping to Table 1 of the paper:

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

These records contain no benchmark query text and no third-party page text.

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
- `scoring.py` reads uncompressed JSON. Decompress before scoring:
  `gunzip -k data/collected-sources/<slug>.json.gz`

## Not included

- **Human annotations.** The annotated benchmark pages, the calibration set, and
  per-annotator ratings are not released. Results that depend on them — the
  evaluator-versus-human agreement table in particular — cannot be recomputed from the
  files here. See the paper's reproducibility appendix.
- **Stored page text for `claude-sonnet-4.5`.** `content-scores/claude-sonnet-4.5.json`
  is present (259 records, with `url` and `text_length`), but there is no
  `collected-sources/claude-sonnet-4.5.json.gz`. Claude-Sonnet-4.5's scores are
  reproducible from the stored scores, but its page text is not available for rescoring.

## Reproducing the main results

`compute_metrics.py` needs no third-party packages.

```bash
python src/evaluation/compute_metrics.py \
  $(for f in data/content-scores/*.json; do \
      echo -n "--run $(basename "$f" .json)=$f "; done) \
  --query-metadata data/queries/sourcebench_queries.csv \
  --exclude-incomplete \
  --out-dir output/metrics
```

`--exclude-incomplete` averages the weighted score only over sources scored on all
eight dimensions. Omit it to include partially scored sources. The two settings give
different numbers, so check which one a given table in the paper uses before comparing.

Outputs: `leaderboard_overall.csv`, `leaderboard_by_query_type.csv`, `main_table.csv`,
`query_metrics.csv`, `source_metrics.csv`, `leaderboard_data.json`.

## Contact

Corrections, attribution changes, and takedown requests: Stephen Liu <xul052@ucsd.edu>
