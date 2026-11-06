"""Compute Tables 3, 4 and 8 of the paper from the released scores.

Usage (from the repository root; standard library only, no API key needed):

    python src/evaluation/reproduce_paper.py

Reads `data/content-scores/`, runs `compute_metrics.py`, and prints:

- Table 3: weighted score per system (`compute_metrics.py --exclude-incomplete`).
- Table 4: weighted score per query type (`compute_metrics.py` default
  settings). The "Overall" column is the Table 3 score.
- Table 8: per-dimension and weighted scores of the four DeepSeek
  configurations (`compute_metrics.py --exclude-incomplete`).

The full metric tables are written to `--out-dir` as CSV and JSON.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPUTE_METRICS = REPO_ROOT / "src" / "evaluation" / "compute_metrics.py"

# Paper name -> file stem in data/content-scores/, in Table 3 order.
MAIN_SYSTEMS = [
    ("GPT-5", "gpt-5"),
    ("Grok-4.1-Fast", "grok-4.1-fast"),
    ("GPT-4o", "gpt-4o"),
    ("Claude-Sonnet-4.5", "claude-sonnet-4.5"),
    ("Gemini-3-Pro", "gemini-3-pro"),
    ("Gemini-3-Flash", "gemini-3-flash"),
    ("Perplexity-Sonar-Pro", "perplexity-sonar-pro"),
    ("Gemini-2.5-Flash", "gemini-2.5-flash"),
    ("Gensee", "gensee"),
    ("Exa", "exa"),
    ("Tavily", "tavily"),
    ("Google-Search", "google-search"),
]

# Table 8 configurations.
DEEPSEEK_SYSTEMS = [
    ("DS-Chat-Tavily", "deepseek-chat-tavily"),
    ("DS-Chat-Gensee", "deepseek-chat-gensee"),
    ("DS-Reasoning-Tavily", "deepseek-reasoning-tavily"),
    ("DS-Reasoning-Gensee", "deepseek-reasoning-gensee"),
]

# Table 8 column -> metric key in compute_metrics.py output.
DIMENSIONS = [
    ("CR", "semantic_relevance"),
    ("FA", "factual_accuracy"),
    ("NE", "objectivity_tone"),
    ("FR", "freshness"),
    ("OA", "transparency"),
    ("AA", "accountability"),
    ("DA", "authority"),
    ("LC", "layout_ad_density"),
]

# Table 4 column -> value of the `source` column in sourcebench_queries.csv.
QUERY_TYPES = [
    ("Argument", "DebateQA"),
    ("Reasoning", "HotpotQA"),
    ("Factual", "Pinocchios"),
    ("Social", "QuoraQuestions"),
    ("Shopping", "VACOS"),
]


def run_compute_metrics(systems: list, scores_dir: Path, queries_csv: Path, out_dir: Path,
                        exclude_incomplete: bool) -> None:
    command = [sys.executable, str(COMPUTE_METRICS)]
    for _, stem in systems:
        command += ["--run", f"{stem}={scores_dir / (stem + '.json')}"]
    command += ["--query-metadata", str(queries_csv), "--out-dir", str(out_dir)]
    if exclude_incomplete:
        command.append("--exclude-incomplete")
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def read_rows(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: str) -> str:
    return f"{float(value):.3f}" if value not in (None, "") else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scores-dir", type=Path, default=REPO_ROOT / "data" / "content-scores")
    parser.add_argument("--queries", type=Path, default=REPO_ROOT / "data" / "queries" / "sourcebench_queries.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("output") / "paper")
    args = parser.parse_args()

    table3_dir = args.out_dir / "table3"
    table4_dir = args.out_dir / "table4"
    table8_dir = args.out_dir / "table8"
    run_compute_metrics(MAIN_SYSTEMS, args.scores_dir, args.queries, table3_dir, exclude_incomplete=True)
    run_compute_metrics(MAIN_SYSTEMS, args.scores_dir, args.queries, table4_dir, exclude_incomplete=False)
    run_compute_metrics(DEEPSEEK_SYSTEMS, args.scores_dir, args.queries, table8_dir, exclude_incomplete=True)

    overall = {row["model_name"]: row for row in read_rows(table3_dir / "leaderboard_overall.csv")}
    by_type = {
        (row["model_name"], row["query_type"]): row
        for row in read_rows(table4_dir / "leaderboard_by_query_type.csv")
    }

    print("Table 3: weighted score per system")
    print(f"  {'System':<22}{'Weighted':>10}")
    for name, stem in MAIN_SYSTEMS:
        print(f"  {name:<22}{fmt(overall[stem]['weighted_total_content_score']):>10}")

    print()
    print("Table 4: weighted score by query type")
    header = "".join(f"{label:>11}" for label, _ in QUERY_TYPES) + f"{'Overall':>11}"
    print(f"  {'System':<22}{header}")
    for name, stem in MAIN_SYSTEMS:
        cells = "".join(
            f"{fmt(by_type.get((stem, source), {}).get('weighted_total_content_score')):>11}"
            for _, source in QUERY_TYPES
        )
        print(f"  {name:<22}{cells}{fmt(overall[stem]['weighted_total_content_score']):>11}")

    deepseek = {row["model_name"]: row for row in read_rows(table8_dir / "leaderboard_overall.csv")}
    print()
    print("Table 8: DeepSeek configurations")
    header = "".join(f"{label:>8}" for label, _ in DIMENSIONS) + f"{'Weighted':>10}"
    print(f"  {'System':<22}{header}")
    for name, stem in DEEPSEEK_SYSTEMS:
        row = deepseek[stem]
        cells = "".join(f"{fmt(row[key]):>8}" for _, key in DIMENSIONS)
        print(f"  {name:<22}{cells}{fmt(row['weighted_total_content_score']):>10}")

    print()
    print(f"Full metric tables: {table3_dir}/, {table4_dir}/ and {table8_dir}/")


if __name__ == "__main__":
    main()
