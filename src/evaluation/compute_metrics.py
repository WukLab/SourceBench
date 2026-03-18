"""Aggregate scored source files into leaderboard-ready metrics and artifacts."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCORE_KEYS = [
    "semantic_relevance",
    "factual_accuracy",
    "freshness",
    "objectivity_tone",
    "layout_ad_density",
    "accountability",
    "transparency",
    "authority",
]

WEIGHTED_SCORE_WEIGHTS = {
    "semantic_relevance": 3,
    "factual_accuracy": 3,
    "objectivity_tone": 3,
    "freshness": 2,
    "transparency": 2,
    "authority": 2,
    "layout_ad_density": 2,
    "accountability": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute leaderboard-ready metrics from one or more scored source files. "
            "Supports the new nested `.enriched.json` output from scoring.py and the older flat scoring JSON."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Run specification in the form MODEL_NAME=/path/to/scored.json. "
            "Repeat this flag to compare multiple systems."
        ),
    )
    parser.add_argument(
        "--rank-run",
        action="append",
        default=[],
        help=(
            "Optional rank-score file in the form MODEL_NAME=/path/to/geo_scores.json. "
            "Used to add overlap and search-rank metrics."
        ),
    )
    parser.add_argument(
        "--query-metadata",
        type=Path,
        help=(
            "Optional CSV with query metadata. The current repo format is "
            "`query_ID,query,source`, where `source` is the query type."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for leaderboard artifacts.",
    )
    parser.add_argument(
        "--json-indent",
        type=int,
        default=2,
        help="Indent level for JSON outputs.",
    )
    return parser.parse_args()


def parse_name_path_pairs(items: Iterable[str], flag_name: str) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{flag_name} must be in NAME=PATH format: {item}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip()).expanduser()
        if not name:
            raise ValueError(f"{flag_name} missing model name: {item}")
        if not path.exists():
            raise FileNotFoundError(f"{flag_name} path does not exist: {path}")
        parsed[name] = path
    return parsed


def normalize_query_id(value: Any) -> Any:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return value


def safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def get_query_type_map(csv_path: Optional[Path]) -> Dict[Any, str]:
    if csv_path is None:
        return {}

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return {}

        id_field = "query_ID" if "query_ID" in reader.fieldnames else "query_id"
        type_field = "source" if "source" in reader.fieldnames else "query_type"
        type_map = {}
        for row in reader:
            query_id = normalize_query_id(row.get(id_field))
            query_type = (row.get(type_field) or "").strip()
            if query_id is not None and query_type:
                type_map[query_id] = query_type
        return type_map


def get_source_url(source: Dict[str, Any]) -> str:
    for key in ("source_url", "url", "url_canonical"):
        value = source.get(key)
        if value:
            return str(value)
    url_examples = source.get("url_examples")
    if isinstance(url_examples, list) and url_examples:
        return str(url_examples[0])
    if isinstance(url_examples, str):
        return url_examples
    return ""


def compute_scores(content_vector: Dict[str, Any]) -> Dict[str, Any]:
    scores = {key: safe_float(content_vector.get(key)) for key in SCORE_KEYS}
    complete = all(scores[key] is not None for key in SCORE_KEYS)

    unweighted_mean = mean(scores.values())
    weighted_sum = sum((scores[key] or 0.0) * WEIGHTED_SCORE_WEIGHTS[key] for key in SCORE_KEYS)
    weighted_total = weighted_sum * (100.0 / 95.0)

    return {
        "scores": scores,
        "unweighted_mean_score": unweighted_mean,
        "weighted_total_content_score": weighted_total,
        "score_complete": complete,
    }


def iter_flat_records(
    data: List[Dict[str, Any]],
    model_name: str,
    query_type_map: Dict[Any, str],
) -> Iterable[Dict[str, Any]]:
    for item in data:
        if not isinstance(item, dict):
            continue
        content_vector = item.get("content_vector")
        if not isinstance(content_vector, dict):
            continue
        query_id = normalize_query_id(item.get("query_id"))
        query_type = item.get("query_type") or query_type_map.get(query_id) or "Unknown"
        metrics = compute_scores(content_vector)
        yield {
            "model_name": model_name,
            "query_id": query_id,
            "query_text": item.get("query_text", ""),
            "query_type": query_type,
            "agg_source_id": item.get("agg_source_id"),
            "source_url": item.get("source_url") or item.get("url") or "",
            "text_content_length": safe_float(item.get("text_length") or item.get("text_content_length")),
            "content_vector": content_vector,
            **metrics,
        }


def iter_nested_records(
    data: List[Dict[str, Any]],
    model_name: str,
    query_type_map: Dict[Any, str],
) -> Iterable[Dict[str, Any]]:
    for query_record in data:
        if not isinstance(query_record, dict):
            continue
        query_id = normalize_query_id(query_record.get("query_id"))
        query_text = query_record.get("query_text", "")
        query_type = query_record.get("query_type") or query_type_map.get(query_id) or "Unknown"
        for source in query_record.get("aggregated_sources") or []:
            if not isinstance(source, dict):
                continue
            content_vector = source.get("content_vector")
            if not isinstance(content_vector, dict):
                continue
            metrics = compute_scores(content_vector)
            yield {
                "model_name": model_name,
                "query_id": query_id,
                "query_text": query_text,
                "query_type": query_type,
                "agg_source_id": source.get("agg_source_id"),
                "source_url": get_source_url(source),
                "text_content_length": safe_float(source.get("text_content_length")),
                "content_vector": content_vector,
                **metrics,
            }


def load_scored_records(
    path: Path,
    model_name: str,
    query_type_map: Dict[Any, str],
) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level JSON list in {path}")

    if not data:
        return []

    first = data[0]
    if isinstance(first, dict) and "aggregated_sources" in first:
        return list(iter_nested_records(data, model_name, query_type_map))

    if isinstance(first, dict) and "content_vector" in first:
        return list(iter_flat_records(data, model_name, query_type_map))

    raise ValueError(
        f"Unsupported scoring schema in {path}. "
        "Expected nested query records with `aggregated_sources` or flat source records with `content_vector`."
    )


def load_rank_metrics(path: Path) -> Tuple[Dict[Tuple[Any, Any], Dict[str, Optional[float]]], Dict[Any, Dict[str, Optional[float]]]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        results = payload.get("results", [])
    else:
        results = payload

    source_map: Dict[Tuple[Any, Any], Dict[str, Optional[float]]] = {}
    query_map: Dict[Any, Dict[str, Optional[float]]] = {}

    for entry in results:
        if not isinstance(entry, dict):
            continue
        query_id = normalize_query_id(entry.get("query_id"))
        not_in_se = safe_float(entry.get("percentage_ge_sources_not_in_se_sources"))
        in_se = None if not_in_se is None else 100.0 - not_in_se
        query_map[query_id] = {
            "percentage_ge_sources_not_in_se_sources": not_in_se,
            "percentage_ge_sources_in_se_sources": in_se,
        }
        for source in entry.get("ge_sources") or []:
            source_map[(query_id, source.get("agg_source_id"))] = {
                "avg_ge_freq": safe_float(source.get("avg_ge_freq")),
                "relative_se_rank": safe_float(source.get("relative_se_rank")),
                "normalized_reciprocal_se_rank": safe_float(source.get("normalized_reciprocal_se_rank")),
                "reciprocal_se_rank": safe_float(source.get("reciprocal_se_rank")),
            }

    return source_map, query_map


def merge_rank_metrics(
    records: List[Dict[str, Any]],
    rank_source_map: Dict[Tuple[Any, Any], Dict[str, Optional[float]]],
    rank_query_map: Dict[Any, Dict[str, Optional[float]]],
) -> None:
    for record in records:
        source_metrics = rank_source_map.get((record["query_id"], record["agg_source_id"]), {})
        query_metrics = rank_query_map.get(record["query_id"], {})
        record.update(source_metrics)
        record.update(query_metrics)


def aggregate_source_groups(records: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in group_keys)].append(record)

    output = []
    for key_tuple, rows in grouped.items():
        row = {group_keys[index]: key_tuple[index] for index in range(len(group_keys))}
        row["num_sources"] = len(rows)
        row["num_queries"] = len({item["query_id"] for item in rows})
        row["num_complete_scores"] = sum(1 for item in rows if item["score_complete"])
        row["unweighted_mean_score"] = mean(item["unweighted_mean_score"] for item in rows)
        row["weighted_total_content_score"] = mean(item["weighted_total_content_score"] for item in rows)
        for metric_key in SCORE_KEYS:
            row[metric_key] = mean(item["scores"].get(metric_key) for item in rows)
        for extra_key in [
            "avg_ge_freq",
            "relative_se_rank",
            "normalized_reciprocal_se_rank",
            "reciprocal_se_rank",
            "percentage_ge_sources_not_in_se_sources",
            "percentage_ge_sources_in_se_sources",
        ]:
            row[extra_key] = mean(item.get(extra_key) for item in rows)
        output.append(row)
    return output


def aggregate_query_groups(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model_name"], record["query_id"], record["query_type"])].append(record)

    output = []
    for (model_name, query_id, query_type), rows in grouped.items():
        row = {
            "model_name": model_name,
            "query_id": query_id,
            "query_type": query_type,
            "num_sources": len(rows),
            "unweighted_mean_score": mean(item["unweighted_mean_score"] for item in rows),
            "weighted_total_content_score": mean(item["weighted_total_content_score"] for item in rows),
        }
        for metric_key in SCORE_KEYS:
            row[metric_key] = mean(item["scores"].get(metric_key) for item in rows)
        for extra_key in [
            "avg_ge_freq",
            "relative_se_rank",
            "normalized_reciprocal_se_rank",
            "reciprocal_se_rank",
            "percentage_ge_sources_not_in_se_sources",
            "percentage_ge_sources_in_se_sources",
        ]:
            row[extra_key] = mean(item.get(extra_key) for item in rows)
        output.append(row)
    return output


def sort_rows(rows: List[Dict[str, Any]], primary_key: str) -> List[Dict[str, Any]]:
    def sort_key(row: Dict[str, Any]) -> Tuple[float, str]:
        value = row.get(primary_key)
        numeric = -1e18 if value is None else -float(value)
        return (numeric, str(row.get("model_name", "")))

    return sorted(rows, key=sort_key)


def write_csv_file(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json_file(path: Path, payload: Any, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False)


def build_leaderboard_payload(
    run_specs: Dict[str, Path],
    rank_specs: Dict[str, Path],
    overall_rows: List[Dict[str, Any]],
    by_query_type_rows: List[Dict[str, Any]],
    query_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scoring_dimensions": SCORE_KEYS,
            "weighted_score_formula": {
                "weights": WEIGHTED_SCORE_WEIGHTS,
                "scale_factor": 100.0 / 95.0,
            },
            "runs": [
                {
                    "model_name": model_name,
                    "score_file": str(path),
                    "rank_file": str(rank_specs[model_name]) if model_name in rank_specs else None,
                }
                for model_name, path in run_specs.items()
            ],
        },
        "overall": overall_rows,
        "by_query_type": by_query_type_rows,
        "queries": query_rows,
    }


def main() -> None:
    args = parse_args()
    run_specs = parse_name_path_pairs(args.run, "--run")
    rank_specs = parse_name_path_pairs(args.rank_run, "--rank-run")
    query_type_map = get_query_type_map(args.query_metadata)

    all_records: List[Dict[str, Any]] = []
    for model_name, score_path in run_specs.items():
        records = load_scored_records(score_path, model_name, query_type_map)
        if model_name in rank_specs:
            rank_source_map, rank_query_map = load_rank_metrics(rank_specs[model_name])
            merge_rank_metrics(records, rank_source_map, rank_query_map)
        all_records.extend(records)

    source_rows = []
    for record in all_records:
        source_row = {
            "model_name": record["model_name"],
            "query_id": record["query_id"],
            "query_type": record["query_type"],
            "agg_source_id": record["agg_source_id"],
            "source_url": record["source_url"],
            "text_content_length": record["text_content_length"],
            "unweighted_mean_score": record["unweighted_mean_score"],
            "weighted_total_content_score": record["weighted_total_content_score"],
            "score_complete": record["score_complete"],
        }
        source_row.update(record["scores"])
        for extra_key in [
            "avg_ge_freq",
            "relative_se_rank",
            "normalized_reciprocal_se_rank",
            "reciprocal_se_rank",
            "percentage_ge_sources_not_in_se_sources",
            "percentage_ge_sources_in_se_sources",
        ]:
            source_row[extra_key] = record.get(extra_key)
        source_rows.append(source_row)

    query_rows = sort_rows(aggregate_query_groups(all_records), "weighted_total_content_score")
    overall_rows = sort_rows(
        aggregate_source_groups(all_records, ["model_name"]),
        "weighted_total_content_score",
    )
    by_query_type_rows = sort_rows(
        aggregate_source_groups(all_records, ["model_name", "query_type"]),
        "weighted_total_content_score",
    )

    main_table_rows = [
        {
            "model_name": row["model_name"],
            "query_type": row["query_type"],
            "unweighted_mean_score": row["unweighted_mean_score"],
            "weighted_total_content_score": row["weighted_total_content_score"],
            "num_sources": row["num_sources"],
            "num_queries": row["num_queries"],
        }
        for row in by_query_type_rows
    ]

    payload = build_leaderboard_payload(run_specs, rank_specs, overall_rows, by_query_type_rows, query_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_file(args.out_dir / "source_metrics.csv", source_rows)
    write_csv_file(args.out_dir / "query_metrics.csv", query_rows)
    write_csv_file(args.out_dir / "leaderboard_overall.csv", overall_rows)
    write_csv_file(args.out_dir / "leaderboard_by_query_type.csv", by_query_type_rows)
    write_csv_file(args.out_dir / "main_table.csv", main_table_rows)
    write_json_file(args.out_dir / "leaderboard_data.json", payload, args.json_indent)

    print(f"Wrote leaderboard artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
