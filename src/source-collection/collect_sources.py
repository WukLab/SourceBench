import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Default output dir (gitignored) for pipeline artifacts
_DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".output")

from dotenv import load_dotenv

load_dotenv()

from get_urls import URLGenerationConfig, generate_urls_for_queries
from collect_sources_from_urls import ScrapeConfig, collect_sources_from_url_file


@dataclass
class OrchestratorConfig:
    # LLM settings
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_urls: int = 5
    max_retries: int = 2
    sleep_seconds: float = 0.5
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    ai_model_name: str = "OpenAI"

    # Scraping settings
    timeout: int = 15
    min_chars: int = 500

    # Postprocessing / patching
    min_success_per_query: int = 1
    patch_rounds: int = 0  # how many extra rounds of URL generation+scrape for under-covered queries


def _load_queries_jsonl(
    input_path: str,
    id_field: str,
    query_field: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if id_field in row and query_field in row:
                records.append(row)
    return records


def _index_queries_by_id(
    records: List[Dict[str, Any]],
    id_field: str,
) -> Dict[Any, Dict[str, Any]]:
    return {row[id_field]: row for row in records}


def _count_successes_per_query(scraped_path: str) -> Counter:
    """Count successful aggregated_sources per query_id from JSON array output."""
    counts: Counter = Counter()
    with open(scraped_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return counts
    for record in data:
        qid = record.get("query_id")
        if qid is not None:
            counts[qid] += len(record.get("aggregated_sources") or [])
    return counts


def _subset_queries(
    all_records: List[Dict[str, Any]],
    id_field: str,
    subset_ids: List[Any],
) -> List[Dict[str, Any]]:
    idx = _index_queries_by_id(all_records, id_field)
    return [idx[qid] for qid in subset_ids if qid in idx]


def _write_subset_jsonl(
    records: List[Dict[str, Any]],
    path: str,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_collection_pipeline(
    query_path: str,
    urls_path: str,
    scraped_path: str,
    rejected_path: Optional[str],
    id_field: str,
    query_field: str,
    cfg: OrchestratorConfig,
) -> None:
    """
    Orchestrate the full pipeline:
      1. Generate URLs with get_urls.py-style logic.
      2. Scrape URLs with collect_sources_from_urls.py-style logic.
      3. Optionally run patch rounds for queries with too few successful sources.
    """
    # Step 1: initial URL generation for all queries
    url_gen_cfg = URLGenerationConfig(
        model=cfg.model,
        temperature=cfg.temperature,
        max_urls=cfg.max_urls,
        max_retries=cfg.max_retries,
        sleep_seconds=cfg.sleep_seconds,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
    )
    generate_urls_for_queries(
        input_path=query_path,
        output_path=urls_path,
        id_field=id_field,
        query_field=query_field,
        cfg=url_gen_cfg,
        ai_model_name=cfg.ai_model_name,
    )

    # Step 2: scrape URLs (output: JSON array matching reference format)
    scrape_cfg = ScrapeConfig(timeout=cfg.timeout, min_chars=cfg.min_chars)
    collect_sources_from_url_file(
        input_path=urls_path,
        output_path=scraped_path,
        rejected_output_path=rejected_path,
        cfg=scrape_cfg,
        ai_model_name=cfg.ai_model_name,
    )

    # Step 3: optional patch rounds to top up coverage for poorly-covered queries
    if cfg.patch_rounds <= 0 or cfg.min_success_per_query <= 0:
        return

    all_queries = _load_queries_jsonl(query_path, id_field, query_field)

    for round_idx in range(cfg.patch_rounds):
        success_counts = _count_successes_per_query(scraped_path)

        # Identify queries below the coverage threshold
        missing_by_id: Dict[Any, int] = defaultdict(int)
        for row in all_queries:
            qid = row[id_field]
            current = success_counts.get(qid, 0)
            if current < cfg.min_success_per_query:
                missing_by_id[qid] = cfg.min_success_per_query - current

        if not missing_by_id:
            # All queries are sufficiently covered
            break

        subset_ids = list(missing_by_id.keys())
        subset_records = _subset_queries(all_queries, id_field, subset_ids)
        if not subset_records:
            break

        # Create temporary subset query file and URL file for this round
        subset_query_path = f"{query_path}.patch_round_{round_idx+1}.jsonl"
        subset_urls_path = f"{urls_path}.patch_round_{round_idx+1}.json"
        subset_scraped_path = f"{scraped_path}.patch_round_{round_idx+1}.json"
        subset_rejected_path = (
            f"{rejected_path}.patch_round_{round_idx+1}.jsonl"
            if rejected_path
            else None
        )

        _write_subset_jsonl(subset_records, subset_query_path)

        # Generate additional URLs for under-covered queries
        generate_urls_for_queries(
            input_path=subset_query_path,
            output_path=subset_urls_path,
            id_field=id_field,
            query_field=query_field,
            cfg=url_gen_cfg,
            ai_model_name=cfg.ai_model_name,
        )

        # Scrape those URLs (output: JSON array)
        collect_sources_from_url_file(
            input_path=subset_urls_path,
            output_path=subset_scraped_path,
            rejected_output_path=subset_rejected_path,
            cfg=scrape_cfg,
            ai_model_name=cfg.ai_model_name,
        )

        # Merge patch results into main scraped JSON array
        with open(scraped_path, "r", encoding="utf-8") as main_f:
            main_data = json.load(main_f)
        with open(subset_scraped_path, "r", encoding="utf-8") as sub_f:
            patch_data = json.load(sub_f)
        main_by_qid: Dict[Any, Dict[str, Any]] = {r["query_id"]: r for r in main_data}
        for patch_record in patch_data:
            qid = patch_record.get("query_id")
            new_sources = patch_record.get("aggregated_sources") or []
            if not new_sources:
                continue
            if qid in main_by_qid:
                existing = main_by_qid[qid].get("aggregated_sources") or []
                next_id = len(existing) + 1
                for src in new_sources:
                    src["agg_source_id"] = next_id
                    next_id += 1
                    existing.append(src)
                main_by_qid[qid]["aggregated_sources"] = existing
            else:
                main_by_qid[qid] = patch_record
        merged_list = list(main_by_qid.values())
        merged_list.sort(key=lambda r: (str(r.get("query_id", "")), r.get("query_id")))
        with open(scraped_path, "w", encoding="utf-8") as main_f:
            json.dump(merged_list, main_f, ensure_ascii=False, indent=2)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Orchestrate URL generation (LLM) and source collection (scraping) "
            "into a single pipeline."
        ),
    )
    # Core paths
    parser.add_argument(
        "--queries",
        "-q",
        required=True,
        help="Path to input JSONL with queries.",
    )
    parser.add_argument(
        "--urls-out",
        "-u",
        default=None,
        help=(
            "Path to intermediate JSON file for generated URLs. "
            "Default: source-collection/.output/urls.json",
        ),
    )
    parser.add_argument(
        "--scraped-out",
        "-s",
        default=None,
        help=(
            "Path to output JSON file with scraped sources. "
            "Default: source-collection/.output/scraped.json",
        ),
    )
    parser.add_argument(
        "--rejected-out",
        default=None,
        help=(
            "Path to JSONL file for blocked/short/error scrapes. "
            "Default: source-collection/.output/rejected.jsonl",
        ),
    )

    # Field names in the query file
    parser.add_argument(
        "--id-field",
        default="id",
        help="Field name in input JSONL for the unique query id (default: id).",
    )
    parser.add_argument(
        "--query-field",
        default="query",
        help="Field name in input JSONL for the query text (default: query).",
    )

    # LLM config
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI chat model name for URL generation (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for URL generation (default: 0.2).",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=5,
        help="Maximum number of URLs to request per query (default: 5).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum number of retries per LLM call (default: 2).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Seconds to sleep between LLM calls (default: 0.5).",
    )
    parser.add_argument(
        "--openai-base-url",
        help=(
            "Optional custom OpenAI base URL "
            "(e.g. http://localhost:8000/v1 for a fake endpoint)."
        ),
    )
    parser.add_argument(
        "--openai-api-key",
        help=(
            "Optional OpenAI API key override. "
            "If not set, the OpenAI client will use environment configuration."
        ),
    )
    parser.add_argument(
        "--ai-model-name",
        default="OpenAI",
        help="Value for ai_model_name in scraped output (default: OpenAI).",
    )

    # Scraper config
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="HTTP timeout per request in seconds (default: 15).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="Minimum character length for a scrape to be considered valid (default: 500).",
    )

    # Patching / coverage
    parser.add_argument(
        "--min-success-per-query",
        type=int,
        default=1,
        help=(
            "Minimum number of successful sources per query before a query is "
            "considered sufficiently covered (default: 1)."
        ),
    )
    parser.add_argument(
        "--patch-rounds",
        type=int,
        default=0,
        help=(
            "Number of patch rounds: additional URL generation + scraping passes "
            "for queries that are under-covered (default: 0 = disabled)."
        ),
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    urls_path = args.urls_out or os.path.join(_DEFAULT_OUTPUT_DIR, "urls.json")
    scraped_path = args.scraped_out or os.path.join(_DEFAULT_OUTPUT_DIR, "scraped.json")
    rejected_path = args.rejected_out or os.path.join(_DEFAULT_OUTPUT_DIR, "rejected.jsonl")
    os.makedirs(_DEFAULT_OUTPUT_DIR, exist_ok=True)

    cfg = OrchestratorConfig(
        model=args.model,
        temperature=args.temperature,
        max_urls=args.max_urls,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        base_url=args.openai_base_url,
        api_key=args.openai_api_key,
        ai_model_name=args.ai_model_name,
        timeout=args.timeout,
        min_chars=args.min_chars,
        min_success_per_query=args.min_success_per_query,
        patch_rounds=args.patch_rounds,
    )

    run_collection_pipeline(
        query_path=args.queries,
        urls_path=urls_path,
        scraped_path=scraped_path,
        rejected_path=rejected_path,
        id_field=args.id_field,
        query_field=args.query_field,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()

