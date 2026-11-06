"""Run both source-collection steps with one command.

    queries.jsonl --get_urls.py--> urls.json --collect_sources_from_urls.py--> scraped.json

`scraped.json` is the input of `src/content-scoring/scripts/scoring.py`.
Intermediate files go to `src/source-collection/.output/` unless overridden.
"""

import argparse
import os
from typing import List, Optional

from collect_sources_from_urls import ScrapeConfig, collect_sources_from_url_file
from get_urls import CollectionConfig, generate_urls_for_queries

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

_DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect sources end to end: query the system (step 1), then select and scrape (step 2).",
    )
    parser.add_argument("--queries", "-q", required=True, help="Query file (JSONL).")
    parser.add_argument("--urls-out", "-u", help="Step 1 output (default: .output/urls.json).")
    parser.add_argument("--scraped-out", "-s", help="Step 2 output (default: .output/scraped.json).")
    parser.add_argument("--rejected-out", help="Failed existence checks (default: .output/rejected.jsonl).")
    parser.add_argument("--id-field", default="id", help="Query id field (default: id).")
    parser.add_argument("--query-field", default="query", help="Query text field (default: query).")
    parser.add_argument("--query-ids", nargs="+", help="Only collect these query ids.")

    step1 = parser.add_argument_group("step 1: querying the system")
    step1.add_argument("--model", required=True, help="Model name, e.g. gpt-5 or gpt-4o.")
    step1.add_argument("--runs", type=int, default=3, help="Runs per query (default: 3).")
    step1.add_argument("--tool-type", default="web_search", help="Hosted search tool type (default: web_search).")
    step1.add_argument("--tool-choice", choices=["required", "auto"], default="required")
    step1.add_argument("--temperature", type=float, default=None)
    step1.add_argument("--max-retries", type=int, default=3)
    step1.add_argument("--max-workers", type=int, default=4, help="Queries processed in parallel in step 1.")
    step1.add_argument("--resume", action="store_true", help="Skip queries already complete in --urls-out.")
    step1.add_argument("--openai-base-url")
    step1.add_argument("--openai-api-key")
    step1.add_argument("--ai-model-name", help="Label stored as ai_model_name (default: --model).")

    step2 = parser.add_argument_group("step 2: existence check and selection")
    step2.add_argument("--top-k", type=int, default=5, help="Sources selected per query (default: 5).")
    step2.add_argument("--timeout", type=int, default=15)
    step2.add_argument("--min-chars", type=int, default=1)
    step2.add_argument("--scrape-workers", type=int, default=8, help="Parallel page fetches in step 2.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    if load_dotenv is not None:
        load_dotenv()
    args = build_parser().parse_args(argv)
    urls_path = args.urls_out or os.path.join(_DEFAULT_OUTPUT_DIR, "urls.json")
    scraped_path = args.scraped_out or os.path.join(_DEFAULT_OUTPUT_DIR, "scraped.json")
    rejected_path = args.rejected_out or os.path.join(_DEFAULT_OUTPUT_DIR, "rejected.jsonl")
    for path in (urls_path, scraped_path, rejected_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    generate_urls_for_queries(
        input_path=args.queries,
        output_path=urls_path,
        cfg=CollectionConfig(
            model=args.model,
            runs=args.runs,
            tool_type=args.tool_type,
            tool_choice=args.tool_choice,
            temperature=args.temperature,
            max_retries=args.max_retries,
            base_url=args.openai_base_url,
            api_key=args.openai_api_key,
        ),
        id_field=args.id_field,
        query_field=args.query_field,
        ai_model_name=args.ai_model_name,
        query_ids=args.query_ids,
        max_workers=args.max_workers,
        resume=args.resume,
    )
    collect_sources_from_url_file(
        input_path=urls_path,
        output_path=scraped_path,
        rejected_output_path=rejected_path,
        cfg=ScrapeConfig(
            top_k=args.top_k,
            timeout=args.timeout,
            min_chars=args.min_chars,
            max_workers=args.scrape_workers,
        ),
    )


if __name__ == "__main__":
    main()
