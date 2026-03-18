import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


# Output schema matches data/collected-sources/gpt-5/GE_Response_Data_5_web_search_scored_deduped_with_avg_ge_freq.json
# Top-level: list of { query_id, query_text, ai_model_name, runs, aggregated_sources }
# Each aggregated_sources item: agg_source_id, url_canonical, url_examples, frequency, ge_rank_list,
#   ge_score, selected_topk, text_content, text_content_length, domain_category, search_result_date,
#   search_result_last_updated, title, text_content_w_links_length, text_content_w_short_links_length,
#   text_content_short_links, avg_ge_freq


@dataclass
class ScrapeConfig:
    timeout: int = 15
    min_chars: int = 500
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )


def _get_domain_category(url: str) -> str:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return "." + domain.split(".")[-1] if "." in domain else ".unknown"
    except Exception:
        return ".unknown"


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _html_to_text_and_title(html: str) -> Tuple[str, Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")
    title: Optional[str] = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip() or None
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text_chunks: List[str] = []
    for line in soup.get_text("\n").splitlines():
        line = line.strip()
        if line:
            text_chunks.append(line)
    text = "\n".join(text_chunks)
    if title is None and text_chunks:
        title = text_chunks[0][:200] if text_chunks[0] else None
    return text, title


def scrape_url(url: str, cfg: ScrapeConfig) -> Tuple[str, Optional[int], str, Optional[str]]:
    """
    Scrape a single URL.

    Returns (status, http_status, content_or_error, title_or_none)
      - status: "ok", "blocked", "too_short", or "error"
      - title only set when status == "ok" and HTML had a title or first line used
    """
    headers = {"User-Agent": cfg.user_agent}
    try:
        resp = requests.get(url, headers=headers, timeout=cfg.timeout)
    except requests.RequestException as exc:
        return "error", None, f"request_error: {exc}", None

    http_status = resp.status_code
    if http_status != 200:
        return "blocked", http_status, f"http_status_{http_status}", None

    content_type = resp.headers.get("Content-Type", "")
    text: str
    title: Optional[str] = None
    if "text/html" in content_type or "application/xhtml+xml" in content_type:
        text, title = _html_to_text_and_title(resp.text)
    else:
        try:
            text = resp.text
        except Exception:  # pylint: disable=broad-except
            return "error", http_status, "non_text_content", None
        if text and "\n" in text:
            title = text.split("\n")[0].strip()[:200] or None
        elif text:
            title = text[:200] or None

    if len(text) < cfg.min_chars:
        return "too_short", http_status, text, None

    return "ok", http_status, text, title


def _build_aggregated_source_item(
    query_id: Any,
    url: str,
    url_examples: List[str],
    text_content: str,
    title: Optional[str],
    agg_source_id: int,
) -> Dict[str, Any]:
    """Build one aggregated_sources entry matching reference JSON schema (all fields)."""
    n = len(text_content)
    domain_category = _get_domain_category(url)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return {
        "agg_source_id": agg_source_id,
        "url_canonical": url,
        "url_examples": list(dict.fromkeys(url_examples)),
        "frequency": 1,
        "ge_rank_list": [],
        "ge_score": 0.0,
        "selected_topk": True,
        "text_content": text_content,
        "text_content_length": n,
        "domain_category": domain_category,
        "search_result_date": today,
        "search_result_last_updated": None,
        "title": title or "",
        "text_content_w_links_length": n,
        "text_content_w_short_links_length": n,
        "text_content_short_links": text_content,
        "avg_ge_freq": 1.0,
    }


def _urls_from_record(record: Dict[str, Any]) -> List[str]:
    """Extract list of URLs from a record in reference format (runs[0].cited_sources or raw_response.output)."""
    urls: List[str] = []
    runs = record.get("runs") or []
    if not runs:
        return urls
    run = runs[0]
    # Prefer cited_sources (each item has "url" key)
    for src in run.get("cited_sources") or []:
        if isinstance(src, dict) and src.get("url"):
            urls.append(src["url"])
        elif isinstance(src, str):
            urls.append(src)
    if urls:
        return urls
    # Fallback: raw_response.output[].action.sources
    rr = run.get("raw_response") or {}
    for item in rr.get("output") or []:
        action = item.get("action") or {}
        for s in action.get("sources") or []:
            if isinstance(s, dict) and s.get("url"):
                urls.append(s["url"])
            elif isinstance(s, str):
                urls.append(s)
    return urls


def collect_sources_from_url_file(
    input_path: str,
    output_path: str,
    rejected_output_path: Optional[str],
    cfg: ScrapeConfig,
    ai_model_name: Optional[str] = None,
) -> None:
    """
    Read reference-format JSON (output of get_urls.py) and scrape URLs; fill aggregated_sources.

    Input: single JSON file (array) matching
    data/collected-sources/gpt-5/GE_Response_Data_5_web_search_scored_deduped_with_avg_ge_freq.json
    Each element has query_id, query_text, ai_model_name, runs, aggregated_sources (empty).
    URLs are taken from runs[0].cited_sources or runs[0].raw_response.output[].action.sources.

    Output: same JSON structure with aggregated_sources filled (and runs preserved unchanged).
    Rejected scrapes (blocked/too_short/error) written to rejected_output_path as JSONL.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    rejected_records: List[Dict[str, Any]] = []
    output_list: List[Dict[str, Any]] = []

    for record in data:
        query_id = record.get("query_id")
        query_text = record.get("query_text", "")
        model_name = ai_model_name or record.get("ai_model_name") or "OpenAI"
        urls = _urls_from_record(record)
        seen: Set[str] = set()
        ok_scrapes: List[Dict[str, Any]] = []

        for url in urls:
            if not url or url in seen:
                continue
            seen.add(url)
            status, http_status, content_or_error, title = scrape_url(url, cfg)
            if status == "ok":
                ok_scrapes.append({
                    "url": url,
                    "content": content_or_error,
                    "title": title,
                })
            else:
                rejected_records.append({
                    "query_id": query_id,
                    "query_text": query_text,
                    "url": url,
                    "status": status,
                    "http_status": http_status,
                    "error": content_or_error,
                })

        aggregated_sources: List[Dict[str, Any]] = []
        for idx, s in enumerate(ok_scrapes, start=1):
            item = _build_aggregated_source_item(
                query_id=query_id,
                url=s["url"],
                url_examples=[s["url"]],
                text_content=s["content"],
                title=s.get("title"),
                agg_source_id=idx,
            )
            aggregated_sources.append(item)

        # Preserve full record structure (runs, etc.) and only set aggregated_sources
        out_record = dict(record)
        out_record["aggregated_sources"] = aggregated_sources
        if "ai_model_name" not in out_record:
            out_record["ai_model_name"] = model_name
        output_list.append(out_record)

    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(output_list, out_f, ensure_ascii=False, indent=2)

    if rejected_output_path:
        with open(rejected_output_path, "w", encoding="utf-8") as rej_f:
            for rec in rejected_records:
                rej_f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape content for URLs produced by get_urls.py and fill aggregated_sources.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input JSON file in reference format.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output JSON file (reference format with aggregated_sources filled).",
    )
    parser.add_argument(
        "--rejected-output",
        help=(
            "Optional path to JSONL file where blocked, short, or errored "
            "scrapes will be written."
        ),
    )
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
        help=(
            "Minimum character length for a scrape to be considered valid "
            "(default: 500)."
        ),
    )
    parser.add_argument(
        "--ai-model-name",
        default="OpenAI",
        help="Value for ai_model_name in output (default: OpenAI).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = ScrapeConfig(timeout=args.timeout, min_chars=args.min_chars)
    collect_sources_from_url_file(
        input_path=args.input,
        output_path=args.output,
        rejected_output_path=args.rejected_output,
        cfg=cfg,
        ai_model_name=args.ai_model_name,
    )


if __name__ == "__main__":
    main()
