"""Step 2 of SourceBench source collection: aggregate, check, scrape, select.

Input is the output of ``get_urls.py``. For each query:

1. Canonicalize the cited URLs of every run and take their union.
2. Existence check: a URL passes when it can be retrieved (HTTP 200) and
   yields non-empty extractable page text.
3. Select up to K = 5 passing sources by three ordered criteria:
   (a) citation frequency across runs (3 > 2 > 1),
   (b) average rank in the answers (lower is better),
   (c) extracted text length (longer is better).

Candidates are fetched in (frequency, average rank) order, one tie group at a
time, so only the pages needed for the selection are downloaded.

The extracted text of each selected source is stored in ``text_content``;
``scoring.py`` scores exactly this stored text. Each fetched source also gets
``retrieved_at`` (UTC time of retrieval) and ``text_sha256`` (hash of the stored
text). ``search_result_date`` and ``search_result_last_updated`` come from the
HTTP ``Date`` and ``Last-Modified`` headers; the freshness rubric reads them.

Text extraction uses requests + BeautifulSoup and does not run JavaScript.
Pages that only render their content in the browser will fail the existence
check; use a headless browser if you need them.
"""

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from url_utils import canonicalize_url, format_http_date, get_domain_category


@dataclass
class ScrapeConfig:
    top_k: int = 5
    timeout: int = 15
    min_chars: int = 1  # non-empty text
    max_workers: int = 8
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )


BLOCKED_CONTENT_PATTERNS = [
    "routing to checkout",
    "this page no longer exists",
    "has been moved",
    "page not found",
    "access denied",
    "robot check",
    "captcha",
]

HTML_TYPES = ("text/html", "application/xhtml+xml")


def extract_text(html: str) -> tuple:
    """Return (text, title) for an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True)).strip()
    return text, title


def fetch_page(url: str, cfg: ScrapeConfig) -> Dict[str, Any]:
    """Fetch one URL and run the existence check.

    Returns a dict with `ok`, `status` ("ok" or the failure reason), and, when
    retrieval succeeded, the extracted text and HTTP metadata.
    """
    result: Dict[str, Any] = {
        "ok": False,
        "status": "",
        "http_status": None,
        "text": "",
        "title": "",
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "http_date": None,
        "http_last_modified": None,
    }
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": cfg.user_agent},
            timeout=cfg.timeout,
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        result["status"] = f"request_error: {type(exc).__name__}"
        return result

    result["http_status"] = resp.status_code
    result["http_date"] = format_http_date(resp.headers.get("Date"))
    result["http_last_modified"] = format_http_date(resp.headers.get("Last-Modified"))
    if resp.status_code != 200:
        result["status"] = f"http_status_{resp.status_code}"
        return result

    content_type = (resp.headers.get("Content-Type") or "").lower()
    if resp.encoding is None or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding
    if any(kind in content_type for kind in HTML_TYPES) or not content_type:
        text, title = extract_text(resp.text)
    elif content_type.startswith("text/"):
        text, title = re.sub(r"\s+", " ", resp.text).strip(), ""
    else:
        result["status"] = f"unsupported_content_type: {content_type.split(';')[0]}"
        return result

    result["title"] = title
    if any(pattern in text.lower() for pattern in BLOCKED_CONTENT_PATTERNS):
        result["status"] = "blocked_content"
        return result
    if len(text) < max(1, cfg.min_chars):
        result["status"] = "empty_text" if not text else "too_short"
        return result

    result.update(ok=True, status="ok", text=text)
    return result


def aggregate_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Union the cited URLs of all runs by canonical URL.

    A URL counts once per run toward `frequency`; `ge_rank_list` holds its rank
    in each run that cited it.
    """
    union: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        seen = set()
        for index, source in enumerate(run.get("cited_sources") or [], start=1):
            url = source.get("url") if isinstance(source, dict) else source
            if not url:
                continue
            canonical = canonicalize_url(url)
            if canonical in seen:
                continue
            seen.add(canonical)
            rank = source.get("ge_rank_in_run", index) if isinstance(source, dict) else index
            entry = union.setdefault(canonical, {
                "url_canonical": canonical,
                "url_examples": [],
                "title": "",
                "frequency": 0,
                "ge_rank_list": [],
            })
            entry["frequency"] += 1
            entry["ge_rank_list"].append(rank)
            if url not in entry["url_examples"]:
                entry["url_examples"].append(url)
            if not entry["title"] and isinstance(source, dict):
                entry["title"] = source.get("title") or ""

    candidates = list(union.values())
    for entry in candidates:
        ranks = entry["ge_rank_list"]
        entry["avg_ge_rank"] = sum(ranks) / len(ranks)
        entry["ge_score"] = entry["frequency"] / entry["avg_ge_rank"]
        entry["domain_category"] = get_domain_category(entry["url_examples"][0])
    candidates.sort(key=lambda e: (-e["frequency"], e["avg_ge_rank"]))
    return candidates


def select_sources(candidates: List[Dict[str, Any]], cfg: ScrapeConfig,
                   pool: ThreadPoolExecutor) -> List[Dict[str, Any]]:
    """Run the existence check and pick up to top_k sources.

    Returns every candidate, selected sources first, with selection and
    retrieval fields filled in.
    """
    selected: List[Dict[str, Any]] = []
    for _, group_iter in groupby(candidates, key=lambda e: (e["frequency"], e["avg_ge_rank"])):
        if len(selected) >= cfg.top_k:
            break
        group = list(group_iter)
        fetched = list(pool.map(lambda e: fetch_page(e["url_examples"][0], cfg), group))
        passing = []
        for entry, page in zip(group, fetched):
            entry["_page"] = page
            if page["ok"]:
                passing.append(entry)
        passing.sort(key=lambda e: -len(e["_page"]["text"]))  # criterion (c)
        selected.extend(passing[: cfg.top_k - len(selected)])

    selected_ids = {id(entry) for entry in selected}
    ordered = selected + [entry for entry in candidates if id(entry) not in selected_ids]

    output: List[Dict[str, Any]] = []
    for agg_id, entry in enumerate(ordered, start=1):
        page = entry.pop("_page", None)
        is_selected = id(entry) in selected_ids
        text = page["text"] if (page and is_selected) else ""
        output.append({
            "agg_source_id": agg_id,
            "url_canonical": entry["url_canonical"],
            "url_examples": entry["url_examples"],
            "title": (page or {}).get("title") or entry["title"],
            "frequency": entry["frequency"],
            "ge_rank_list": entry["ge_rank_list"],
            "avg_ge_rank": entry["avg_ge_rank"],
            "ge_score": entry["ge_score"],
            "domain_category": entry["domain_category"],
            "citation_existence_check": page["ok"] if page else None,
            "existence_check_status": page["status"] if page else "not_checked",
            "http_status": page["http_status"] if page else None,
            "selected_topk": is_selected,
            "text_content": text,
            "text_content_length": len(page["text"]) if page else 0,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None,
            "retrieved_at": page["retrieved_at"] if page else None,
            "search_result_date": page["http_date"] if page else None,
            "search_result_last_updated": page["http_last_modified"] if page else None,
        })
    return output


def collect_sources_from_url_file(
    input_path: str,
    output_path: str,
    rejected_output_path: Optional[str],
    cfg: ScrapeConfig,
    ai_model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fill `aggregated_sources` for every query in a get_urls.py output file."""
    with open(input_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        data = [data]

    rejected: List[Dict[str, Any]] = []
    output: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as pool:
        for record in data:
            query_id = record.get("query_id")
            candidates = aggregate_runs(record.get("runs") or [])
            sources = select_sources(candidates, cfg, pool)
            n_selected = sum(1 for s in sources if s["selected_topk"])
            print(f"query {query_id}: {len(candidates)} unique cited URLs, {n_selected} selected")
            for source in sources:
                if source["citation_existence_check"] is False:
                    rejected.append({
                        "query_id": query_id,
                        "url": source["url_examples"][0],
                        "status": source["existence_check_status"],
                        "http_status": source["http_status"],
                    })
            out_record = dict(record)
            out_record["aggregated_sources"] = sources
            if ai_model_name:
                out_record["ai_model_name"] = ai_model_name
            output.append(out_record)

    for path in (output_path, rejected_output_path):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {len(output)} queries to {output_path}")

    if rejected_output_path:
        with open(rejected_output_path, "w", encoding="utf-8") as handle:
            for row in rejected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rejected)} failed existence checks to {rejected_output_path}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step 2: union cited URLs across runs, run the existence check, "
            "scrape, and select the top-K sources per query."
        ),
    )
    parser.add_argument("--input", "-i", required=True, help="Output JSON of get_urls.py.")
    parser.add_argument("--output", "-o", required=True, help="Output JSON with aggregated_sources.")
    parser.add_argument("--rejected-output", help="Optional JSONL of URLs that failed the existence check.")
    parser.add_argument("--top-k", type=int, default=5, help="Sources selected per query (default: 5).")
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout in seconds (default: 15).")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=1,
        help="Minimum extracted text length to pass the existence check (default: 1, i.e. non-empty).",
    )
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel page fetches (default: 8).")
    parser.add_argument("--ai-model-name", help="Override ai_model_name in the output.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = ScrapeConfig(
        top_k=args.top_k,
        timeout=args.timeout,
        min_chars=args.min_chars,
        max_workers=args.max_workers,
    )
    collect_sources_from_url_file(
        input_path=args.input,
        output_path=args.output,
        rejected_output_path=args.rejected_output,
        cfg=cfg,
        ai_model_name=args.ai_model_name,
    )


if __name__ == "__main__":
    main()
