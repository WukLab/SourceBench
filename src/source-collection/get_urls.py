"""Step 1 of SourceBench source collection: query a search-enabled system.

Each query is sent to the system ``--runs`` times (default 3). For every run we
store the final answer, the raw API response, and the URLs the answer cites, in
citation order. URLs the system visited while searching but did not cite in its
final answer are not kept.

Step 2 (``collect_sources_from_urls.py``) takes the union across runs, runs the
existence check, and selects the top-K sources.

Provider adaptation
-------------------
``query_system()`` is the only provider-specific function. This reference
implementation calls the OpenAI Responses API with the hosted web-search tool
and reads the ``url_citation`` annotations of the final message. To collect
from another provider, replace ``query_system()`` with a function that returns
a ``SystemResponse`` (answer text, cited URLs in order, raw response).

Output: one JSON array, one record per query:
    { query_id, query_text, ai_model_name,
      runs: [ { query_order, response_text, raw_response, cited_sources: [ {url, title, ge_rank_in_run} ],
                usage, error } ],
      aggregated_sources: [] }
"""

import argparse
import json
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from url_utils import canonicalize_url

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    load_dotenv = None


@dataclass
class CollectionConfig:
    model: str
    runs: int = 3
    tool_type: str = "web_search"
    tool_choice: str = "required"
    temperature: Optional[float] = None
    max_retries: int = 3
    max_urls: int = 0  # 0 keeps every cited URL
    base_url: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class SystemResponse:
    response_text: str
    citations: List[Dict[str, str]]  # [{"url": ..., "title": ...}] in citation order
    raw_response: Dict[str, Any]
    usage: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provider-specific part (OpenAI Responses API + web search)
# ---------------------------------------------------------------------------

def make_client(cfg: CollectionConfig) -> Any:
    from openai import OpenAI  # imported here so --help works without openai installed

    kwargs: Dict[str, Any] = {}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    return OpenAI(**kwargs)


def _to_dict(response: Any) -> Dict[str, Any]:
    for method in ("to_dict", "model_dump", "dict"):
        fn = getattr(response, method, None)
        if callable(fn):
            return fn()
    return dict(response)


_URL_IN_TEXT = re.compile(r"https?://[^\s<>\"')\]]+")


def extract_answer_and_citations(raw: Dict[str, Any]) -> tuple:
    """Return (answer_text, citations) from a Responses API payload.

    Citations are the `url_citation` annotations of the final message, ordered
    by their position in the answer. If the answer has no annotations, URLs
    written in the answer text are used instead.
    """
    texts: List[str] = []
    citations: List[Dict[str, str]] = []
    for item in raw.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("text"):
                texts.append(part["text"])
            annotations = [
                ann for ann in part.get("annotations") or []
                if isinstance(ann, dict) and ann.get("type") == "url_citation" and ann.get("url")
            ]
            annotations.sort(key=lambda ann: ann.get("start_index") or 0)
            for ann in annotations:
                citations.append({"url": ann["url"], "title": ann.get("title") or ""})
    answer = "\n".join(texts)
    if not citations and answer:
        for url in _URL_IN_TEXT.findall(answer):
            citations.append({"url": url.rstrip(".,;:"), "title": ""})
    return answer, citations


def query_system(client: Any, query: str, cfg: CollectionConfig) -> SystemResponse:
    """Send one query to the system and return its answer and cited URLs.

    Replace this function to collect from a provider other than OpenAI.
    The query is sent as-is, without any added instructions.
    """
    request: Dict[str, Any] = {
        "model": cfg.model,
        "input": query,
        "tools": [{"type": cfg.tool_type}],
    }
    if cfg.tool_choice != "auto":
        request["tool_choice"] = cfg.tool_choice
    if cfg.temperature is not None:
        request["temperature"] = cfg.temperature
    raw = _to_dict(client.responses.create(**request))
    answer, citations = extract_answer_and_citations(raw)
    return SystemResponse(
        response_text=answer,
        citations=citations,
        raw_response=raw,
        usage=raw.get("usage") or {},
    )


# ---------------------------------------------------------------------------
# Provider-independent part
# ---------------------------------------------------------------------------

def rank_citations(citations: List[Dict[str, str]], max_urls: int = 0) -> List[Dict[str, Any]]:
    """Deduplicate by canonical URL and assign 1-based ranks in citation order."""
    ranked: List[Dict[str, Any]] = []
    seen = set()
    for citation in citations:
        canonical = canonicalize_url(citation["url"])
        if canonical in seen:
            continue
        seen.add(canonical)
        ranked.append({
            "url": citation["url"],
            "title": citation.get("title") or "",
            "ge_rank_in_run": len(ranked) + 1,
        })
        if max_urls and len(ranked) >= max_urls:
            break
    return ranked


def run_once(client: Any, query_id: Any, query_text: str, query_order: int,
             ai_model_name: str, cfg: CollectionConfig) -> Dict[str, Any]:
    error: Optional[str] = None
    response: Optional[SystemResponse] = None
    for attempt in range(cfg.max_retries + 1):
        try:
            response = query_system(client, query_text, cfg)
            error = None
            break
        except Exception as exc:  # pylint: disable=broad-except
            error = f"{type(exc).__name__}: {exc}"
            if attempt < cfg.max_retries:
                time.sleep(2 ** attempt)
    if response is None:
        print(f"  query {query_id} run {query_order}: failed ({error})", file=sys.stderr)
        return {
            "query_id": query_id,
            "query_text": query_text,
            "ai_model_name": ai_model_name,
            "query_order": query_order,
            "response_text": "",
            "raw_response": {},
            "cited_sources": [],
            "usage": {},
            "error": error,
        }
    cited = rank_citations(response.citations, cfg.max_urls)
    print(f"  query {query_id} run {query_order}: {len(cited)} cited URLs")
    return {
        "query_id": query_id,
        "query_text": query_text,
        "ai_model_name": ai_model_name,
        "query_order": query_order,
        "response_text": response.response_text,
        "raw_response": response.raw_response,
        "cited_sources": cited,
        "usage": response.usage,
        "error": None,
    }


def collect_query(client: Any, query_id: Any, query_text: str, ai_model_name: str,
                  cfg: CollectionConfig) -> Dict[str, Any]:
    runs = [
        run_once(client, query_id, query_text, order, ai_model_name, cfg)
        for order in range(1, cfg.runs + 1)
    ]
    return {
        "query_id": query_id,
        "query_text": query_text,
        "ai_model_name": ai_model_name,
        "runs": runs,
        "aggregated_sources": [],
    }


def iter_queries(path: str, id_field: str, query_field: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if id_field in row and query_field in row:
                yield {"query_id": row[id_field], "query_text": row[query_field]}


def _write_json_atomic(path: str, payload: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _is_complete(record: Dict[str, Any], runs: int) -> bool:
    record_runs = record.get("runs") or []
    return len(record_runs) == runs and all(not run.get("error") for run in record_runs)


def generate_urls_for_queries(
    input_path: str,
    output_path: str,
    cfg: CollectionConfig,
    id_field: str = "id",
    query_field: str = "query",
    ai_model_name: Optional[str] = None,
    query_ids: Optional[List[Any]] = None,
    max_workers: int = 4,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    """Run step 1 for every query in a JSONL file and write the JSON output.

    The output file is rewritten after each finished query, so an interrupted
    run can continue with ``resume=True``.
    """
    model_label = ai_model_name or cfg.model
    queries = list(iter_queries(input_path, id_field, query_field))
    if query_ids:
        wanted = {str(qid) for qid in query_ids}
        queries = [q for q in queries if str(q["query_id"]) in wanted]

    # With --resume, keep every record already in the output file and redo only
    # the requested queries that are missing or have failed runs.
    results: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as handle:
            for record in json.load(handle):
                results[str(record["query_id"])] = record
                order.append(str(record["query_id"]))
        complete = sum(1 for record in results.values() if _is_complete(record, cfg.runs))
        print(f"Resuming: {complete} of {len(results)} queries in {output_path} are complete")
    order += [str(q["query_id"]) for q in queries if str(q["query_id"]) not in results]

    todo = [
        q for q in queries
        if str(q["query_id"]) not in results or not _is_complete(results[str(q["query_id"])], cfg.runs)
    ]
    print(f"Collecting {len(todo)} queries x {cfg.runs} runs with model={cfg.model}")

    client = make_client(cfg)
    lock = threading.Lock()

    def snapshot() -> List[Dict[str, Any]]:
        return [results[qid] for qid in order if qid in results]

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {
            pool.submit(collect_query, client, q["query_id"], q["query_text"], model_label, cfg): q
            for q in todo
        }
        for future in as_completed(futures):
            record = future.result()
            with lock:
                results[str(record["query_id"])] = record
                _write_json_atomic(output_path, snapshot())

    final = snapshot()
    _write_json_atomic(output_path, final)
    failed = sum(1 for record in final for run in record["runs"] if run.get("error"))
    print(f"Wrote {len(final)} queries to {output_path} ({failed} failed runs)")
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step 1: query a search-enabled system several times per query and record "
            "the URLs cited in its final answers (reference implementation: OpenAI "
            "Responses API with web search)."
        ),
    )
    parser.add_argument("--input", "-i", required=True, help="Query file (JSONL).")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file.")
    parser.add_argument("--model", required=True, help="Model name, e.g. gpt-5 or gpt-4o.")
    parser.add_argument("--runs", type=int, default=3, help="Runs per query (default: 3).")
    parser.add_argument(
        "--tool-type",
        default="web_search",
        help="Hosted search tool type sent to the API (default: web_search; "
             "some accounts or models need web_search_preview).",
    )
    parser.add_argument(
        "--tool-choice",
        choices=["required", "auto"],
        default="required",
        help="'required' forces the model to search before answering (default).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Omitted by default (reasoning models use a fixed temperature).",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per run on API errors.")
    parser.add_argument("--max-workers", type=int, default=4, help="Queries processed in parallel.")
    parser.add_argument(
        "--max-urls",
        type=int,
        default=0,
        help="Keep at most this many cited URLs per run (default: 0 = keep all).",
    )
    parser.add_argument("--query-ids", nargs="+", help="Only collect these query ids.")
    parser.add_argument("--resume", action="store_true", help="Skip queries already complete in --output.")
    parser.add_argument("--id-field", default="id", help="Query id field in the JSONL (default: id).")
    parser.add_argument("--query-field", default="query", help="Query text field (default: query).")
    parser.add_argument("--openai-base-url", help="Custom base URL for an OpenAI-compatible Responses API.")
    parser.add_argument("--openai-api-key", help="API key (default: OPENAI_API_KEY).")
    parser.add_argument("--ai-model-name", help="Label stored as ai_model_name (default: --model).")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    if load_dotenv is not None:
        load_dotenv()
    args = build_parser().parse_args(argv)
    cfg = CollectionConfig(
        model=args.model,
        runs=args.runs,
        tool_type=args.tool_type,
        tool_choice=args.tool_choice,
        temperature=args.temperature,
        max_retries=args.max_retries,
        max_urls=args.max_urls,
        base_url=args.openai_base_url,
        api_key=args.openai_api_key,
    )
    generate_urls_for_queries(
        input_path=args.input,
        output_path=args.output,
        cfg=cfg,
        id_field=args.id_field,
        query_field=args.query_field,
        ai_model_name=args.ai_model_name,
        query_ids=args.query_ids,
        max_workers=args.max_workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
