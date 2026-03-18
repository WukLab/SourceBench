import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Output schema matches data/collected-sources/gpt-5/GE_Response_Data_5_web_search_scored_deduped_with_avg_ge_freq.json
# Top-level: [ { query_id, query_text, ai_model_name, runs: [ { query_id, query_text, ai_model_name, query_order, response_text, raw_response, cited_sources, search_results, search_results_rank, usage } ], aggregated_sources: [] } ]


@dataclass
class URLGenerationConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_urls: int = 5
    max_retries: int = 2
    sleep_seconds: float = 0.5
    system_prompt: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


DEFAULT_SYSTEM_PROMPT = (
    "You are a research assistant that suggests high-quality, trustworthy web sources "
    "for answering user questions.\n\n"
    "For each user query you receive, you MUST return up to {max_urls} URLs that are:\n"
    "- from reputable sources (official documentation, research groups, major news outlets, etc.)\n"
    "- as direct and specific as possible for answering the question\n\n"
    "Important formatting requirements:\n"
    "- Respond with a single JSON object only.\n"
    '- The JSON must have the form: {{"urls": [{{"url": "https://...", "source_type": "short description", "notes": "optional notes"}}]}}.\n'
    "- Do not add any extra keys.\n"
    "- Do not include markdown, commentary, or explanations outside the JSON.\n"
)


def build_system_prompt(cfg: URLGenerationConfig) -> str:
    if cfg.system_prompt:
        return cfg.system_prompt
    return DEFAULT_SYSTEM_PROMPT.format(max_urls=cfg.max_urls)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from an LLM response that may contain extra text.
    Raises ValueError if a JSON object cannot be parsed.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON object in LLM response:\n{text}")
    snippet = text[start : end + 1]
    return json.loads(snippet)


def call_openai_for_urls(
    client: OpenAI,
    query: str,
    cfg: URLGenerationConfig,
) -> List[Dict[str, Any]]:
    """
    Call the OpenAI chat completion API to get URLs for a single query.
    Returns a list of URL dicts with at least key 'url'.
    """
    system_prompt = build_system_prompt(cfg)
    user_prompt = (
        "User query:\n"
        f"{query}\n\n"
        "Return only the JSON object as specified in the instructions."
    )

    last_error: Optional[Exception] = None
    for attempt in range(cfg.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=cfg.model,
                temperature=cfg.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            data = _extract_json_object(content)
            urls = data.get("urls") or []
            # Normalize to list of dicts with at least "url"
            normalized: List[Dict[str, Any]] = []
            for item in urls:
                if isinstance(item, str):
                    normalized.append({"url": item})
                elif isinstance(item, dict) and "url" in item:
                    normalized.append(item)
            return normalized[: cfg.max_urls]
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            if attempt < cfg.max_retries:
                time.sleep(cfg.sleep_seconds)
            else:
                raise
    # Should not reach here because of raise above
    raise RuntimeError(f"Failed to get URLs for query after retries: {last_error}")


def _build_raw_response_skeleton(
    model: str,
    query_text: str,
    url_list: List[Dict[str, Any]],
    created_at: Optional[float] = None,
) -> Dict[str, Any]:
    """Build raw_response dict matching reference schema (all keys present)."""
    if created_at is None:
        created_at = time.time()
    sources = [{"type": "url", "url": u.get("url", u) if isinstance(u, dict) else u} for u in url_list]
    output_item = {
        "id": None,
        "status": "completed",
        "type": "web_search_call",
        "action": {
            "type": "search",
            "queries": [query_text],
            "query": query_text,
            "sources": sources,
        },
    }
    return {
        "id": None,
        "created_at": created_at,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "model": model,
        "object": "response",
        "output": [output_item],
        "parallel_tool_calls": False,
        "temperature": 0.2,
        "tool_choice": None,
        "tools": [],
        "top_p": 1.0,
        "max_output_tokens": None,
        "reasoning": None,
        "service_tier": "default",
        "status": "completed",
        "text": None,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {},
            "total_tokens": 0,
        },
        "user": None,
        "background": False,
        "billing": {},
        "completed_at": int(created_at) + 1,
        "frequency_penalty": 0.0,
        "max_tool_calls": None,
        "presence_penalty": 0.0,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "prompt_cache_retention": None,
        "safety_identifier": None,
        "store": False,
        "top_logprobs": 0,
    }


def _build_cited_sources(url_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build cited_sources list from URL list (reference uses same shape as raw_response output sources)."""
    return [{"type": "url", "url": u.get("url", u) if isinstance(u, dict) else u} for u in url_list]


def _build_run(
    query_id: Any,
    query_text: str,
    ai_model_name: str,
    model: str,
    url_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build one run object matching reference schema (all keys present)."""
    cited = _build_cited_sources(url_list)
    return {
        "query_id": query_id,
        "query_text": query_text,
        "ai_model_name": ai_model_name,
        "query_order": 1,
        "response_text": "",
        "raw_response": _build_raw_response_skeleton(model, query_text, url_list),
        "cited_sources": cited,
        "search_results": [],
        "search_results_rank": {},
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {},
            "total_tokens": 0,
        },
    }


def generate_urls_for_queries(
    input_path: str,
    output_path: str,
    id_field: str,
    query_field: str,
    cfg: URLGenerationConfig,
    ai_model_name: str = "OpenAI",
) -> None:
    """
    Read queries from a JSONL file and write URL suggestions in reference format.

    Input JSONL format (per line):
        { "<id_field>": "...", "<query_field>": "..." , ... }

    Output: single JSON file (array) matching
    data/collected-sources/gpt-5/GE_Response_Data_5_web_search_scored_deduped_with_avg_ge_freq.json
    Each element: { query_id, query_text, ai_model_name, runs: [ one run ], aggregated_sources: [] }
    """
    client_kwargs: Dict[str, Any] = {}
    if cfg.base_url:
        client_kwargs["base_url"] = cfg.base_url
    if cfg.api_key:
        client_kwargs["api_key"] = cfg.api_key
    client = OpenAI(**client_kwargs)

    results: List[Dict[str, Any]] = []
    for row in _iter_jsonl(input_path):
        if id_field not in row or query_field not in row:
            continue
        query_id = row[id_field]
        query_text = row[query_field]
        url_list = call_openai_for_urls(client, query_text, cfg)
        run = _build_run(query_id, query_text, ai_model_name, cfg.model, url_list)
        record = {
            "query_id": query_id,
            "query_text": query_text,
            "ai_model_name": ai_model_name,
            "runs": [run],
            "aggregated_sources": [],
        }
        results.append(record)
        if cfg.sleep_seconds > 0:
            time.sleep(cfg.sleep_seconds)

    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate candidate source URLs for queries using an OpenAI LLM.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input JSONL file containing queries.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output JSON file (reference format: array of query records with runs, empty aggregated_sources).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI chat model name (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the model (default: 0.2).",
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
        help="Maximum number of retries per query on API failure (default: 2).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Seconds to sleep between API calls (default: 0.5).",
    )
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
        help="Value for ai_model_name in output (default: OpenAI).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = URLGenerationConfig(
        model=args.model,
        temperature=args.temperature,
        max_urls=args.max_urls,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        base_url=args.openai_base_url,
        api_key=args.openai_api_key,
    )
    generate_urls_for_queries(
        input_path=args.input,
        output_path=args.output,
        id_field=args.id_field,
        query_field=args.query_field,
        cfg=cfg,
        ai_model_name=args.ai_model_name,
    )


if __name__ == "__main__":
    main()

