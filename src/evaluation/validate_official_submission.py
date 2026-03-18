"""Validate official SourceBench submission JSON before any server-side execution."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse


ENDPOINT_REQUIRED_FIELDS = [
    "submission_mode",
    "submitter_name",
    "contact_email",
    "model_name",
    "api_base",
    "api_key",
    "api_format",
    "web_search_mode",
    "agrees_to_reproducibility_policy",
]

ANSWER_BUNDLE_REQUIRED_FIELDS = [
    "submission_mode",
    "submitter_name",
    "contact_email",
    "model_name",
    "web_search_mode",
    "agrees_to_reproducibility_policy",
    "runs",
]

RUN_REQUIRED_FIELDS = [
    "query_id",
    "answer_text",
    "cited_urls",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an official SourceBench submission bundle. "
            "Supports `endpoint` and `answer_url_bundle` submission modes."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to submission JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write a validation report JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Submission JSON must be a top-level object.")
    return payload


def is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_valid_email(value: Any) -> bool:
    return is_non_empty_string(value) and "@" in value and "." in value.split("@")[-1]


def is_http_url(value: Any) -> bool:
    if not is_non_empty_string(value):
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def add_error(errors: List[str], message: str) -> None:
    errors.append(message)


def add_warning(warnings: List[str], message: str) -> None:
    warnings.append(message)


def require_fields(payload: Dict[str, Any], fields: List[str], errors: List[str]) -> None:
    for field in fields:
        if field not in payload:
            add_error(errors, f"Missing required field: `{field}`")


def validate_common_fields(payload: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    if not is_non_empty_string(payload.get("submitter_name")):
        add_error(errors, "`submitter_name` must be a non-empty string")
    if not is_valid_email(payload.get("contact_email")):
        add_error(errors, "`contact_email` must be a valid email-like string")
    if not is_non_empty_string(payload.get("model_name")):
        add_error(errors, "`model_name` must be a non-empty string")
    if payload.get("agrees_to_reproducibility_policy") is not True:
        add_error(errors, "`agrees_to_reproducibility_policy` must be true")
    if not is_non_empty_string(payload.get("web_search_mode")):
        add_error(errors, "`web_search_mode` must be a non-empty string")
    if "model_version" not in payload:
        add_warning(warnings, "Recommended field missing: `model_version`")
    if "organization" not in payload:
        add_warning(warnings, "Recommended field missing: `organization`")


def validate_endpoint_submission(payload: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    require_fields(payload, ENDPOINT_REQUIRED_FIELDS, errors)
    validate_common_fields(payload, errors, warnings)

    if payload.get("submission_mode") != "endpoint":
        add_error(errors, "`submission_mode` must equal `endpoint`")
    if not is_http_url(payload.get("api_base")):
        add_error(errors, "`api_base` must be a valid http(s) URL")
    if not is_non_empty_string(payload.get("api_key")):
        add_error(errors, "`api_key` must be a non-empty string")
    if payload.get("api_format") != "openai-compatible":
        add_error(errors, "`api_format` must equal `openai-compatible` for v1")

    generation_config = payload.get("generation_config")
    if generation_config is not None and not isinstance(generation_config, dict):
        add_error(errors, "`generation_config` must be an object when provided")

    system_prompt = payload.get("system_prompt")
    if system_prompt is not None and not isinstance(system_prompt, (str, type(None))):
        add_error(errors, "`system_prompt` must be a string or null")


def validate_run(run: Dict[str, Any], index: int, errors: List[str], warnings: List[str]) -> None:
    for field in RUN_REQUIRED_FIELDS:
        if field not in run:
            add_error(errors, f"`runs[{index}]` missing required field `{field}`")

    query_id = run.get("query_id")
    if not isinstance(query_id, (int, str)):
        add_error(errors, f"`runs[{index}].query_id` must be an int or string")

    if not is_non_empty_string(run.get("answer_text")):
        add_error(errors, f"`runs[{index}].answer_text` must be a non-empty string")

    cited_urls = run.get("cited_urls")
    if not isinstance(cited_urls, list) or not cited_urls:
        add_error(errors, f"`runs[{index}].cited_urls` must be a non-empty list")
    else:
        for url_index, url in enumerate(cited_urls):
            if not is_http_url(url):
                add_error(errors, f"`runs[{index}].cited_urls[{url_index}]` must be a valid http(s) URL")

    raw_response = run.get("raw_response")
    if raw_response is None:
        add_warning(warnings, f"`runs[{index}].raw_response` is recommended but missing")
    elif not isinstance(raw_response, (dict, list, str, int, float, bool, type(None))):
        add_error(errors, f"`runs[{index}].raw_response` must be JSON-serializable")


def validate_answer_bundle_submission(payload: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    require_fields(payload, ANSWER_BUNDLE_REQUIRED_FIELDS, errors)
    validate_common_fields(payload, errors, warnings)

    if payload.get("submission_mode") != "answer_url_bundle":
        add_error(errors, "`submission_mode` must equal `answer_url_bundle`")

    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        add_error(errors, "`runs` must be a non-empty list")
        return

    seen_query_ids = set()
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
          add_error(errors, f"`runs[{index}]` must be an object")
          continue
        validate_run(run, index, errors, warnings)
        query_id = run.get("query_id")
        if query_id in seen_query_ids:
            add_warning(warnings, f"Duplicate query_id detected in runs: `{query_id}`")
        seen_query_ids.add(query_id)


def build_report(payload: Dict[str, Any], errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    return {
        "submission_mode": payload.get("submission_mode"),
        "model_name": payload.get("model_name"),
        "valid": len(errors) == 0,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def validate_submission_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    mode = payload.get("submission_mode")
    if mode == "endpoint":
        validate_endpoint_submission(payload, errors, warnings)
    elif mode == "answer_url_bundle":
        validate_answer_bundle_submission(payload, errors, warnings)
    else:
        add_error(errors, "`submission_mode` must be either `endpoint` or `answer_url_bundle`")

    return build_report(payload, errors, warnings)


def main() -> None:
    args = parse_args()
    payload = load_json(args.input)
    report = validate_submission_payload(payload)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if report["valid"]:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
