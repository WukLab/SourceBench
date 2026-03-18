"""Intake a validated official submission into the internal evaluation queue."""

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from validate_official_submission import load_json, validate_submission_payload


DEFAULT_SUBMISSIONS_DIR = (
    Path(__file__).resolve().parents[2] / "leaderboard" / ".official_submissions"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal official submission intake backend for SourceBench. "
            "Validates a submission, assigns a submission id, writes status artifacts, "
            "and creates a pending official evaluation record."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to submission JSON.",
    )
    parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=DEFAULT_SUBMISSIONS_DIR,
        help="Directory where submission intake artifacts will be stored.",
    )
    parser.add_argument(
        "--holdout-manifest",
        type=Path,
        help=(
            "Optional internal holdout manifest path. "
            "If omitted, uses SOURCEBENCH_HOLDOUT_MANIFEST when available."
        ),
    )
    parser.add_argument(
        "--store-secrets",
        action="store_true",
        help="Store the raw submission including secrets. Default behavior stores only a redacted copy.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-") or "submission"


def make_submission_id(payload: Dict[str, Any]) -> str:
    base = f"{payload.get('model_name', 'submission')}|{payload.get('submitter_name', 'unknown')}|{utc_now()}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:10]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = slugify(str(payload.get("model_name", "submission")))
    return f"{timestamp}-{slug}-{digest}"


def redact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    redacted = json.loads(json.dumps(payload))
    for key in ("api_key",):
        if key in redacted:
            redacted[key] = "***REDACTED***"
    return redacted


def resolve_holdout_manifest(cli_path: Optional[Path]) -> Optional[Path]:
    if cli_path:
        return cli_path
    env_value = os.getenv("SOURCEBENCH_HOLDOUT_MANIFEST")
    if env_value:
        return Path(env_value).expanduser()
    return None


def load_holdout_query_count(path: Optional[Path]) -> Optional[int]:
    if path is None or not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def extend_report_with_holdout_checks(
    payload: Dict[str, Any],
    report: Dict[str, Any],
    holdout_query_count: Optional[int],
) -> Dict[str, Any]:
    extended = json.loads(json.dumps(report))
    extended["holdout_query_count"] = holdout_query_count

    if payload.get("submission_mode") == "answer_url_bundle" and holdout_query_count is not None:
        runs = payload.get("runs") or []
        if len(runs) != holdout_query_count:
            extended["warning_count"] += 1
            extended["warnings"].append(
                f"Expected {holdout_query_count} runs for the current holdout configuration, "
                f"but received {len(runs)}."
            )
    return extended


def compute_submission_status(payload: Dict[str, Any], report: Dict[str, Any], holdout_query_count: Optional[int]) -> str:
    if not report["valid"]:
        return "rejected_validation"

    if holdout_query_count is None:
        return "validated_pending_holdout_config"

    if payload.get("submission_mode") == "answer_url_bundle":
        runs = payload.get("runs") or []
        if len(runs) != holdout_query_count:
            return "validated_needs_review"

    return "validated_pending_execution"


def append_queue_record(queue_file: Path, record: Dict[str, Any]) -> None:
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    with open(queue_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_submission_record(
    submission_id: str,
    status: str,
    payload: Dict[str, Any],
    report: Dict[str, Any],
    holdout_manifest: Optional[Path],
) -> Dict[str, Any]:
    return {
        "submission_id": submission_id,
        "created_at": utc_now(),
        "status": status,
        "submission_mode": payload.get("submission_mode"),
        "model_name": payload.get("model_name"),
        "submitter_name": payload.get("submitter_name"),
        "contact_email": payload.get("contact_email"),
        "valid": report["valid"],
        "error_count": report["error_count"],
        "warning_count": report["warning_count"],
        "holdout_manifest": str(holdout_manifest) if holdout_manifest else None,
    }


def build_evaluation_request(
    submission_id: str,
    status: str,
    payload: Dict[str, Any],
    holdout_query_count: Optional[int],
) -> Dict[str, Any]:
    return {
        "submission_id": submission_id,
        "created_at": utc_now(),
        "status": status,
        "submission_mode": payload.get("submission_mode"),
        "model_name": payload.get("model_name"),
        "web_search_mode": payload.get("web_search_mode"),
        "holdout_query_count": holdout_query_count,
        "next_step": "Run official evaluation pipeline if status is validated_pending_execution.",
    }


def main() -> None:
    args = parse_args()
    payload = load_json(args.input)
    report = validate_submission_payload(payload)

    holdout_manifest = resolve_holdout_manifest(args.holdout_manifest)
    holdout_query_count = load_holdout_query_count(holdout_manifest)
    report = extend_report_with_holdout_checks(payload, report, holdout_query_count)

    submission_id = make_submission_id(payload)
    status = compute_submission_status(payload, report, holdout_query_count)
    submission_record = build_submission_record(submission_id, status, payload, report, holdout_manifest)
    evaluation_request = build_evaluation_request(submission_id, status, payload, holdout_query_count)

    submission_dir = args.submissions_dir / submission_id
    submission_dir.mkdir(parents=True, exist_ok=True)

    write_json(submission_dir / "validation_report.json", report)
    write_json(submission_dir / "submission_status.json", submission_record)
    write_json(submission_dir / "evaluation_request.json", evaluation_request)
    write_json(submission_dir / "submission.redacted.json", redact_payload(payload))

    if args.store_secrets:
        write_json(submission_dir / "submission.raw.json", payload)

    queue_record = {
        "submission_id": submission_id,
        "created_at": submission_record["created_at"],
        "status": status,
        "model_name": payload.get("model_name"),
        "submission_mode": payload.get("submission_mode"),
    }
    append_queue_record(args.submissions_dir / "queue.jsonl", queue_record)

    print(json.dumps({
        "submission_id": submission_id,
        "status": status,
        "submission_dir": str(submission_dir),
        "valid": report["valid"],
        "error_count": report["error_count"],
        "warning_count": report["warning_count"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
