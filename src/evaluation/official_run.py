"""Run the server-side official SourceBench evaluation pipeline for one submission."""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
GET_URLS_SCRIPT = REPO_ROOT / "src" / "source-collection" / "get_urls.py"
COLLECT_SOURCES_SCRIPT = REPO_ROOT / "src" / "source-collection" / "collect_sources_from_urls.py"
SCORING_SCRIPT = REPO_ROOT / "src" / "content-scoring" / "scripts" / "scoring.py"
COMPUTE_METRICS_SCRIPT = REPO_ROOT / "src" / "evaluation" / "compute_metrics.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal SourceBench official runner. Loads an accepted submission, "
            "executes the current evaluation stages, and writes official artifacts."
        )
    )
    parser.add_argument(
        "--submission-dir",
        required=True,
        type=Path,
        help="Path to a submission intake directory produced by official_submission_backend.py.",
    )
    parser.add_argument(
        "--holdout-manifest",
        type=Path,
        help=(
            "Internal CSV for the official holdout split. Required for endpoint submissions "
            "and recommended for answer_url_bundle submissions."
        ),
    )
    parser.add_argument(
        "--query-metadata",
        type=Path,
        help="Optional CSV with query metadata for compute_metrics.py. Defaults to --holdout-manifest when provided.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Optional directory for runner outputs. Defaults to <submission-dir>/official_run.",
    )
    parser.add_argument(
        "--dry-run-judge",
        action="store_true",
        help="Run scoring.py with --dry-run to validate the pipeline without Qwen API calls.",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=5,
        help="Maximum number of URLs to request per query in endpoint mode.",
    )
    parser.add_argument(
        "--scrape-timeout",
        type=int,
        default=15,
        help="Timeout passed to collect_sources_from_urls.py.",
    )
    parser.add_argument(
        "--scrape-min-chars",
        type=int,
        default=500,
        help="Minimum character threshold passed to collect_sources_from_urls.py.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow running even if submission_status.json is not in a validated state.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a top-level JSON object in {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def require_existing_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def load_submission_payload(submission_dir: Path) -> Dict[str, Any]:
    raw_path = submission_dir / "submission.raw.json"
    redacted_path = submission_dir / "submission.redacted.json"
    if raw_path.exists():
        return load_json(raw_path)
    return load_json(redacted_path)


def validate_submission_state(submission_status: Dict[str, Any], force: bool) -> None:
    status = submission_status.get("status")
    if status in {"validated_pending_execution", "validated_pending_holdout_config", "validated_needs_review"}:
        return
    if force:
        return
    raise ValueError(
        f"Submission status `{status}` is not runnable by default. "
        "Use --force to override after review."
    )


def make_query_jsonl_from_csv(csv_path: Path, output_path: Path) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, newline="", encoding="utf-8") as handle, open(output_path, "w", encoding="utf-8") as out:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Query CSV has no header: {csv_path}")
        id_field = "query_ID" if "query_ID" in reader.fieldnames else "query_id"
        query_field = "query" if "query" in reader.fieldnames else "query_text"
        for row in reader:
            query_id = row.get(id_field)
            query_text = row.get(query_field)
            if not query_id or not query_text:
                continue
            out.write(json.dumps({"id": query_id, "query": query_text}, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_reference_records_from_answer_bundle(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    model_name = str(payload.get("model_name") or "Unknown")
    records: List[Dict[str, Any]] = []
    for run in payload.get("runs") or []:
        query_id = run.get("query_id")
        answer_text = run.get("answer_text") or ""
        query_text = run.get("query_text") or ""
        cited_urls = run.get("cited_urls") or []
        cited_sources = [{"type": "url", "url": url} for url in cited_urls if isinstance(url, str)]
        raw_response = run.get("raw_response")
        if raw_response is None:
            raw_response = {"type": "participant_bundle"}
        record = {
            "query_id": query_id,
            "query_text": query_text,
            "ai_model_name": model_name,
            "runs": [
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "ai_model_name": model_name,
                    "query_order": 1,
                    "response_text": answer_text,
                    "raw_response": raw_response,
                    "cited_sources": cited_sources,
                    "search_results": [],
                    "search_results_rank": {},
                    "usage": {},
                }
            ],
            "aggregated_sources": [],
        }
        records.append(record)
    return records


def run_command(command: List[str], env: Optional[Dict[str, str]] = None) -> None:
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def run_stage1_endpoint(
    payload: Dict[str, Any],
    holdout_manifest: Path,
    stage_dir: Path,
    max_urls: int,
) -> Path:
    if str(payload.get("api_key", "")).startswith("***REDACTED***"):
        raise ValueError(
            "Endpoint submission cannot be executed from a redacted payload. "
            "Re-run official_submission_backend.py with --store-secrets in the secure evaluation environment."
        )

    queries_jsonl = stage_dir / "holdout_queries.jsonl"
    record_count = make_query_jsonl_from_csv(holdout_manifest, queries_jsonl)
    if record_count == 0:
        raise ValueError(f"No runnable queries found in holdout manifest: {holdout_manifest}")

    output_path = stage_dir / "stage1_urls.json"
    command = [
        sys.executable,
        str(GET_URLS_SCRIPT),
        "--input",
        str(queries_jsonl),
        "--output",
        str(output_path),
        "--id-field",
        "id",
        "--query-field",
        "query",
        "--model",
        str(payload.get("model_name")),
        "--max-urls",
        str(max_urls),
        "--openai-base-url",
        str(payload.get("api_base")),
        "--openai-api-key",
        str(payload.get("api_key")),
        "--ai-model-name",
        str(payload.get("model_name")),
    ]
    run_command(command)
    return output_path


def run_stage1_answer_bundle(payload: Dict[str, Any], stage_dir: Path) -> Path:
    output_path = stage_dir / "stage1_urls.json"
    records = build_reference_records_from_answer_bundle(payload)
    if not records:
        raise ValueError("answer_url_bundle submission contains no runs")
    write_json(output_path, records)
    return output_path


def run_stage2_collect(stage1_output: Path, stage_dir: Path, model_name: str, timeout: int, min_chars: int) -> Path:
    output_path = stage_dir / "stage2_collected_sources.json"
    rejected_path = stage_dir / "stage2_rejected.jsonl"
    command = [
        sys.executable,
        str(COLLECT_SOURCES_SCRIPT),
        "--input",
        str(stage1_output),
        "--output",
        str(output_path),
        "--rejected-output",
        str(rejected_path),
        "--timeout",
        str(timeout),
        "--min-chars",
        str(min_chars),
        "--ai-model-name",
        model_name,
    ]
    run_command(command)
    return output_path


def run_stage3_score(stage2_output: Path, stage_dir: Path, model_name: str, dry_run_judge: bool) -> Path:
    output_dir = stage_dir / "scored"
    run_name = f"{model_name.replace(' ', '_')}_official"
    command = [
        sys.executable,
        str(SCORING_SCRIPT),
        "--input-file",
        str(stage2_output),
        "--out-dir",
        str(output_dir),
        "--run-name",
        run_name,
    ]
    if dry_run_judge:
        command.append("--dry-run")
    run_command(command)
    return output_dir / f"{run_name}.enriched.json"


def run_stage4_metrics(scored_output: Path, stage_dir: Path, model_name: str, query_metadata: Optional[Path]) -> Path:
    output_dir = stage_dir / "metrics"
    command = [
        sys.executable,
        str(COMPUTE_METRICS_SCRIPT),
        "--run",
        f"{model_name}={scored_output}",
        "--out-dir",
        str(output_dir),
    ]
    if query_metadata:
        command.extend(["--query-metadata", str(query_metadata)])
    run_command(command)
    return output_dir


def update_submission_status(submission_dir: Path, status_payload: Dict[str, Any], new_status: str) -> None:
    updated = dict(status_payload)
    updated["status"] = new_status
    updated["updated_at"] = utc_now()
    write_json(submission_dir / "submission_status.json", updated)


def main() -> None:
    args = parse_args()
    submission_dir = args.submission_dir.resolve()
    status_path = require_existing_file(submission_dir / "submission_status.json", "submission_status.json")
    require_existing_file(submission_dir / "validation_report.json", "validation_report.json")

    submission_status = load_json(status_path)
    validate_submission_state(submission_status, args.force)
    payload = load_submission_payload(submission_dir)

    submission_mode = payload.get("submission_mode")
    model_name = str(payload.get("model_name") or "Unknown")
    run_root = (args.run_root or (submission_dir / "official_run")).resolve()
    stage1_dir = run_root / "stage1"
    stage2_dir = run_root / "stage2"
    stage3_dir = run_root / "stage3"
    stage4_dir = run_root / "stage4"
    query_metadata = args.query_metadata or args.holdout_manifest

    run_manifest = {
        "submission_id": submission_status.get("submission_id"),
        "submission_mode": submission_mode,
        "model_name": model_name,
        "started_at": utc_now(),
        "run_root": str(run_root),
        "stages": {},
    }
    write_json(run_root / "official_run_manifest.json", run_manifest)
    update_submission_status(submission_dir, submission_status, "official_run_in_progress")

    try:
        if submission_mode == "endpoint":
            if not args.holdout_manifest:
                raise ValueError("--holdout-manifest is required for endpoint submissions")
            stage1_output = run_stage1_endpoint(payload, args.holdout_manifest, stage1_dir, args.max_urls)
        elif submission_mode == "answer_url_bundle":
            stage1_output = run_stage1_answer_bundle(payload, stage1_dir)
        else:
            raise ValueError(f"Unsupported submission_mode: {submission_mode}")

        run_manifest["stages"]["stage1"] = {"status": "completed", "output": str(stage1_output)}
        write_json(run_root / "official_run_manifest.json", run_manifest)

        stage2_output = run_stage2_collect(
            stage1_output=stage1_output,
            stage_dir=stage2_dir,
            model_name=model_name,
            timeout=args.scrape_timeout,
            min_chars=args.scrape_min_chars,
        )
        run_manifest["stages"]["stage2"] = {
            "status": "completed",
            "output": str(stage2_output),
            "rejected_output": str(stage2_dir / "stage2_rejected.jsonl"),
        }
        write_json(run_root / "official_run_manifest.json", run_manifest)

        stage3_output = run_stage3_score(
            stage2_output=stage2_output,
            stage_dir=stage3_dir,
            model_name=model_name,
            dry_run_judge=args.dry_run_judge,
        )
        run_manifest["stages"]["stage3"] = {
            "status": "completed",
            "output": str(stage3_output),
            "dry_run_judge": args.dry_run_judge,
        }
        write_json(run_root / "official_run_manifest.json", run_manifest)

        stage4_output = run_stage4_metrics(
            scored_output=stage3_output,
            stage_dir=stage4_dir,
            model_name=model_name,
            query_metadata=query_metadata,
        )
        run_manifest["stages"]["stage4"] = {"status": "completed", "output_dir": str(stage4_output)}
        run_manifest["completed_at"] = utc_now()
        run_manifest["status"] = "completed"
        write_json(run_root / "official_run_manifest.json", run_manifest)
        update_submission_status(submission_dir, submission_status, "official_run_completed")

        print(
            json.dumps(
                {
                    "submission_id": submission_status.get("submission_id"),
                    "status": "official_run_completed",
                    "run_root": str(run_root),
                    "leaderboard_data": str(stage4_output / "leaderboard_data.json"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    except Exception as exc:
        run_manifest["status"] = "failed"
        run_manifest["failed_at"] = utc_now()
        run_manifest["error"] = str(exc)
        write_json(run_root / "official_run_manifest.json", run_manifest)
        update_submission_status(submission_dir, submission_status, "official_run_failed")
        raise


if __name__ == "__main__":
    main()
