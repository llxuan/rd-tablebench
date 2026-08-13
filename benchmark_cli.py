"""Stable provider + evaluation CLI for RD-TableBench."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from convert import html_to_numpy
from grading import table_similarity
from parsing import (
    parse_azure_content_understanding_response,
    parse_mistral_ocr_response,
)
from providers.azure_content_understanding import analyze as analyze_azure_cu
from providers.mistral_ocr import analyze as analyze_mistral


@dataclass(frozen=True)
class Case:
    id: str
    pdf: str
    image: str
    ground_truth: str
    language: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _read_cases(dataset_root: Path) -> list[Case]:
    scores_path = dataset_root / "providers" / "scores.csv"
    if not scores_path.is_file():
        raise FileNotFoundError(f"RD-TableBench metadata is missing: {scores_path}")
    cases: list[Case] = []
    with scores_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pdf_name = row.get("pdf_path", "").strip()
            if not pdf_name:
                raise ValueError("scores.csv contains an empty pdf_path.")
            case_id = Path(pdf_name).stem
            case = Case(
                id=case_id,
                pdf=f"pdfs/{pdf_name}",
                image=f"_images/{case_id}.jpg",
                ground_truth=f"groundtruth/{case_id}.html",
                language=row.get("language", "").strip() or "unknown",
            )
            missing = [
                relative
                for relative in (case.pdf, case.image, case.ground_truth)
                if not (dataset_root / relative).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"RD-TableBench case {case.id!r} is incomplete: {', '.join(missing)}"
                )
            cases.append(case)
    cases.sort(key=lambda case: case.id)
    if not cases:
        raise ValueError("RD-TableBench contains no complete cases.")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("RD-TableBench contains duplicate case IDs.")
    return cases


def _configuration(args: argparse.Namespace) -> dict[str, object]:
    if args.provider == "azure-cu":
        return {
            "provider": args.provider,
            "parallel": args.parallel,
            "analyzer_id": args.analyzer_id,
        }
    return {
        "provider": args.provider,
        "parallel": args.parallel,
        "mistral_provider": args.mistral_provider,
        "model": args.mistral_model,
    }


def _validate_environment(args: argparse.Namespace) -> None:
    required = (
        ["AZURE_CONTENT_UNDERSTANDING_ENDPOINT", "AZURE_CONTENT_UNDERSTANDING_KEY"]
        if args.provider == "azure-cu"
        else ["MISTRAL_API_KEY"]
    )
    if args.provider == "mistral" and args.mistral_provider == "azure":
        required.append("MISTRAL_API_ENDPOINT")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def _analyzer(args: argparse.Namespace) -> Callable[[Path], dict[str, Any]]:
    if args.provider == "azure-cu":
        return lambda path: analyze_azure_cu(path, args.analyzer_id)
    return lambda path: analyze_mistral(
        path,
        args.mistral_provider,
        args.mistral_model,
    )


def _parser(provider: str) -> Callable[[str], tuple[str | None, Any]]:
    if provider == "azure-cu":
        return parse_azure_content_understanding_response
    return parse_mistral_ocr_response


def _error_record(case: Case, status: str, error: Exception) -> dict[str, object]:
    return {
        "case_id": case.id,
        "score": None,
        "aggregate_contribution": 0.0,
        "status": status,
        "language": case.language,
        "source": case.pdf,
        "preview": case.image,
        "output": None,
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    cases = _read_cases(dataset_root)
    _validate_environment(args)
    configuration = _configuration(args)
    configuration_sha256 = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    outputs_dir = output_root / "outputs"
    raw_dir = output_root / "raw"
    status_dir = output_root / "status"
    evaluation_dir = output_root / "evaluation"
    for directory in (outputs_dir, raw_dir, status_dir, evaluation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _write_json(
        output_root / "manifest.json",
        {
            "schema_version": 1,
            "benchmark": "rd-tablebench",
            "status": "running",
            "configuration": configuration,
            "case_count": len(cases),
        },
    )
    analyze = _analyzer(args)
    parse = _parser(args.provider)

    def process(case: Case) -> dict[str, object]:
        input_path = dataset_root / case.pdf
        ground_truth_path = dataset_root / case.ground_truth
        output_path = outputs_dir / f"{case.id}.html"
        raw_path = raw_dir / f"{case.id}.json"
        status_path = status_dir / f"{case.id}.json"
        input_sha256 = _sha256_file(input_path)
        if status_path.is_file() and output_path.is_file() and raw_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                status.get("status") == "scored"
                and status.get("input_sha256") == input_sha256
                and status.get("configuration_sha256") == configuration_sha256
                and status.get("output_sha256") == _sha256_file(output_path)
                and status.get("raw_sha256") == _sha256_file(raw_path)
                and isinstance(status.get("result"), dict)
            ):
                return status["result"]

        output_path.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)
        started = time.monotonic()
        try:
            raw = analyze(input_path)
            _write_json(raw_path, raw)
        except Exception as error:  # Provider SDKs expose different error hierarchies.
            result = _error_record(case, "inference_error", error)
            _write_json(
                status_path,
                {
                    "schema_version": 1,
                    "case_id": case.id,
                    "status": "inference_error",
                    "input_sha256": input_sha256,
                    "configuration_sha256": configuration_sha256,
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "result": result,
                },
            )
            return result

        try:
            prediction_html, _ = parse(str(raw_path))
            if not prediction_html:
                raise ValueError("The provider response contains no HTML table.")
            _write_text(output_path, prediction_html)
            ground_truth = html_to_numpy(ground_truth_path.read_text(encoding="utf-8"))
            prediction = html_to_numpy(prediction_html)
            if ground_truth.size == 0 or prediction.size == 0:
                raise ValueError("HTML did not contain a table with cells.")
            score = float(table_similarity(ground_truth, prediction))
            result: dict[str, object] = {
                "case_id": case.id,
                "score": score,
                "aggregate_contribution": score,
                "status": "scored",
                "language": case.language,
                "source": case.pdf,
                "preview": case.image,
                "output": f"outputs/{case.id}.html",
            }
            _write_json(
                status_path,
                {
                    "schema_version": 1,
                    "case_id": case.id,
                    "status": "scored",
                    "input_sha256": input_sha256,
                    "configuration_sha256": configuration_sha256,
                    "output_sha256": _sha256_file(output_path),
                    "raw_sha256": _sha256_file(raw_path),
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "output": f"outputs/{case.id}.html",
                    "raw": f"raw/{case.id}.json",
                    "result": result,
                },
            )
            return result
        except Exception as error:  # Parsing and native evaluator failures are distinct.
            result = _error_record(case, "evaluation_error", error)
            result["output"] = (
                f"outputs/{case.id}.html" if output_path.is_file() else None
            )
            _write_json(
                status_path,
                {
                    "schema_version": 1,
                    "case_id": case.id,
                    "status": "evaluation_error",
                    "input_sha256": input_sha256,
                    "configuration_sha256": configuration_sha256,
                    "raw_sha256": _sha256_file(raw_path),
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "raw": f"raw/{case.id}.json",
                    "result": result,
                },
            )
            return result

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(process, case): case.id for case in cases}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda result: str(result["case_id"]))
    failures = [result for result in results if result["status"] != "scored"]
    score = sum(float(result["aggregate_contribution"]) for result in results) / len(results)
    inference_errors = sum(result["status"] == "inference_error" for result in results)
    evaluation_errors = sum(result["status"] == "evaluation_error" for result in results)
    summary = {
        "schema_version": 1,
        "benchmark": "rd-tablebench",
        "native_metric": {
            "name": "table_similarity",
            "value": score,
            "direction": "higher_is_better",
            "range": [0.0, 1.0],
        },
        "score": score * 100.0,
        "case_count": len(results),
        "scored_count": len(results) - len(failures),
        "inference_error_count": inference_errors,
        "evaluation_error_count": evaluation_errors,
        "failure_count": len(failures),
        "skipped_count": 0,
        "denominator_policy": (
            "All selected cases; inference and evaluation errors contribute zero."
        ),
    }
    _write_jsonl(evaluation_dir / "results.jsonl", results)
    _write_jsonl(evaluation_dir / "failures.jsonl", failures)
    _write_json(evaluation_dir / "summary.json", summary)
    _write_json(
        output_root / "manifest.json",
        {
            "schema_version": 1,
            "benchmark": "rd-tablebench",
            "status": "completed",
            "configuration": configuration,
            "case_count": len(cases),
            "score": summary["score"],
            "artifacts": {
                "outputs": "outputs",
                "raw": "raw",
                "status": "status",
                "results": "evaluation/results.jsonl",
                "failures": "evaluation/failures.jsonl",
                "summary": "evaluation/summary.json",
            },
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run providers and native evaluation.")
    run_parser.add_argument("--dataset-root", required=True)
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--provider", choices=("azure-cu", "mistral"), required=True)
    run_parser.add_argument("--parallel", type=int, default=1)
    run_parser.add_argument("--analyzer-id", default="prebuilt-layout")
    run_parser.add_argument("--mistral-provider", choices=("azure", "mistral"), default="azure")
    run_parser.add_argument("--mistral-model", default="mistral-ocr-4-0")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.parallel <= 0:
        raise ValueError("--parallel must be positive.")
    if args.command == "run":
        summary = run(args)
        print(json.dumps(summary, sort_keys=True))
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
