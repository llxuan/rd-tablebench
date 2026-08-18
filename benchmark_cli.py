"""Stable provider + evaluation CLI for RD-TableBench."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from error_analysis import (
    ERROR_ANALYSIS_VERSION,
    build_error_analysis,
    summarize_error_analysis,
)
from metric_tracks import (
    EVALUATION_POLICY,
    METRIC_KEYS,
    OUTPUT_KEYS,
    PRIMARY_METRIC_KEY,
    build_metric_outputs,
    score_metric_outputs,
)
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
    ground_truth: str
    language: str


def _evaluation_revision() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "benchmark_cli.py",
        "convert.py",
        "error_analysis.py",
        "formula_text.py",
        "grading.py",
        "html_normalization.py",
        "metric_tracks.py",
        "teds_struct.py",
        "text_normalization.py",
    ):
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


EVALUATION_REVISION = _evaluation_revision()


def _inference_revision() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative in (
        "benchmark_cli.py",
        "providers/azure_content_understanding.py",
        "providers/mistral_ocr.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


INFERENCE_REVISION = _inference_revision()
LEGACY_INFERENCE_REVISIONS = {
    # 9fa08fd: provider/input code is unchanged; this revision predates only
    # post-evaluation error diagnostics, so its cached raw responses are safe.
    # Both CRLF and LF checkout hashes are retained.
    "10cab94cb22d2f3db96aa8887f08d790e1f1713c561f0d1ff631791a1df72ffb",
    "f972c4a3daa2a152bce9db4773c633e6a29e473132890cc5dac6dbd51dd7b671",
    # 119d5a5: the released-PDF provider request is unchanged by the subsequent
    # rendered-track cleanup, so its cached raw responses remain safe.
    "1e39c577da78160e6a8c9731faf7900066cd91beddcdc551d88a5623a13670f3",
    "92035547ad3a4474e7919d82abe28be05beb1504151dd17bcc6e1be795a4583f",
    # 388a4cd: the released-PDF request path is unchanged by removing the JPG
    # input contract, so its cached raw responses remain safe.
    "16261bb1a86529144c798d6de37f6cd7e12e6bf8365ad0785611e2f44f52cca8",
    "fd0ba045a8d2af40766faaf52d3ecde892239f9f6d1468aded9e308ab654fb6f",
}


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
                ground_truth=f"groundtruth/{case_id}.html",
                language=row.get("language", "").strip() or "unknown",
            )
            missing = [
                relative
                for relative in (case.pdf, case.ground_truth)
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
        configuration: dict[str, object] = {
            "provider": args.provider,
            "parallel": args.parallel,
            "analyzer_id": args.analyzer_id,
            "evaluation_policy": args.evaluation_policy,
        }
    else:
        configuration = {
            "provider": args.provider,
            "parallel": args.parallel,
            "mistral_provider": args.mistral_provider,
            "model": args.mistral_model,
            "table_format": args.mistral_table_format,
            "evaluation_policy": args.evaluation_policy,
        }
    return configuration


def _provider_configuration(args: argparse.Namespace) -> dict[str, object]:
    configuration = _configuration(args).copy()
    configuration.pop("parallel", None)
    configuration.pop("evaluation_policy", None)
    return configuration


def _configuration_sha256(configuration: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _compatible_provider_configuration_sha256s(
    args: argparse.Namespace,
) -> set[str]:
    current = _provider_configuration(args)
    legacy_pdf = {**current, "input_mode": "pdf"}
    return {
        _configuration_sha256(current),
        _configuration_sha256(legacy_pdf),
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


def _provider_input_configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "source_input_mode": "pdf",
        "media_type": "application/pdf",
    }


def _analyzer(args: argparse.Namespace) -> Callable[[Path], dict[str, Any]]:
    if args.provider == "azure-cu":
        return lambda path: analyze_azure_cu(
            path,
            args.analyzer_id,
        )
    return lambda path: analyze_mistral(
        path,
        args.mistral_provider,
        args.mistral_model,
        args.mistral_table_format,
    )


def _parser(provider: str) -> Callable[[str], tuple[str | None, Any]]:
    if provider == "azure-cu":
        return parse_azure_content_understanding_response
    return parse_mistral_ocr_response


def _error_record(
    case: Case,
    status: str,
    error: Exception,
    source: str,
    error_analysis: dict[str, object],
) -> dict[str, object]:
    return {
        "case_id": case.id,
        "score": None,
        "aggregate_contribution": 0.0,
        "metrics": {key: None for key in METRIC_KEYS},
        "aggregate_contributions": {key: 0.0 for key in METRIC_KEYS},
        "status": status,
        "language": case.language,
        "source": source,
        "output": None,
        "outputs": {},
        "error": {"type": type(error).__name__, "message": str(error)},
        "error_analysis": error_analysis,
    }


def _case_output_paths(output_root: Path, case_id: str) -> dict[str, Path]:
    return {
        "raw_largest": output_root / "outputs" / f"{case_id}.html",
        **{
            key: output_root / "derived" / key / f"{case_id}.html"
            for key in OUTPUT_KEYS
            if key != "raw_largest"
        },
    }


def _relative_outputs(case_id: str) -> dict[str, str]:
    return {
        "raw_largest": f"outputs/{case_id}.html",
        **{
            key: f"derived/{key}/{case_id}.html"
            for key in OUTPUT_KEYS
            if key != "raw_largest"
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    cases = _read_cases(dataset_root)
    _validate_environment(args)
    configuration = _configuration(args)
    configuration_sha256 = _configuration_sha256(configuration)
    compatible_provider_configuration_sha256s = (
        _compatible_provider_configuration_sha256s(args)
    )
    provider_configuration_sha256 = _configuration_sha256(
        _provider_configuration(args)
    )
    raw_dir = output_root / "raw"
    status_dir = output_root / "status"
    evaluation_dir = output_root / "evaluation"
    derived_dir = output_root / "derived"
    outputs_dir = output_root / "outputs"
    for directory in (outputs_dir, derived_dir, raw_dir, status_dir, evaluation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _write_json(
        output_root / "manifest.json",
        {
            "schema_version": 2,
            "benchmark": "rd-tablebench",
            "status": "running",
            "configuration": configuration,
            "provider_input": _provider_input_configuration(args),
            "inference_revision": INFERENCE_REVISION,
            "evaluation_revision": EVALUATION_REVISION,
            "error_analysis_version": ERROR_ANALYSIS_VERSION,
            "case_count": len(cases),
        },
    )
    analyze = _analyzer(args)
    parse = _parser(args.provider)
    score_workers = min(args.parallel, os.cpu_count() or 1)
    score_executor = (
        ProcessPoolExecutor(max_workers=score_workers)
        if len(cases) > 1 and score_workers > 1
        else None
    )

    def process(case: Case) -> dict[str, object]:
        source = case.pdf
        source_path = dataset_root / source
        ground_truth_path = dataset_root / case.ground_truth
        ground_truth_html = ground_truth_path.read_text(encoding="utf-8")
        output_paths = _case_output_paths(output_root, case.id)
        relative_outputs = _relative_outputs(case.id)
        raw_path = raw_dir / f"{case.id}.json"
        status_path = status_dir / f"{case.id}.json"
        source_sha256 = _sha256_file(source_path)
        ground_truth_sha256 = _sha256_file(ground_truth_path)
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file()
            else {}
        )
        input_sha256 = source_sha256
        provider_input = {
            "sha256": input_sha256,
            "media_type": "application/pdf",
        }
        if raw_path.is_file() and all(
            path.is_file() for path in output_paths.values()
        ):
            output_sha256s = {
                key: _sha256_file(path) for key, path in output_paths.items()
            }
            if (
                status.get("status") == "scored"
                and status.get("source_sha256") == source_sha256
                and status.get("input_sha256") == input_sha256
                and status.get("configuration_sha256") == configuration_sha256
                and status.get("ground_truth_sha256") == ground_truth_sha256
                and status.get("inference_revision") == INFERENCE_REVISION
                and status.get("evaluation_revision") == EVALUATION_REVISION
                and status.get("output_sha256s") == output_sha256s
                and status.get("raw_sha256") == _sha256_file(raw_path)
                and isinstance(status.get("result"), dict)
            ):
                return status["result"]

        for output_path in output_paths.values():
            output_path.unlink(missing_ok=True)
        inference_compatible = (
            raw_path.is_file()
            and status.get("source_sha256") == source_sha256
            and status.get("input_sha256") == input_sha256
            and status.get("provider_configuration_sha256")
            in compatible_provider_configuration_sha256s
            and status.get("inference_revision")
            in {INFERENCE_REVISION, *LEGACY_INFERENCE_REVISIONS}
            and status.get("raw_sha256") == _sha256_file(raw_path)
        )
        started = time.monotonic()
        if not inference_compatible:
            raw_path.unlink(missing_ok=True)
            try:
                raw = analyze(source_path)
                _write_json(raw_path, raw)
            except Exception as error:  # noqa: BLE001 - provider SDKs use unrelated errors.
                error_analysis = build_error_analysis(
                    provider=args.provider,
                    data=None,
                    fallback_html=None,
                    ground_truth_html=ground_truth_html,
                    outputs=None,
                    score=None,
                    status="inference_error",
                )
                result = _error_record(
                    case,
                    "inference_error",
                    error,
                    source,
                    error_analysis,
                )
                _write_json(
                    status_path,
                    {
                        "schema_version": 2,
                        "case_id": case.id,
                        "status": "inference_error",
                        "source_sha256": source_sha256,
                        "input_sha256": input_sha256,
                        "provider_configuration_sha256": provider_configuration_sha256,
                        "configuration_sha256": configuration_sha256,
                        "ground_truth_sha256": ground_truth_sha256,
                        "provider_input": provider_input,
                        "inference_revision": INFERENCE_REVISION,
                        "evaluation_revision": EVALUATION_REVISION,
                        "duration_seconds": round(time.monotonic() - started, 3),
                        "result": result,
                    },
                )
                return result

        fallback_html: str | None = None
        parsed_data: dict[str, Any] | None = None
        outputs: dict[str, str] | None = None
        try:
            fallback_html, parsed_data = parse(str(raw_path))
            if not isinstance(parsed_data, dict):
                raise TypeError("The provider response is not a JSON object.")
            outputs = build_metric_outputs(args.provider, parsed_data, fallback_html)
            for key, value in outputs.items():
                _write_text(output_paths[key], value)
            scores = (
                score_executor.submit(
                    score_metric_outputs,
                    ground_truth_html,
                    outputs,
                ).result()
                if score_executor is not None
                else score_metric_outputs(ground_truth_html, outputs)
            )
            score = scores[PRIMARY_METRIC_KEY]
            error_analysis = build_error_analysis(
                provider=args.provider,
                data=parsed_data,
                fallback_html=fallback_html,
                ground_truth_html=ground_truth_html,
                outputs=outputs,
                score=score,
                status="scored",
            )
            result: dict[str, object] = {
                "case_id": case.id,
                "score": score,
                "aggregate_contribution": score,
                "metrics": scores,
                "aggregate_contributions": scores,
                "status": "scored",
                "language": case.language,
                "source": source,
                "output": relative_outputs["raw_largest"],
                "outputs": relative_outputs,
                "error_analysis": error_analysis,
            }
            _write_json(
                status_path,
                {
                    "schema_version": 2,
                    "case_id": case.id,
                    "status": "scored",
                    "source_sha256": source_sha256,
                    "input_sha256": input_sha256,
                    "provider_configuration_sha256": provider_configuration_sha256,
                    "configuration_sha256": configuration_sha256,
                    "ground_truth_sha256": ground_truth_sha256,
                    "provider_input": provider_input,
                    "inference_revision": INFERENCE_REVISION,
                    "evaluation_revision": EVALUATION_REVISION,
                    "output_sha256s": {
                        key: _sha256_file(path)
                        for key, path in output_paths.items()
                    },
                    "raw_sha256": _sha256_file(raw_path),
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "output": relative_outputs["raw_largest"],
                    "outputs": relative_outputs,
                    "raw": f"raw/{case.id}.json",
                    "result": result,
                },
            )
            return result
        except Exception as error:  # noqa: BLE001 - records evaluator boundary failures.
            error_analysis = build_error_analysis(
                provider=args.provider,
                data=parsed_data,
                fallback_html=fallback_html,
                ground_truth_html=ground_truth_html,
                outputs=outputs,
                score=None,
                status="evaluation_error",
            )
            result = _error_record(
                case,
                "evaluation_error",
                error,
                source,
                error_analysis,
            )
            available_outputs = {
                key: relative_outputs[key]
                for key, path in output_paths.items()
                if path.is_file()
            }
            result["output"] = available_outputs.get("raw_largest")
            result["outputs"] = available_outputs
            _write_json(
                status_path,
                {
                    "schema_version": 2,
                    "case_id": case.id,
                    "status": "evaluation_error",
                    "source_sha256": source_sha256,
                    "input_sha256": input_sha256,
                    "provider_configuration_sha256": provider_configuration_sha256,
                    "configuration_sha256": configuration_sha256,
                    "ground_truth_sha256": ground_truth_sha256,
                    "provider_input": provider_input,
                    "inference_revision": INFERENCE_REVISION,
                    "evaluation_revision": EVALUATION_REVISION,
                    "raw_sha256": _sha256_file(raw_path),
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "raw": f"raw/{case.id}.json",
                    "outputs": available_outputs,
                    "result": result,
                },
            )
            return result

    results: list[dict[str, object]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(process, case): case.id for case in cases}
            for future in as_completed(futures):
                results.append(future.result())
    finally:
        if score_executor is not None:
            score_executor.shutdown()
    results.sort(key=lambda result: str(result["case_id"]))
    failures = [result for result in results if result["status"] != "scored"]
    aggregate_scores = {
        key: sum(
            float(result["aggregate_contributions"][key])  # type: ignore[index]
            for result in results
        )
        / len(results)
        for key in METRIC_KEYS
    }
    score = aggregate_scores[PRIMARY_METRIC_KEY]
    inference_errors = sum(result["status"] == "inference_error" for result in results)
    evaluation_errors = sum(result["status"] == "evaluation_error" for result in results)
    error_analysis_summary = summarize_error_analysis(results)
    summary = {
        "schema_version": 2,
        "benchmark": "rd-tablebench",
        "evaluation_policy": EVALUATION_POLICY,
        "native_metric": {
            "name": "table_similarity",
            "key": PRIMARY_METRIC_KEY,
            "value": score,
            "direction": "higher_is_better",
            "range": [0.0, 1.0],
        },
        "score": score * 100.0,
        "metrics": {
            key: {
                "value": value,
                "score": value * 100.0,
                "direction": "higher_is_better",
                "range": [0.0, 1.0],
                "scored_count": len(results) - len(failures),
                "inference_error_count": inference_errors,
                "evaluation_error_count": evaluation_errors,
                "skipped_count": 0,
            }
            for key, value in aggregate_scores.items()
        },
        "case_count": len(results),
        "scored_count": len(results) - len(failures),
        "inference_error_count": inference_errors,
        "evaluation_error_count": evaluation_errors,
        "failure_count": len(failures),
        "skipped_count": 0,
        "error_analysis": error_analysis_summary,
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
            "schema_version": 2,
            "benchmark": "rd-tablebench",
            "status": "completed",
            "configuration": configuration,
            "provider_input": _provider_input_configuration(args),
            "inference_revision": INFERENCE_REVISION,
            "evaluation_revision": EVALUATION_REVISION,
            "error_analysis_version": ERROR_ANALYSIS_VERSION,
            "case_count": len(cases),
            "score": summary["score"],
            "artifacts": {
                "outputs": "outputs",
                "derived": "derived",
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
    run_parser.add_argument(
        "--evaluation-policy",
        choices=(EVALUATION_POLICY,),
        default=EVALUATION_POLICY,
    )
    run_parser.add_argument("--analyzer-id", default="prebuilt-layout")
    run_parser.add_argument("--mistral-provider", choices=("azure", "mistral"), default="azure")
    run_parser.add_argument("--mistral-model", default="mistral-ocr-4-0")
    run_parser.add_argument(
        "--mistral-table-format",
        choices=("inline", "markdown", "html"),
        default="inline",
    )
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
