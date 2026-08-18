from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock

UPSTREAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UPSTREAM_ROOT))

import benchmark_cli  # noqa: E402
from parsing import (  # noqa: E402
    parse_azure_content_understanding_response,
    parse_mistral_ocr_response,
)
from providers.mistral_ocr import input_document, process  # noqa: E402


class ParsingTests(unittest.TestCase):
    def _write_json(self, root: Path, value: object) -> Path:
        path = root / "response.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_parses_azure_content_understanding_table_with_spans(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_json(
                Path(directory),
                {
                    "contents": [
                        {
                            "tables": [
                                {
                                    "rowCount": 2,
                                    "columnCount": 2,
                                    "cells": [
                                        {
                                            "rowIndex": 0,
                                            "columnIndex": 0,
                                            "columnSpan": 2,
                                            "kind": "columnHeader",
                                            "content": "A & B",
                                        },
                                        {"rowIndex": 1, "columnIndex": 0, "content": "1"},
                                        {"rowIndex": 1, "columnIndex": 1, "content": "2"},
                                    ],
                                }
                            ]
                        }
                    ]
                },
            )
            table, _ = parse_azure_content_understanding_response(str(path))

        self.assertEqual(
            table,
            '<table><tr><th colspan="2">A &amp; B</th></tr>'
            "<tr><td>1</td><td>2</td></tr></table>",
        )

    def test_parses_mistral_markdown_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_json(
                Path(directory),
                {"pages": [{"markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |"}]},
            )
            table, _ = parse_mistral_ocr_response(str(path))

        self.assertEqual(
            table,
            "<table><tr><th>A</th><th>B</th></tr>"
            "<tr><td>1</td><td>2</td></tr></table>",
        )

    def test_parses_all_ordered_mistral_html_table_fragments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_json(
                Path(directory),
                {
                    "pages": [
                        {
                            "markdown": "[tbl-0.html](tbl-0.html)",
                            "tables": [
                                {
                                    "format": "html",
                                    "content": "<table><tr><td>A</td></tr></table>",
                                },
                                {
                                    "format": "html",
                                    "content": "<table><tr><td>B</td></tr></table>",
                                },
                            ],
                        }
                    ]
                },
            )
            table, _ = parse_mistral_ocr_response(str(path))

        self.assertEqual(
            table,
            "<table><tr><td>A</td></tr></table>\n"
            "<table><tr><td>B</td></tr></table>",
        )

    def test_mistral_uses_pdf_document_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "case.pdf"
            pdf.write_bytes(b"pdf")

            pdf_document = input_document(pdf)

        self.assertTrue(pdf_document["document_url"].startswith("data:application/pdf;base64,"))
        self.assertEqual(pdf_document["type"], "document_url")

    def test_mistral_retries_using_azure_retry_after_ms(self):
        class RateLimitError(RuntimeError):
            status_code = 429
            headers = {"retry-after-ms": "5000"}

        response = object()
        client = Mock()
        client.ocr.process.side_effect = [RateLimitError(), response]
        with patch("providers.mistral_ocr.time.sleep") as sleep:
            actual = process(
                client,
                "mistral-ocr-4-0",
                {"type": "document_url", "document_url": "data:application/pdf;base64,"},
                "html",
            )

        self.assertIs(actual, response)
        self.assertEqual(client.ocr.process.call_args.kwargs["table_format"], "html")
        sleep.assert_called_once_with(5.0)


class BenchmarkCliTests(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        for relative, content in {
            "pdfs/case.pdf": b"pdf",
            "groundtruth/case.html": (
                b"<table><tr><th>A</th><th>B</th></tr>"
                b"<tr><td>1</td><td>2</td></tr></table>"
            ),
        }.items():
            path = dataset / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        scores = dataset / "providers" / "scores.csv"
        scores.parent.mkdir(parents=True)
        with scores.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pdf_path", "language"])
            writer.writeheader()
            writer.writerow({"pdf_path": "case.pdf", "language": "en"})
        return dataset

    def _args(self, dataset: Path, output: Path) -> argparse.Namespace:
        return argparse.Namespace(
            command="run",
            dataset_root=str(dataset),
            output_root=str(output),
            provider="mistral",
            parallel=1,
            analyzer_id="prebuilt-layout",
            mistral_provider="azure",
            mistral_model="mistral-ocr-4-0",
            mistral_table_format="inline",
            evaluation_policy=benchmark_cli.EVALUATION_POLICY,
        )

    def test_run_scores_and_resumes_matching_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(self._dataset(root), root / "output")
            raw = {"pages": [{"markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |"}]}
            with (
                patch.object(benchmark_cli, "_validate_environment"),
                patch.object(benchmark_cli, "_analyzer", return_value=lambda _: raw),
            ):
                first = benchmark_cli.run(args)
            with (
                patch.object(benchmark_cli, "_validate_environment"),
                patch.object(
                    benchmark_cli,
                    "_analyzer",
                    return_value=lambda _: self.fail("matching case was not resumed"),
                ),
            ):
                second = benchmark_cli.run(args)

            results = [
                json.loads(line)
                for line in (root / "output/evaluation/results.jsonl").read_text().splitlines()
            ]
            manifest = json.loads(
                (root / "output/manifest.json").read_text(encoding="utf-8")
            )
            derived_outputs_exist = all(
                (root / "output" / relative).is_file()
                for relative in results[0]["outputs"].values()
            )

        self.assertEqual(first["score"], 100.0)
        self.assertEqual(second["score"], 100.0)
        self.assertEqual(results[0]["status"], "scored")
        self.assertEqual(set(results[0]["metrics"]), set(benchmark_cli.METRIC_KEYS))
        self.assertTrue(all(score == 1.0 for score in results[0]["metrics"].values()))
        self.assertEqual(set(first["metrics"]), set(benchmark_cli.METRIC_KEYS))
        self.assertEqual(
            manifest["configuration"]["evaluation_policy"],
            benchmark_cli.EVALUATION_POLICY,
        )
        self.assertTrue(derived_outputs_exist)

    def test_multiple_cases_use_process_safe_metric_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self._dataset(root)
            (dataset / "pdfs/case-2.pdf").write_bytes(b"pdf-2")
            (dataset / "groundtruth/case-2.html").write_text(
                "<table><tr><th>A</th><th>B</th></tr>"
                "<tr><td>1</td><td>2</td></tr></table>",
                encoding="utf-8",
            )
            with (dataset / "providers/scores.csv").open(
                "a", encoding="utf-8", newline=""
            ) as handle:
                csv.writer(handle).writerow(["case-2.pdf", "en"])
            args = self._args(dataset, root / "output")
            args.parallel = 2
            raw = {"pages": [{"markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |"}]}
            with (
                patch.object(benchmark_cli, "_validate_environment"),
                patch.object(benchmark_cli, "_analyzer", return_value=lambda _: raw),
            ):
                summary = benchmark_cli.run(args)

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["score"], 100.0)

    def test_ground_truth_change_reuses_raw_and_rescores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self._dataset(root)
            args = self._args(dataset, root / "output")
            raw = {"pages": [{"markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |"}]}
            with (
                patch.object(benchmark_cli, "_validate_environment"),
                patch.object(benchmark_cli, "_analyzer", return_value=lambda _: raw),
            ):
                first = benchmark_cli.run(args)
            (dataset / "groundtruth/case.html").write_text(
                "<table><tr><th>A</th><th>B</th></tr>"
                "<tr><td>9</td><td>9</td></tr></table>",
                encoding="utf-8",
            )
            with (
                patch.object(benchmark_cli, "_validate_environment"),
                patch.object(
                    benchmark_cli,
                    "_analyzer",
                    return_value=lambda _: self.fail("compatible raw response was not reused"),
                ),
            ):
                second = benchmark_cli.run(args)

        self.assertEqual(first["score"], 100.0)
        self.assertLess(second["score"], 100.0)

    def test_evaluator_revision_change_reuses_raw_and_rescores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(self._dataset(root), root / "output")
            raw = {"pages": [{"markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |"}]}
            with (
                patch.object(benchmark_cli, "_validate_environment"),
                patch.object(benchmark_cli, "_analyzer", return_value=lambda _: raw),
            ):
                benchmark_cli.run(args)
            with (
                patch.object(benchmark_cli, "_validate_environment"),
                patch.object(benchmark_cli, "EVALUATION_REVISION", "changed"),
                patch.object(
                    benchmark_cli,
                    "_analyzer",
                    return_value=lambda _: self.fail("compatible raw response was not reused"),
                ),
            ):
                benchmark_cli.run(args)
            status = json.loads(
                (root / "output/status/case.json").read_text(encoding="utf-8")
            )

        self.assertEqual(status["evaluation_revision"], "changed")

    def test_legacy_inference_revision_reuses_raw_for_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(self._dataset(root), root / "output")
            raw = {"pages": [{"markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |"}]}
            with (
                patch.object(benchmark_cli, "_validate_environment"),
                patch.object(benchmark_cli, "_analyzer", return_value=lambda _: raw),
            ):
                benchmark_cli.run(args)
            status_path = root / "output/status/case.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["inference_revision"] = next(
                iter(benchmark_cli.LEGACY_INFERENCE_REVISIONS)
            )
            benchmark_cli._write_json(status_path, status)
            with (
                patch.object(benchmark_cli, "_validate_environment"),
                patch.object(
                    benchmark_cli,
                    "_analyzer",
                    return_value=lambda _: self.fail(
                        "legacy compatible raw response was not reused"
                    ),
                ),
            ):
                benchmark_cli.run(args)
            updated = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(
            updated["inference_revision"],
            benchmark_cli.INFERENCE_REVISION,
        )
        self.assertEqual(
            updated["result"]["error_analysis"]["version"],
            benchmark_cli.ERROR_ANALYSIS_VERSION,
        )

    def test_provider_failure_is_not_a_scored_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(self._dataset(root), root / "output")

            def fail(_: Path) -> dict[str, object]:
                raise RuntimeError("provider failed")

            with (
                patch.object(benchmark_cli, "_validate_environment"),
                patch.object(benchmark_cli, "_analyzer", return_value=fail),
            ):
                summary = benchmark_cli.run(args)
            result = json.loads(
                (root / "output/evaluation/results.jsonl").read_text().splitlines()[0]
            )

        self.assertEqual(summary["score"], 0.0)
        self.assertEqual(result["status"], "inference_error")
        self.assertIsNone(result["score"])
        self.assertEqual(set(result["metrics"]), set(benchmark_cli.METRIC_KEYS))
        self.assertTrue(all(score is None for score in result["metrics"].values()))


if __name__ == "__main__":
    unittest.main()
