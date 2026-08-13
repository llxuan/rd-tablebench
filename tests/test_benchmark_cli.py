from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

UPSTREAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UPSTREAM_ROOT))

import benchmark_cli  # noqa: E402
from parsing import (  # noqa: E402
    parse_azure_content_understanding_response,
    parse_mistral_ocr_response,
)


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


class BenchmarkCliTests(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        for relative, content in {
            "pdfs/case.pdf": b"pdf",
            "_images/case.jpg": b"jpg",
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

        self.assertEqual(first["score"], 100.0)
        self.assertEqual(second["score"], 100.0)
        self.assertEqual(results[0]["status"], "scored")

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


if __name__ == "__main__":
    unittest.main()
