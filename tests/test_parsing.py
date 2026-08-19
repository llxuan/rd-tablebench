from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

UPSTREAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UPSTREAM_ROOT))

from convert import html_to_numpy  # noqa: E402
from parsing import parse_azure_content_understanding_response  # noqa: E402


class AzureContentUnderstandingParsingTests(unittest.TestCase):
    @staticmethod
    def _write_response(root: Path, raw: object) -> Path:
        path = root / "response.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path

    def test_selects_logically_largest_table_and_preserves_spans(self):
        raw = {
            "contents": [
                {
                    "tables": [
                        {
                            "rowCount": 2,
                            "columnCount": 2,
                            "cells": [
                                {"rowIndex": 0, "columnIndex": 0, "content": "SMALL"}
                            ],
                        },
                        {
                            "rowCount": 3,
                            "columnCount": 2,
                            "cells": [
                                {
                                    "rowIndex": 0,
                                    "columnIndex": 0,
                                    "columnSpan": 2,
                                    "rowSpan": 2,
                                    "kind": "columnHeader",
                                    "content": "LARGE & RAW",
                                },
                                {"rowIndex": 2, "columnIndex": 0, "content": "A"},
                                {"rowIndex": 2, "columnIndex": 1, "content": "B"},
                            ],
                        },
                    ]
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_response(Path(directory), raw)
            table, parsed = parse_azure_content_understanding_response(str(path))

        assert table is not None
        self.assertEqual(parsed, raw)
        self.assertIn(
            '<th rowspan="2" colspan="2">LARGE &amp; RAW</th>',
            table,
        )
        self.assertNotIn("SMALL", table)
        self.assertEqual(html_to_numpy(table).shape, (3, 2))

    def test_returns_none_when_no_table_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_response(Path(directory), {"contents": []})
            table, _ = parse_azure_content_understanding_response(str(path))

        self.assertIsNone(table)


if __name__ == "__main__":
    unittest.main()
