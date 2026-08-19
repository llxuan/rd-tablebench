from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

UPSTREAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UPSTREAM_ROOT))

from convert import html_to_numpy  # noqa: E402
from parsing import (  # noqa: E402
    parse_azure_content_understanding_response,
    parse_mistral_ocr_response,
)


class ProviderParsingTests(unittest.TestCase):
    @staticmethod
    def _write_response(root: Path, raw: object) -> Path:
        path = root / "response.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path

    def test_azure_selects_logically_largest_table_and_preserves_spans(self):
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

    def test_mistral_selects_logically_largest_raw_html(self):
        large = (
            '<table><thead><tr><th colspan="2">LARGE</th></tr></thead>'
            "<tr><td>1</td><td>2</td></tr></table>"
        )
        raw = {
            "pages": [
                {
                    "tables": [
                        {
                            "format": "html",
                            "content": "<table><tr><td>LONG SMALL TEXT</td></tr></table>",
                        },
                        {"format": "html", "content": large},
                    ]
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_response(Path(directory), raw)
            table, parsed = parse_mistral_ocr_response(str(path))

        self.assertEqual(parsed, raw)
        self.assertEqual(table, large)
        self.assertEqual(html_to_numpy(table).shape, (2, 2))

    def test_provider_parsers_return_none_when_no_table_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            azure = self._write_response(root, {"contents": []})
            self.assertIsNone(parse_azure_content_understanding_response(str(azure))[0])
            mistral = self._write_response(root, {"pages": []})
            self.assertIsNone(parse_mistral_ocr_response(str(mistral))[0])


if __name__ == "__main__":
    unittest.main()
