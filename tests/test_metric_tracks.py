from __future__ import annotations

import sys
import unittest
from pathlib import Path

UPSTREAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UPSTREAM_ROOT))

from formula_text import latex_to_visible_text  # noqa: E402
from html_normalization import html_metric_outputs  # noqa: E402
from metric_tracks import (  # noqa: E402
    METRIC_KEYS,
    build_metric_outputs,
    score_metric_outputs,
)
from text_normalization import (  # noqa: E402
    normalize_display_text,
    normalize_scoring_text,
)


class FormulaTextTests(unittest.TestCase):
    def test_renders_observed_formula_patterns(self):
        cases = {
            r"$$\mathrm{mean} \pm \mathrm{SD}\mathrm{or}\mathrm{n}\left(\%\right)$$": "mean ± SD or n (%)",
            r"$$-\frac{\partial P}{\partial x}$$": "-∂P/∂x",
            r"$$\left[\frac{\pi}{180}, \frac{\pi}{4}\right]$$": "[π/180, π/4]",
            r"$$\mathbb{H}$$": "H",
            r"$$D_{1}$$": "D1",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(latex_to_visible_text(source), expected)

    def test_unknown_command_is_not_silently_deleted(self):
        self.assertEqual(latex_to_visible_text(r"$$\mystery{abc}$$"), "mysteryabc")


class TextNormalizationTests(unittest.TestCase):
    def test_display_normalization_is_narrow(self):
        self.assertEqual(
            normalize_display_text("✗ ✘ ✓ ☑ ☒ ☐"),
            "X X ✓ ☑ ☒ ☐",
        )

    def test_checkbox_scoring_is_symmetric_and_state_preserving(self):
        checked = {normalize_scoring_text(value) for value in ("✓", "✔", "☑", "[x]")}
        optional = {normalize_scoring_text(value) for value in ("☒", "☐", "□", "[ ]")}
        self.assertEqual(len(checked), 1)
        self.assertEqual(optional, {""})
        self.assertNotEqual(checked, optional)

    def test_multiplication_and_cross_variants_are_comparable(self):
        self.assertEqual(
            normalize_scoring_text("27 × 142 mm"),
            normalize_scoring_text("27 x 142 mm"),
        )


class HtmlMetricOutputTests(unittest.TestCase):
    def test_ocr_adapter_preserves_plain_underscore_text(self):
        outputs = html_metric_outputs(
            "<table><tr><td>account_name</td></tr></table>"
        )
        self.assertIn("account_name", outputs["ocr_largest"])

    def test_ocr_adapter_renders_only_explicit_math_in_mixed_text(self):
        outputs = html_metric_outputs(
            "<table><tr>"
            "<td>account_name $D_1$ and $x$</td>"
            "<td>$100</td>"
            "<td>malformed $x</td>"
            "<td>escaped\\_name</td>"
            "</tr></table>"
        )
        adapted = outputs["ocr_largest"]
        self.assertIn("account_name D1 and x", adapted)
        self.assertIn(">$100<", adapted)
        self.assertIn(">malformed $x<", adapted)
        self.assertIn(">escaped_name<", adapted)

    def test_builds_largest_normalized_and_merged_outputs(self):
        source = (
            '<table><thead><tr><th colspan="2"><b>A</b></th></tr></thead>'
            '<tbody><tr><td>D</td><td>E</td></tr></tbody></table>'
            '<table><tr><td>B</td><td>C</td></tr></table>'
        )
        outputs = html_metric_outputs(source, merge_fragments=True)
        self.assertIn("<thead>", outputs["raw_largest"])
        self.assertNotIn("<thead>", outputs["html_largest"])
        self.assertIn('<td colspan="2">A</td>', outputs["html_largest"])
        self.assertEqual(outputs["html_merged"].count("<tr>"), 3)

    def test_incompatible_fragments_fall_back_to_largest(self):
        source = (
            "<table><tr><td>A</td></tr></table>"
            "<table><tr><td>B</td><td>C</td></tr>"
            "<tr><td>D</td><td>E</td></tr></table>"
        )
        outputs = html_metric_outputs(source)
        self.assertIn("B", outputs["ocr_merged"])
        self.assertNotIn("A", outputs["ocr_merged"])

    def test_equal_shape_tables_do_not_merge_without_fragment_provenance(self):
        source = (
            "<table><tr><td>A</td></tr></table>"
            "<table><tr><td>B</td></tr></table>"
        )
        outputs = html_metric_outputs(source)
        self.assertEqual(outputs["html_merged"].count("<tr>"), 1)


class AzureMetricOutputTests(unittest.TestCase):
    @staticmethod
    def _table(
        top: float,
        bottom: float,
        value: str,
        span: dict[str, int] | None = None,
        page: int = 1,
        left: float = 0,
        right: float = 1,
    ) -> dict:
        cell = {
            "rowIndex": 0,
            "columnIndex": 0,
            "content": value,
            "source": (
                f"D({page},{left},{top},{right},{top},"
                f"{right},{bottom},{left},{bottom})"
            ),
        }
        if span is not None:
            cell["span"] = span
        return {
            "rowCount": 1,
            "columnCount": 1,
            "source": (
                f"D({page},{left},{top},{right},{top},"
                f"{right},{bottom},{left},{bottom})"
            ),
            "cells": [cell],
        }

    def test_recovers_formula_from_cu_markdown_span(self):
        markdown = "prefix $$-\\frac{\\partial P}{\\partial x}$$ suffix"
        formula = markdown[markdown.index("$$") : markdown.rindex("$$") + 2]
        span = {"offset": markdown.index(formula), "length": len(formula)}
        data = {
            "contents": [
                {
                    "markdown": markdown,
                    "tables": [self._table(0, 1, ":formula:", span)],
                }
            ]
        }
        outputs = build_metric_outputs("azure-cu", data, None)
        self.assertIn(":formula:", outputs["raw_largest"])
        self.assertIn("-∂P/∂x", outputs["ocr_largest"])

    def test_formula_only_span_preserves_surrounding_cell_prose(self):
        markdown = "prefix $$\\beta$$ suffix"
        formula = "$$\\beta$$"
        span = {"offset": markdown.index(formula), "length": len(formula)}
        data = {
            "contents": [
                {
                    "markdown": markdown,
                    "tables": [self._table(0, 1, "Rate :formula: today", span)],
                }
            ]
        }
        outputs = build_metric_outputs("azure-cu", data, None)
        self.assertIn("Rate β today", outputs["ocr_largest"])

    def test_recovers_formula_from_selfhost_utf16_span(self):
        markdown = "😀 prefix $$\\beta$$ suffix"
        formula = "$$\\beta$$"
        prefix = markdown[: markdown.index(formula)]
        span = {
            "offset": len(prefix.encode("utf-16-le")) // 2,
            "length": len(formula.encode("utf-16-le")) // 2,
        }
        data = {
            "stringEncoding": "utf16",
            "contents": [
                {
                    "markdown": markdown,
                    "tables": [self._table(0, 1, ":formula:", span)],
                }
            ],
        }
        outputs = build_metric_outputs("azure-cu", data, None)
        self.assertIn("β", outputs["ocr_largest"])

    def test_missing_formula_span_drops_placeholder_without_fabrication(self):
        data = {
            "contents": [
                {"markdown": "", "tables": [self._table(0, 1, ":formula:")]}
            ]
        }
        outputs = build_metric_outputs("azure-cu", data, None)
        self.assertNotIn(":formula:", outputs["ocr_largest"])

    def test_geometry_concat_merges_vertical_fragments(self):
        data = {
            "contents": [
                {
                    "markdown": "",
                    "tables": [
                        self._table(0, 1, "A"),
                        self._table(1, 2, "B"),
                    ],
                }
            ]
        }
        outputs = build_metric_outputs("azure-cu", data, None)
        self.assertEqual(outputs["raw_largest"].count("<tr>"), 1)
        self.assertEqual(outputs["ocr_merged"].count("<tr>"), 2)
        self.assertEqual(outputs["html_merged"].count("<tr>"), 2)

    def test_geometry_concat_merges_horizontal_fragments(self):
        data = {
            "contents": [
                {
                    "markdown": "",
                    "tables": [
                        self._table(0, 1, "A", left=0, right=1),
                        self._table(0, 1, "B", left=1, right=2),
                    ],
                }
            ]
        }
        outputs = build_metric_outputs("azure-cu", data, None)
        self.assertIn("<td>A</td><td>B</td>", outputs["ocr_merged"])

    def test_geometry_concat_rejects_unrelated_fragments(self):
        cases = {
            "different_page": [
                self._table(0, 1, "A", page=1),
                self._table(1, 2, "B", page=2),
            ],
            "far_apart": [self._table(0, 1, "A"), self._table(3, 4, "B")],
            "overlapping": [
                self._table(0, 1, "A"),
                self._table(0.2, 0.8, "B"),
            ],
        }
        for name, tables in cases.items():
            with self.subTest(name=name):
                outputs = build_metric_outputs(
                    "azure-cu",
                    {"contents": [{"markdown": "", "tables": tables}]},
                    None,
                )
                self.assertEqual(outputs["ocr_merged"].count("<tr>"), 1)


class MistralMetricOutputTests(unittest.TestCase):
    @staticmethod
    def _table(value: str) -> dict[str, str]:
        return {
            "format": "html",
            "content": f"<table><tr><td>{value}</td></tr></table>",
        }

    def test_merges_ordered_same_page_html_fragments(self):
        data = {"pages": [{"tables": [self._table("A"), self._table("B")]}]}
        fallback = "\n".join(table["content"] for table in data["pages"][0]["tables"])
        outputs = build_metric_outputs("mistral", data, fallback)
        self.assertEqual(outputs["html_merged"].count("<tr>"), 2)

    def test_does_not_merge_fragments_across_pages(self):
        data = {
            "pages": [
                {"tables": [self._table("A")]},
                {"tables": [self._table("B")]},
            ]
        }
        fallback = "\n".join(
            page["tables"][0]["content"] for page in data["pages"]
        )
        outputs = build_metric_outputs("mistral", data, fallback)
        self.assertEqual(outputs["html_merged"].count("<tr>"), 1)


class MetricScoringTests(unittest.TestCase):
    def test_standard_teds_struct_canonicalizes_header_cell_tags(self):
        ground_truth = "<table><tr><td>A</td></tr></table>"
        outputs = html_metric_outputs("<table><tr><th>A</th></tr></table>")
        scores = score_metric_outputs(ground_truth, outputs)
        self.assertEqual(scores["teds_struct_raw_largest"], 1.0)

    def test_teds_struct_tracks_normalize_and_merge_fragments(self):
        ground_truth = (
            "<table><tr><td>A</td></tr><tr><td>B</td></tr></table>"
        )
        source = (
            "<table><thead><tr><th>A</th></tr></thead></table>"
            "<table><tr><td>B</td></tr></table>"
        )
        scores = score_metric_outputs(
            ground_truth,
            html_metric_outputs(source, merge_fragments=True),
        )
        self.assertGreater(
            scores["teds_struct_html_largest"],
            scores["teds_struct_raw_largest"],
        )
        self.assertEqual(scores["teds_struct_html_merged"], 1.0)

    def test_scores_all_six_metrics(self):
        ground_truth = "<table><tr><td>✓</td></tr></table>"
        outputs = html_metric_outputs("<table><tr><td>[x]</td></tr></table>")
        scores = score_metric_outputs(ground_truth, outputs)
        self.assertEqual(set(scores), set(METRIC_KEYS))
        self.assertEqual(scores["rd_ocr_largest"], 1.0)
        self.assertEqual(scores["teds_struct_html_largest"], 1.0)


if __name__ == "__main__":
    unittest.main()
