from __future__ import annotations

import sys
import unittest
from pathlib import Path

UPSTREAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UPSTREAM_ROOT))

from error_analysis import (  # noqa: E402
    ERROR_ANALYSIS_VERSION,
    ERROR_CATEGORIES,
    build_error_analysis,
    detected_table_count,
    summarize_error_analysis,
)


class ErrorAnalysisTests(unittest.TestCase):
    @staticmethod
    def _mistral(*tables: str) -> dict:
        return {
            "pages": [
                {
                    "tables": [
                        {"format": "html", "content": table}
                        for table in tables
                    ]
                }
            ]
        }

    def _analysis(
        self,
        ground_truth: str,
        prediction: str | None,
        *,
        data: dict | None = None,
        score: float | None = 0.5,
        status: str = "scored",
    ) -> dict[str, object]:
        provider_data = data if data is not None else self._mistral(prediction or "")
        return build_error_analysis(
            provider="mistral",
            data=provider_data,
            fallback_html=prediction,
            ground_truth_html=ground_truth,
            outputs={"html_merged": prediction} if prediction else None,
            score=score,
            status=status,
        )

    def test_assigns_requested_categories_with_exclusive_precedence(self):
        two_cells = "<table><tr><td>A</td><td>B</td></tr></table>"
        cases = {
            "ocr_error": self._analysis(
                two_cells,
                "<table><tr><td>X</td><td>Y</td></tr></table>",
            ),
            "table_missed": self._analysis(
                two_cells,
                None,
                data={"pages": [{"tables": [], "markdown": "plain text"}]},
                score=None,
                status="evaluation_error",
            ),
            "table_over_detected": self._analysis(
                two_cells,
                two_cells,
                data=self._mistral(
                    "<table><tr><td>A</td></tr></table>",
                    "<table><tr><td>B</td></tr></table>",
                ),
            ),
            "table_region_incomplete": self._analysis(
                (
                    "<table><tr><td>A</td><td>B</td><td>C</td>"
                    "<td>D</td><td>E</td></tr></table>"
                ),
                "<table><tr><td>A</td></tr></table>",
            ),
            "cell_over_split": self._analysis(
                two_cells,
                "<table><tr><td>A</td><td>B</td><td>extra</td></tr></table>",
            ),
            "cell_under_split": self._analysis(
                "<table><tr><td>A</td><td>B</td><td>C</td></tr></table>",
                "<table><tr><td>AB</td><td>C</td></tr></table>",
            ),
            "other_error": self._analysis(
                two_cells,
                "<table><tr><td>A</td></tr><tr><td>B</td></tr></table>",
            ),
        }

        self.assertEqual(set(cases), set(ERROR_CATEGORIES))
        for expected, analysis in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(analysis["version"], ERROR_ANALYSIS_VERSION)
                self.assertEqual(analysis["category"], expected)
                self.assertTrue(analysis["score_lost"])

        incomplete = cases["table_region_incomplete"]
        self.assertEqual(incomplete["gt_text_line_coverage"], 0.2)
        self.assertEqual(incomplete["covered_gt_text_line_count"], 1)
        self.assertEqual(incomplete["total_gt_text_line_count"], 5)

    def test_perfect_score_has_no_error_category(self):
        table = "<table><tr><td>A</td></tr></table>"
        analysis = self._analysis(table, table, score=1.0)

        self.assertIsNone(analysis["category"])
        self.assertFalse(analysis["score_lost"])

    def test_counts_selfhost_markdown_tables_without_structured_arrays(self):
        data = {
            "contents": [
                {
                    "markdown": (
                        "<table><tr><td>A</td></tr></table>"
                        "<table><tr><td>B</td></tr></table>"
                    )
                }
            ]
        }

        self.assertEqual(detected_table_count("azure-cu", data, None), 2)

    def test_counts_every_markdown_table_block_for_both_providers(self):
        markdown = (
            "| A | X |\n| --- | --- |\n| 1 | x |\n\n"
            "paragraph\n\n"
            "| B | Y |\n| --- | --- |\n| 2 | y |"
        )
        azure = {"contents": [{"markdown": markdown}]}
        mistral = {"pages": [{"markdown": markdown, "tables": []}]}

        self.assertEqual(detected_table_count("azure-cu", azure, None), 2)
        self.assertEqual(detected_table_count("mistral", mistral, None), 2)

    def test_authoritative_table_objects_are_not_double_counted_with_markdown(self):
        markdown = "| duplicate | value |\n| --- | --- |\n| A | B |"
        azure = {
            "contents": [
                {
                    "markdown": markdown,
                    "tables": [{"rowCount": 1, "columnCount": 2}],
                }
            ]
        }
        mistral = {
            "pages": [
                {
                    "markdown": markdown,
                    "tables": [
                        {
                            "format": "html",
                            "content": "<table><tr><td>A</td><td>B</td></tr></table>",
                        }
                    ],
                }
            ]
        }

        self.assertEqual(detected_table_count("azure-cu", azure, None), 1)
        self.assertEqual(detected_table_count("mistral", mistral, None), 1)

    def test_no_gt_text_is_reported_as_not_measurable(self):
        ground_truth = "<table><tr><td></td></tr></table>"
        prediction = "<table><tr><td>noise</td></tr></table>"
        analysis = self._analysis(ground_truth, prediction)

        self.assertIsNone(analysis["gt_text_line_coverage"])
        self.assertEqual(analysis["total_gt_text_line_count"], 0)

    def test_summary_partitions_every_score_loss_case(self):
        table = "<table><tr><td>A</td></tr></table>"
        error = self._analysis(table, None, data={"pages": []}, score=None, status="evaluation_error")
        perfect = self._analysis(table, table, score=1.0)
        summary = summarize_error_analysis(
            [
                {"error_analysis": error},
                {"error_analysis": perfect},
            ]
        )

        self.assertEqual(summary["total_count"], 2)
        self.assertEqual(summary["score_loss_count"], 1)
        self.assertEqual(summary["no_error_count"], 1)
        self.assertEqual(summary["categories"]["table_missed"], 1)
        self.assertEqual(sum(summary["categories"].values()), 1)


if __name__ == "__main__":
    unittest.main()
