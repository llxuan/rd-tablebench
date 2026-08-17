"""Versioned, ground-truth-aware diagnostics for RD-TableBench errors."""

from __future__ import annotations

import re
from typing import Any

from html_normalization import (
    direct_rows_and_cells,
    parse_top_level_tables,
    select_largest,
)
from teds_struct import structure_signature
from text_normalization import normalize_scoring_text

ERROR_ANALYSIS_VERSION = "rd-error-analysis-v1"
GT_TEXT_COVERAGE_THRESHOLD = 0.80
TEXT_MATCH_TRIGRAM_RECALL = 0.65
ERROR_CATEGORIES = (
    "ocr_error",
    "table_missed",
    "table_over_detected",
    "table_region_incomplete",
    "cell_over_split",
    "cell_under_split",
    "other_error",
)


def _payload(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    nested = data.get("analyzeResult")
    return nested if isinstance(nested, dict) else data


def _safe_tables(source: str | None) -> list[Any]:
    if not str(source or "").strip():
        return []
    try:
        return parse_top_level_tables(str(source))
    except (TypeError, ValueError):
        return []


def _azure_structured_tables(data: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _payload(data)
    tables = [
        table
        for content in payload.get("contents", [])
        if isinstance(content, dict)
        for table in content.get("tables", [])
        if isinstance(table, dict)
    ]
    if tables:
        return tables
    return [
        table
        for table in payload.get("tables", [])
        if isinstance(table, dict)
    ]


def _azure_markdown(data: dict[str, Any]) -> str:
    return "\n\n".join(
        str(content.get("markdown", ""))
        for content in _payload(data).get("contents", [])
        if isinstance(content, dict)
    )


def _mistral_html_tables(data: dict[str, Any]) -> list[str]:
    return [
        str(table.get("content", "")).strip()
        for page in _payload(data).get("pages", [])
        if isinstance(page, dict)
        for table in page.get("tables", [])
        if isinstance(table, dict)
        and str(table.get("format", "")).lower() == "html"
        and str(table.get("content", "")).strip()
    ]


def _markdown_table_count(markdown: str) -> int:
    # Match parsing._extract_html_table precedence. Providers may include an
    # HTML table and an alternate Markdown rendering of that same table; count
    # the authoritative HTML representation instead of double-counting both.
    html_tables = _safe_tables(markdown)
    if html_tables:
        return len(html_tables)
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in markdown.splitlines():
        if "|" in line:
            current.append(line.strip())
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return sum(
        len(block) >= 2
        and re.fullmatch(
            r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*",
            block[1],
        )
        is not None
        for block in blocks
    )


def detected_table_count(
    provider: str,
    data: dict[str, Any] | None,
    fallback_html: str | None,
) -> int | None:
    """Count provider-emitted table results before largest-table selection."""
    if not isinstance(data, dict):
        return None
    if provider == "azure-cu":
        # Structured SDK tables are authoritative when present; Markdown is an
        # alternate rendering of the same analyze result.
        structured = _azure_structured_tables(data)
        if structured:
            return len(structured)
        markdown_count = _markdown_table_count(_azure_markdown(data))
        if markdown_count:
            return markdown_count
    elif provider == "mistral":
        # pages[].tables is authoritative; page Markdown commonly contains
        # links or duplicate renderings for these exact table objects.
        mistral_tables = _mistral_html_tables(data)
        if mistral_tables:
            return len(mistral_tables)
        markdown_count = _markdown_table_count(
            "\n\n".join(
                str(page.get("markdown", ""))
                for page in _payload(data).get("pages", [])
                if isinstance(page, dict)
            )
        )
        if markdown_count:
            return markdown_count
    else:
        raise ValueError(f"Unsupported diagnostic provider: {provider}")
    return len(_safe_tables(fallback_html))


def _normalized_text(value: object) -> str:
    return "".join(
        character
        for character in normalize_scoring_text(value)
        if character.isalnum()
    )


def _trigrams(value: str) -> set[str]:
    return {value[index : index + 3] for index in range(max(0, len(value) - 2))}


def _line_is_covered(expected: str, candidates: list[str]) -> bool:
    if not expected:
        return False
    expected_trigrams = _trigrams(expected)
    for candidate in candidates:
        if expected in candidate:
            return True
        if not expected_trigrams:
            if expected == candidate:
                return True
            continue
        candidate_trigrams = _trigrams(candidate)
        if (
            len(expected_trigrams & candidate_trigrams) / len(expected_trigrams)
            >= TEXT_MATCH_TRIGRAM_RECALL
        ):
            return True
    return False


def _largest_table(source: str | None) -> Any | None:
    return select_largest(_safe_tables(source))


def _table_text_lines(table: Any | None) -> list[str]:
    if table is None:
        return []
    return [
        normalized
        for row in direct_rows_and_cells(table)
        for cell in row
        for normalized in [_normalized_text("".join(cell.itertext()))]
        if normalized
    ]


def _prediction_text_candidates(table: Any | None) -> list[str]:
    if table is None:
        return []
    rows = direct_rows_and_cells(table)
    cells = [
        normalized
        for row in rows
        for cell in row
        for normalized in [_normalized_text("".join(cell.itertext()))]
        if normalized
    ]
    row_text = [
        normalized
        for row in rows
        for normalized in [
            _normalized_text("".join("".join(cell.itertext()) for cell in row))
        ]
        if normalized
    ]
    corpus = "".join(cells)
    return cells + row_text + ([corpus] if corpus else [])


def _logical_cell_count(table: Any | None) -> int | None:
    if table is None:
        return None
    return sum(len(row) for row in direct_rows_and_cells(table))


def _gt_text_line_coverage(
    ground_truth_table: Any | None,
    prediction_table: Any | None,
) -> tuple[float | None, int, int]:
    expected_lines = _table_text_lines(ground_truth_table)
    if not expected_lines:
        return None, 0, 0
    candidates = _prediction_text_candidates(prediction_table)
    covered = sum(
        _line_is_covered(expected, candidates) for expected in expected_lines
    )
    return covered / len(expected_lines), covered, len(expected_lines)


def build_error_analysis(
    *,
    provider: str,
    data: dict[str, Any] | None,
    fallback_html: str | None,
    ground_truth_html: str,
    outputs: dict[str, str] | None,
    score: float | None,
    status: str,
) -> dict[str, object]:
    """Build one exclusive diagnostic category without changing official scores.

    The text-coverage feature migrates the reference report's NFKC/fuzzy matching
    idea to a provider-neutral GT-cell text-line proxy. It is post-evaluation,
    explicitly ground-truth-aware, and never participates in inference or scoring.
    """
    ground_truth_tables = _safe_tables(ground_truth_html)
    ground_truth_table = select_largest(ground_truth_tables)
    prediction_html = None
    if isinstance(outputs, dict):
        prediction_html = outputs.get("html_merged") or outputs.get("html_largest")
    prediction_html = prediction_html or fallback_html
    prediction_table = _largest_table(prediction_html)

    table_count = detected_table_count(provider, data, fallback_html)
    gt_table_count = len(ground_truth_tables)
    table_detection_ok = (
        table_count is not None
        and gt_table_count > 0
        and table_count == gt_table_count
    )
    structure_correct = (
        prediction_table is not None
        and structure_signature(prediction_html or "")
        == structure_signature(ground_truth_html)
    )
    coverage, covered_lines, total_lines = _gt_text_line_coverage(
        ground_truth_table,
        prediction_table,
    )
    predicted_cells = _logical_cell_count(prediction_table)
    gt_cells = _logical_cell_count(ground_truth_table)
    provider_response_success = isinstance(data, dict)
    score_lost = (
        status != "scored"
        or score is None
        or float(score) < 1.0 - 1e-12
    )

    category: str | None = None
    if score_lost:
        if table_detection_ok and structure_correct:
            category = "ocr_error"
        elif provider_response_success and table_count == 0:
            category = "table_missed"
        elif table_count is not None and table_count > 1:
            category = "table_over_detected"
        elif (
            table_count is not None
            and table_count >= 1
            and coverage is not None
            and coverage < GT_TEXT_COVERAGE_THRESHOLD
        ):
            category = "table_region_incomplete"
        elif (
            table_detection_ok
            and predicted_cells is not None
            and gt_cells is not None
            and predicted_cells > gt_cells
        ):
            category = "cell_over_split"
        elif (
            table_detection_ok
            and predicted_cells is not None
            and gt_cells is not None
            and predicted_cells < gt_cells
        ):
            category = "cell_under_split"
        else:
            category = "other_error"

    if category is not None and category not in ERROR_CATEGORIES:
        raise ValueError(f"Unsupported RD error category: {category}")
    return {
        "version": ERROR_ANALYSIS_VERSION,
        "diagnostic_only": True,
        "uses_ground_truth": True,
        "category": category,
        "score_lost": score_lost,
        "provider_response_success": provider_response_success,
        "detected_table_count": table_count,
        "gt_table_count": gt_table_count,
        "table_detection_ok": table_detection_ok,
        "tsr_structure_correct": structure_correct,
        "gt_text_line_coverage": coverage,
        "covered_gt_text_line_count": covered_lines,
        "total_gt_text_line_count": total_lines,
        "gt_text_coverage_threshold": GT_TEXT_COVERAGE_THRESHOLD,
        "text_match_trigram_recall": TEXT_MATCH_TRIGRAM_RECALL,
        "predicted_logical_cell_count": predicted_cells,
        "gt_logical_cell_count": gt_cells,
    }


def summarize_error_analysis(
    results: list[dict[str, object]],
) -> dict[str, object]:
    categories = {category: 0 for category in ERROR_CATEGORIES}
    score_loss_count = 0
    for result in results:
        analysis = result.get("error_analysis")
        if not isinstance(analysis, dict):
            raise ValueError("RD result is missing error analysis.")
        if analysis.get("version") != ERROR_ANALYSIS_VERSION:
            raise ValueError("RD result uses an incompatible error analysis version.")
        score_lost = analysis.get("score_lost")
        category = analysis.get("category")
        if not isinstance(score_lost, bool):
            raise TypeError("RD error analysis score_lost must be boolean.")
        score_loss_count += int(score_lost)
        if category is not None:
            if category not in categories:
                raise ValueError(f"Unsupported RD error category: {category}")
            categories[str(category)] += 1
        if score_lost is not (category is not None):
            raise ValueError("Every score-loss case must have exactly one error category.")
    return {
        "version": ERROR_ANALYSIS_VERSION,
        "diagnostic_only": True,
        "uses_ground_truth": True,
        "gt_text_coverage_threshold": GT_TEXT_COVERAGE_THRESHOLD,
        "text_match_trigram_recall": TEXT_MATCH_TRIGRAM_RECALL,
        "total_count": len(results),
        "score_loss_count": score_loss_count,
        "no_error_count": len(results) - score_loss_count,
        "categories": categories,
    }
