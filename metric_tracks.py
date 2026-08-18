"""Build and score RD-TableBench's versioned 3+3 metric policy."""

from __future__ import annotations

import copy
import re
import statistics
from html import escape
from itertools import pairwise
from typing import Any

import numpy as np
from convert import html_to_numpy
from formula_text import latex_to_visible_text
from grading import table_similarity
from html_normalization import (
    html_metric_outputs,
    normalize_table,
    parse_top_level_tables,
    serialize_table,
)
from teds_struct import structure_signature, teds_struct_score
from text_normalization import normalize_display_text, normalize_scoring_text

EVALUATION_POLICY = "rd-tablebench-3x3-v1"
PRIMARY_METRIC_KEY = "rd_raw_largest"
METRIC_KEYS = (
    PRIMARY_METRIC_KEY,
    "rd_ocr_largest",
    "rd_ocr_merged",
    "teds_struct_raw_largest",
    "teds_struct_html_largest",
    "teds_struct_html_merged",
)
OUTPUT_KEYS = (
    "raw_largest",
    "ocr_largest",
    "ocr_merged",
    "html_largest",
    "html_merged",
)
METRIC_OUTPUTS = {
    "rd_raw_largest": "raw_largest",
    "rd_ocr_largest": "ocr_largest",
    "rd_ocr_merged": "ocr_merged",
    "teds_struct_raw_largest": "raw_largest",
    "teds_struct_html_largest": "html_largest",
    "teds_struct_html_merged": "html_merged",
}

_SOURCE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _source_geometry(
    value: object,
) -> tuple[int, tuple[float, float, float, float]] | None:
    numbers = [float(item) for item in _SOURCE_NUMBER.findall(str(value or ""))]
    if len(numbers) < 9:
        return None
    coordinates = numbers[1:9]
    xs = coordinates[0::2]
    ys = coordinates[1::2]
    return int(numbers[0]), (min(xs), min(ys), max(xs), max(ys))


def _source_bbox(value: object) -> tuple[float, float, float, float] | None:
    geometry = _source_geometry(value)
    return geometry[1] if geometry is not None else None


def _table_page(table: dict[str, Any]) -> int | None:
    geometry = _source_geometry(table.get("source"))
    return geometry[0] if geometry is not None else None


def _table_bbox(table: dict[str, Any]) -> tuple[float, float, float, float] | None:
    return _source_bbox(table.get("source"))


def _cell_bbox(cell: dict[str, Any]) -> tuple[float, float, float, float] | None:
    return _source_bbox(cell.get("source"))


def _table_rank(table: dict[str, Any]) -> tuple[float, int, int, int]:
    box = _table_bbox(table)
    geometric_area = (
        max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) if box else 0.0
    )
    rows = max(0, int(table.get("rowCount", 0)))
    columns = max(0, int(table.get("columnCount", 0)))
    cells = table.get("cells") or []
    return (
        geometric_area,
        rows * columns,
        len(cells),
        sum(len(str(cell.get("content", ""))) for cell in cells),
    )


def _azure_tables(data: dict[str, Any]) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    string_encoding = str(data.get("stringEncoding", "codePoint"))
    for content in data.get("contents", []):
        source_text = str(content.get("markdown", ""))
        for source_table in content.get("tables", []):
            table = copy.deepcopy(source_table)
            table["_source_text"] = source_text
            for cell in table.get("cells", []):
                cell["_source_text"] = source_text
                cell["_string_encoding"] = string_encoding
            tables.append(table)
    return tables


def _span_text(
    source: str,
    offset: int,
    length: int,
    string_encoding: str,
) -> str | None:
    if string_encoding.lower() in {"utf16", "utf-16", "utf16codeunit"}:
        encoded = source.encode("utf-16-le")
        start = offset * 2
        end = (offset + length) * 2
        if end > len(encoded):
            return None
        try:
            return encoded[start:end].decode("utf-16-le")
        except UnicodeDecodeError:
            return None
    return source[offset : offset + length] if offset + length <= len(source) else None


def _cell_text(cell: dict[str, Any], ocr_adapter: bool) -> str:
    raw = str(cell.get("content", ""))
    if ocr_adapter and ":formula:" in raw:
        source_text = str(cell.get("_source_text", ""))
        spans = cell.get("spans")
        if not isinstance(spans, list):
            spans = [cell.get("span")]
        resolved_parts: list[str] = []
        for span in spans:
            if not isinstance(span, dict):
                continue
            try:
                offset = max(0, int(span.get("offset", 0)))
                length = max(0, int(span.get("length", 0)))
            except (TypeError, ValueError):
                continue
            resolved = _span_text(
                source_text,
                offset,
                length,
                str(cell.get("_string_encoding", "codePoint")),
            )
            if resolved:
                resolved_parts.append(resolved)
        resolved = " ".join(resolved_parts).strip()
        if resolved:
            rendered = latex_to_visible_text(resolved)
            plain_tokens = re.findall(
                r"\w+",
                raw.replace(":formula:", ""),
                flags=re.UNICODE,
            )
            rendered_folded = rendered.casefold()
            if plain_tokens and not all(
                token.casefold() in rendered_folded for token in plain_tokens
            ):
                text = raw
                for resolved_part in resolved_parts:
                    text = text.replace(
                        ":formula:",
                        latex_to_visible_text(resolved_part),
                        1,
                    )
                text = text.replace(":formula:", "")
            else:
                text = rendered
        else:
            text = raw.replace(":formula:", "")
    else:
        text = raw
    text = text.replace("\n", " ")
    return normalize_display_text(text) if ocr_adapter else text


def _azure_table_to_html(table: dict[str, Any], ocr_adapter: bool) -> str:
    row_count = int(table.get("rowCount", 0))
    column_count = int(table.get("columnCount", 0))
    if row_count <= 0 or column_count <= 0:
        raise ValueError("Azure Content Understanding returned invalid table dimensions.")

    origins: dict[tuple[int, int], dict[str, Any]] = {}
    covered: set[tuple[int, int]] = set()
    for cell in table.get("cells", []):
        row = int(cell.get("rowIndex", 0))
        column = int(cell.get("columnIndex", 0))
        row_span = max(1, int(cell.get("rowSpan", 1)))
        column_span = max(1, int(cell.get("columnSpan", 1)))
        if not (0 <= row < row_count and 0 <= column < column_count):
            continue
        origins[(row, column)] = cell
        for covered_row in range(row, min(row + row_span, row_count)):
            for covered_column in range(column, min(column + column_span, column_count)):
                if (covered_row, covered_column) != (row, column):
                    covered.add((covered_row, covered_column))

    rows: list[str] = []
    for row in range(row_count):
        rendered: list[str] = []
        column = 0
        while column < column_count:
            if (row, column) in covered:
                column += 1
                continue
            cell = origins.get((row, column))
            if cell is None:
                rendered.append("<td></td>")
                column += 1
                continue
            row_span = max(1, int(cell.get("rowSpan", 1)))
            column_span = max(1, int(cell.get("columnSpan", 1)))
            tag = "th" if "header" in str(cell.get("kind", "")).lower() else "td"
            attributes = ""
            if row_span > 1:
                attributes += f' rowspan="{row_span}"'
            if column_span > 1:
                attributes += f' colspan="{column_span}"'
            rendered.append(
                f"<{tag}{attributes}>{escape(_cell_text(cell, ocr_adapter))}</{tag}>"
            )
            column += column_span
        rows.append("<tr>" + "".join(rendered) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _overlap_ratio(left0: float, left1: float, right0: float, right1: float) -> float:
    overlap = max(0.0, min(left1, right1) - max(left0, right0))
    size = min(left1 - left0, right1 - right0)
    return overlap / size if size > 0 else 0.0


def _stack_orientation(
    tables: list[dict[str, Any]], gap: float = 1.0, overlap_tolerance: float = 0.1
) -> str | None:
    pages = {_table_page(table) for table in tables}
    if len(pages) != 1 or None in pages:
        return None
    boxes = [(table, _table_bbox(table)) for table in tables]
    if len(boxes) < 2 or any(box is None for _, box in boxes):
        return None
    vertical = sorted(boxes, key=lambda item: item[1][1])  # type: ignore[index]
    if all(
        _overlap_ratio(left[1][0], left[1][2], right[1][0], right[1][2]) >= 0.6  # type: ignore[index]
        and -overlap_tolerance
        <= right[1][1] - left[1][3]  # type: ignore[index]
        <= gap
        for left, right in pairwise(vertical)
    ):
        return "vertical"
    horizontal = sorted(boxes, key=lambda item: item[1][0])  # type: ignore[index]
    if all(
        _overlap_ratio(left[1][1], left[1][3], right[1][1], right[1][3]) >= 0.6  # type: ignore[index]
        and -overlap_tolerance
        <= right[1][0] - left[1][2]  # type: ignore[index]
        <= gap
        for left, right in pairwise(horizontal)
    ):
        return "horizontal"
    return None


def _axis_boundaries(
    table: dict[str, Any],
    count_name: str,
    index_name: str,
    span_name: str,
    axis: int,
) -> list[float] | None:
    count = int(table.get(count_name, 0))
    box = _table_bbox(table)
    if count <= 0 or box is None:
        return None
    candidates: list[list[float]] = [[] for _ in range(count + 1)]
    candidates[0].append(box[axis])
    candidates[-1].append(box[axis + 2])
    for cell in table.get("cells", []):
        cell_box = _cell_bbox(cell)
        if cell_box is None:
            continue
        start = max(0, min(count, int(cell.get(index_name, 0))))
        span = max(1, int(cell.get(span_name, 1)))
        end = max(start, min(count, start + span))
        candidates[start].append(cell_box[axis])
        candidates[end].append(cell_box[axis + 2])

    boundaries: list[float | None] = [
        statistics.median(values) if values else None for values in candidates
    ]
    known = [index for index, value in enumerate(boundaries) if value is not None]
    if not known or known[0] != 0 or known[-1] != count:
        return None
    for left, right in pairwise(known):
        if right == left + 1:
            continue
        value0 = boundaries[left]
        value1 = boundaries[right]
        assert value0 is not None and value1 is not None
        step = (value1 - value0) / (right - left)
        for index in range(left + 1, right):
            boundaries[index] = value0 + step * (index - left)
    if any(value is None for value in boundaries):
        return None
    output = [float(value) for value in boundaries if value is not None]
    if any(left >= right for left, right in pairwise(output)):
        return None
    return output


def _map_boundaries(
    boundaries: list[float], reference: list[float], local_tolerance: float = 0.30
) -> list[int] | None:
    mapping: list[int] = []
    for index, value in enumerate(boundaries):
        target = min(range(len(reference)), key=lambda item: abs(reference[item] - value))
        difference = abs(reference[target] - value)
        if index in {0, len(boundaries) - 1} and target in {0, len(reference) - 1}:
            tolerance = 0.1 * (reference[-1] - reference[0])
        else:
            source_intervals = []
            if index:
                source_intervals.append(value - boundaries[index - 1])
            if index + 1 < len(boundaries):
                source_intervals.append(boundaries[index + 1] - value)
            reference_intervals = []
            if target:
                reference_intervals.append(reference[target] - reference[target - 1])
            if target + 1 < len(reference):
                reference_intervals.append(reference[target + 1] - reference[target])
            tolerance = local_tolerance * min(source_intervals + reference_intervals)
        if difference > tolerance:
            return None
        mapping.append(target)
    if any(left >= right for left, right in pairwise(mapping)):
        return None
    return mapping


def _row_mappings(tables: list[dict[str, Any]]) -> list[list[int]] | None:
    boxes = [_table_bbox(table) for table in tables]
    if (
        len(tables) < 2
        or any(box is None for box in boxes)
        or _stack_orientation(tables) != "horizontal"
    ):
        return None
    concrete_boxes = [box for box in boxes if box is not None]
    if any(
        _overlap_ratio(left[1], left[3], right[1], right[3]) < 0.9
        for index, left in enumerate(concrete_boxes)
        for right in concrete_boxes[index + 1 :]
    ):
        return None
    boundaries = [
        _axis_boundaries(table, "rowCount", "rowIndex", "rowSpan", 1)
        for table in tables
    ]
    if any(item is None for item in boundaries):
        return None
    concrete = [item for item in boundaries if item is not None]
    reference_index = max(
        enumerate(tables), key=lambda item: int(item[1].get("rowCount", 0))
    )[0]
    reference = concrete[reference_index]
    mappings = [
        list(range(len(reference)))
        if index == reference_index
        else _map_boundaries(item, reference)
        for index, item in enumerate(concrete)
    ]
    return None if any(item is None for item in mappings) else [
        item for item in mappings if item is not None
    ]


def _column_mappings(tables: list[dict[str, Any]]) -> list[list[int]] | None:
    if len(tables) < 2 or _stack_orientation(tables) != "vertical":
        return None
    boundaries = [
        _axis_boundaries(table, "columnCount", "columnIndex", "columnSpan", 0)
        for table in tables
    ]
    if any(item is None for item in boundaries):
        return None
    concrete = [item for item in boundaries if item is not None]
    reference_index = max(
        enumerate(tables), key=lambda item: int(item[1].get("columnCount", 0))
    )[0]
    reference = concrete[reference_index]
    mappings = [
        list(range(len(reference)))
        if index == reference_index
        else _map_boundaries(item, reference)
        for index, item in enumerate(concrete)
    ]
    return None if any(item is None for item in mappings) else [
        item for item in mappings if item is not None
    ]


def _compatible_orientation(tables: list[dict[str, Any]]) -> str | None:
    if len(tables) < 2:
        return None
    if _column_mappings(tables) is not None:
        return "vertical"
    if _row_mappings(tables) is not None:
        return "horizontal"
    boxes = [_table_bbox(table) for table in tables]
    if any(box is not None for box in boxes):
        return None
    same_columns = len({int(table.get("columnCount", 0)) for table in tables}) == 1
    same_rows = len({int(table.get("rowCount", 0)) for table in tables}) == 1
    if not same_columns and not same_rows:
        return None
    if same_columns and not same_rows:
        return "vertical"
    if same_rows and not same_columns:
        return "horizontal"
    return "vertical"


def _concat_tables(
    tables: list[dict[str, Any]], orientation: str | None
) -> dict[str, Any] | None:
    if orientation == "vertical":
        ordered = sorted(tables, key=lambda table: (_table_bbox(table) or (0, 0, 0, 0))[1])
        column_counts = {int(table.get("columnCount", 0)) for table in ordered}
        if len(column_counts) == 1:
            column_count = column_counts.pop()
            mappings = [list(range(column_count + 1)) for _ in ordered]
        else:
            mappings = _column_mappings(ordered)
            if mappings is None:
                return None
        cells: list[dict[str, Any]] = []
        row_offset = 0
        for table, column_mapping in zip(ordered, mappings):
            for source_cell in table.get("cells", []):
                cell = copy.deepcopy(source_cell)
                column = int(cell.get("columnIndex", 0))
                span = max(1, int(cell.get("columnSpan", 1)))
                end = min(len(column_mapping) - 1, column + span)
                cell["rowIndex"] = int(cell.get("rowIndex", 0)) + row_offset
                cell["columnIndex"] = column_mapping[column]
                cell["columnSpan"] = column_mapping[end] - column_mapping[column]
                cells.append(cell)
            row_offset += int(table.get("rowCount", 0))
        return {
            "rowCount": row_offset,
            "columnCount": max(mapping[-1] for mapping in mappings),
            "cells": cells,
        }

    if orientation == "horizontal":
        ordered = sorted(tables, key=lambda table: (_table_bbox(table) or (0, 0, 0, 0))[0])
        mappings = _row_mappings(ordered)
        if mappings is None:
            return None
        cells = []
        column_offset = 0
        for table, row_mapping in zip(ordered, mappings):
            for source_cell in table.get("cells", []):
                cell = copy.deepcopy(source_cell)
                row = int(cell.get("rowIndex", 0))
                span = max(1, int(cell.get("rowSpan", 1)))
                end = min(len(row_mapping) - 1, row + span)
                cell["rowIndex"] = row_mapping[row]
                cell["rowSpan"] = row_mapping[end] - row_mapping[row]
                cell["columnIndex"] = int(cell.get("columnIndex", 0)) + column_offset
                cells.append(cell)
            column_offset += int(table.get("columnCount", 0))
        return {
            "rowCount": max(mapping[-1] for mapping in mappings),
            "columnCount": column_offset,
            "cells": cells,
        }
    return None


def _normalize_single_table(source: str) -> str:
    tables = parse_top_level_tables(source)
    if not tables:
        raise ValueError("Generated HTML contains no table.")
    return serialize_table(normalize_table(tables[0]))


def _azure_metric_outputs(data: dict[str, Any]) -> dict[str, str] | None:
    tables = _azure_tables(data)
    if not tables:
        return None
    largest = max(tables, key=_table_rank)
    merged = _concat_tables(tables, _compatible_orientation(tables)) or largest
    raw_largest = _azure_table_to_html(largest, ocr_adapter=False)
    raw_merged = _azure_table_to_html(merged, ocr_adapter=False)
    return {
        "raw_largest": raw_largest,
        "ocr_largest": _azure_table_to_html(largest, ocr_adapter=True),
        "ocr_merged": _azure_table_to_html(merged, ocr_adapter=True),
        "html_largest": _normalize_single_table(raw_largest),
        "html_merged": _normalize_single_table(raw_merged),
    }


def build_metric_outputs(
    provider: str,
    data: dict[str, Any],
    fallback_html: str | None,
) -> dict[str, str]:
    """Create all derived HTML artifacts without using ground truth."""
    if provider == "azure-cu":
        outputs = _azure_metric_outputs(data)
        if outputs is not None:
            return outputs
    if not fallback_html:
        raise ValueError("The provider response contains no HTML table.")
    mistral_html_pages = {
        page_index
        for page_index, page in enumerate(data.get("pages", []))
        for table in page.get("tables", [])
        if str(table.get("format", "")).lower() == "html"
        and str(table.get("content", "")).strip()
    }
    return html_metric_outputs(
        fallback_html,
        merge_fragments=(provider == "mistral" and len(mistral_html_pages) == 1),
    )


def _rd_score(ground_truth: str, prediction: str, normalize: bool) -> float:
    ground_truth_array = html_to_numpy(ground_truth)
    prediction_array = html_to_numpy(prediction)
    if ground_truth_array.size == 0 or prediction_array.size == 0:
        raise ValueError("HTML did not contain a table with cells.")
    if normalize:
        ground_truth_array = np.vectorize(
            normalize_scoring_text, otypes=[object]
        )(ground_truth_array)
        prediction_array = np.vectorize(
            normalize_scoring_text, otypes=[object]
        )(prediction_array)
    return float(table_similarity(ground_truth_array, prediction_array))


def score_metric_outputs(
    ground_truth: str,
    outputs: dict[str, str],
) -> dict[str, float]:
    """Evaluate all six metrics using the fixed 3+3 policy."""
    if set(outputs) != set(OUTPUT_KEYS):
        raise ValueError("Metric output artifacts are incomplete.")
    scores = {
        "rd_raw_largest": _rd_score(
            ground_truth, outputs["raw_largest"], normalize=False
        ),
        "rd_ocr_largest": _rd_score(
            ground_truth, outputs["ocr_largest"], normalize=True
        ),
        "rd_ocr_merged": _rd_score(
            ground_truth, outputs["ocr_merged"], normalize=True
        ),
    }
    teds_cache: dict[tuple, float] = {}
    truth_signature = structure_signature(ground_truth)
    for metric_key, output_key in (
        ("teds_struct_raw_largest", "raw_largest"),
        ("teds_struct_html_largest", "html_largest"),
        ("teds_struct_html_merged", "html_merged"),
    ):
        signature = structure_signature(outputs[output_key])
        if signature == truth_signature:
            teds_cache[signature] = 1.0
        if signature not in teds_cache:
            teds_cache[signature] = teds_struct_score(
                ground_truth, outputs[output_key]
            )
        scores[metric_key] = teds_cache[signature]
    if set(scores) != set(METRIC_KEYS) or any(
        not 0.0 <= score <= 1.0 for score in scores.values()
    ):
        raise ValueError("Metric scores are incomplete or outside [0, 1].")
    return scores
