"""Normalize provider responses into HTML consumed by the native evaluator."""

import json
import html
import re
from typing import Any
import os


def _extract_html_table(markdown: str) -> str | None:
    tables = re.findall(
        r"<table\b[^>]*>.*?</table>",
        markdown,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if tables:
        return max(tables, key=len)

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
    candidates = [
        block
        for block in blocks
        if len(block) >= 2
        and re.fullmatch(
            r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*",
            block[1],
        )
    ]
    if not candidates:
        return None

    rows: list[str] = []
    for row_index, line in enumerate(max(candidates, key=len)):
        if row_index == 1:
            continue
        cells = [html.escape(cell.strip()) for cell in line.strip("|").split("|")]
        tag = "th" if row_index == 0 else "td"
        rows.append("<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _azure_content_understanding_table_to_html(table: dict[str, Any]) -> str:
    row_count = int(table.get("rowCount", 0))
    column_count = int(table.get("columnCount", 0))
    if row_count <= 0 or column_count <= 0:
        raise ValueError("Azure Content Understanding returned invalid table dimensions.")
    origins: dict[tuple[int, int], dict[str, Any]] = {}
    covered: set[tuple[int, int]] = set()
    for cell in table.get("cells", []):
        row = int(cell.get("rowIndex", 0))
        column = int(cell.get("columnIndex", 0))
        row_span = int(cell.get("rowSpan", 1))
        column_span = int(cell.get("columnSpan", 1))
        origins[(row, column)] = cell
        for covered_row in range(row, min(row + row_span, row_count)):
            for covered_column in range(column, min(column + column_span, column_count)):
                if (covered_row, covered_column) != (row, column):
                    covered.add((covered_row, covered_column))

    rows: list[str] = []
    for row in range(row_count):
        rendered_cells: list[str] = []
        column = 0
        while column < column_count:
            if (row, column) in covered:
                column += 1
                continue
            cell = origins.get((row, column))
            if cell is None:
                rendered_cells.append("<td></td>")
                column += 1
                continue
            row_span = int(cell.get("rowSpan", 1))
            column_span = int(cell.get("columnSpan", 1))
            tag = "th" if "header" in str(cell.get("kind", "")).lower() else "td"
            attributes = ""
            if row_span > 1:
                attributes += f' rowspan="{row_span}"'
            if column_span > 1:
                attributes += f' colspan="{column_span}"'
            content = html.escape(str(cell.get("content", "")))
            rendered_cells.append(f"<{tag}{attributes}>{content}</{tag}>")
            column += column_span
        rows.append("<tr>" + "".join(rendered_cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def parse_azure_content_understanding_response(path: str) -> tuple[str | None, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    contents = data.get("contents", [])
    tables = [table for content in contents for table in content.get("tables", [])]
    if tables:
        largest = max(
            tables,
            key=lambda table: int(table.get("rowCount", 0))
            * int(table.get("columnCount", 0)),
        )
        return _azure_content_understanding_table_to_html(largest), data
    markdown = "\n\n".join(str(content.get("markdown", "")) for content in contents)
    return _extract_html_table(markdown), data


def parse_mistral_ocr_response(path: str) -> tuple[str | None, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    html_tables = [
        str(table.get("content", "")).strip()
        for page in data.get("pages", [])
        for table in page.get("tables", [])
        if str(table.get("format", "")).lower() == "html"
        and str(table.get("content", "")).strip()
    ]
    if html_tables:
        # Each benchmark case contains one logical table. Mistral may split that
        # table into ordered fragments, so retain all fragments for native row
        # alignment instead of dropping every fragment except the largest.
        return "\n".join(html_tables), data
    markdown = "\n\n".join(str(page.get("markdown", "")) for page in data.get("pages", []))
    return _extract_html_table(markdown), data


def parse_textract_response(path: str) -> tuple[str | None, Any]:
    if not os.path.exists(path):
        return None, None

    with open(path, "r") as f:
        data = json.load(f)

    return data["html_table"], data


def parse_gcloud_response(path: str) -> tuple[str | None, Any]:
    if not os.path.exists(path):
        return None, None

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return None, None

    return data["html_table"], data


def parse_reducto_response(path: str) -> tuple[str | None, Any]:
    if not os.path.exists(path):
        return None, None

    with open(path, "r") as f:
        data = json.load(f)

    if "error" in data:
        return None, data

    longest_html = None
    max_length = 0

    for chunk in data["result"]["chunks"]:
        blocks = chunk["blocks"]
        for block in blocks:
            if block["type"] == "Table":
                if len(block["content"]) > max_length:
                    max_length = len(block["content"])
                    longest_html = block["content"]

    return longest_html, data


def parse_chunkr_response(path: str) -> tuple[str | None, Any]:
    if not os.path.exists(path):
        return None, None

    with open(path, "r") as f:
        data = json.load(f)

    if data.get("status") != "Succeeded":
        return None, data

    largest_html = None
    max_length = 0

    try:
        for output in (
            data.get("output", [])
            if "chunks" not in data.get("output")
            else data["output"]["chunks"]
        ):
            for segment in output.get("segments", []):
                if segment.get("segment_type") == "Table" and segment.get("html"):
                    if len(segment["html"]) > max_length:
                        max_length = len(segment["html"])
                        largest_html = segment["html"]
    except Exception:
        import traceback

        traceback.print_exc()
        print(data)

    return largest_html, data


def parse_unstructured_response(path: str) -> tuple[str | None, Any]:
    if not os.path.exists(path):
        return None, None

    with open(path, "r") as f:
        data = json.load(f)

    largest_html = None
    max_length = 0

    for element in data.get("elements", []):
        if element.get("type") == "Table" and element.get("metadata", {}).get(
            "text_as_html"
        ):
            html = element["metadata"]["text_as_html"]
            if len(html) > max_length:
                max_length = len(html)
                largest_html = html

    return largest_html, data


def parse_gpt4o_response(path: str) -> tuple[str | None, Any]:
    if not os.path.exists(path):
        return None, None

    with open(path, "r") as f:
        data = json.load(f)

    html = data["html_table"]
    # Extract just the table portion between <table> and </table>
    start = html.find("<table>")
    end = html.find("</table>") + 8
    if start != -1 and end != -1:
        return html[start:end], data
    return None, data


def parse_azure_response(path: str) -> tuple[str | None, Any]:
    data = None
    try:
        with open(path, "r") as f:
            data = json.load(f)

        def azure_to_html(table: Any) -> str:
            html = "<table>"
            for row_index in range(table["rowCount"]):
                html += "<tr>"
                for col_index in range(table["columnCount"]):
                    cell = next(
                        (
                            c
                            for c in table["cells"]
                            if c["rowIndex"] == row_index
                            and c["columnIndex"] == col_index
                        ),
                        None,
                    )
                    if cell:
                        content = (
                            cell["content"]
                            .replace(":selected:", "")
                            .replace(":unselected:", "")
                        )
                        tag = "th" if cell.get("kind") == "columnHeader" else "td"
                        rowspan = (
                            f" rowspan='{cell['rowSpan']}'" if "rowSpan" in cell else ""
                        )
                        colspan = (
                            f" colspan='{cell['columnSpan']}'"
                            if "columnSpan" in cell
                            else ""
                        )
                        html += f"<{tag}{rowspan}{colspan}>{content}</{tag}>"
                    else:
                        pass
                html += "</tr>"
            html += "</table>"
            return html

        # Find table with largest area (row count * column count)
        largest_table = max(
            data["tables"], key=lambda t: t["rowCount"] * t["columnCount"]
        )
        return azure_to_html(largest_table), data
    except Exception:
        return None, data
