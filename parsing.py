"""
For each format, this code extracts the largest HTML table from the response.
"""

from html import escape
import json
from typing import Any
import os


def _content_understanding_table_to_html(table: dict[str, Any]) -> str:
    rows: list[list[str]] = [[] for _ in range(table["rowCount"])]
    for cell in table["cells"]:
        tag = "th" if "header" in cell.get("kind", "").lower() else "td"
        attributes = ""
        for attribute, key in (("rowspan", "rowSpan"), ("colspan", "columnSpan")):
            if cell.get(key, 1) > 1:
                attributes += f' {attribute}="{cell[key]}"'
        content = escape(cell.get("content", "").replace("\n", " "))
        rows[cell["rowIndex"]].append(f"<{tag}{attributes}>{content}</{tag}>")
    return "<table>" + "".join(
        "<tr>" + "".join(row) + "</tr>" for row in rows
    ) + "</table>"


def parse_azure_content_understanding_response(path: str) -> tuple[str | None, Any]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    payload = data.get("analyzeResult") or data
    tables = [
        table
        for content in payload.get("contents", [])
        for table in content.get("tables", [])
    ]
    largest = max(
        tables,
        key=lambda table: table["rowCount"] * table["columnCount"],
        default=None,
    )
    return (
        _content_understanding_table_to_html(largest) if largest else None,
        data,
    )


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
