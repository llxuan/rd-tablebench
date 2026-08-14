"""Ground-truth-blind HTML table selection, adaptation, and normalization."""

from __future__ import annotations

import copy
import re

from formula_text import latex_to_visible_text
from lxml import etree, html
from text_normalization import normalize_display_text

SECTIONS = {"thead", "tbody", "tfoot"}
CELLS = {"td", "th"}
LATEX_SIGNAL = re.compile(
    r"\$\$.*?\$\$|(?<!\$)\$[^$]+\$(?!\$)|\\[A-Za-z]+|\\[_%&#$]|[\^_]\{",
    flags=re.DOTALL,
)


def parse_top_level_tables(source: str) -> list[html.HtmlElement]:
    if not str(source or "").strip():
        return []
    document = html.fromstring(source)
    if str(document.tag).lower() == "table":
        return [document]
    return document.xpath(".//table[not(ancestor::table)]")


def direct_rows_and_cells(table: html.HtmlElement) -> list[list[html.HtmlElement]]:
    rows: list[list[html.HtmlElement]] = []

    def consume(container: html.HtmlElement) -> None:
        orphan: list[html.HtmlElement] = []

        def flush() -> None:
            if orphan:
                rows.append(list(orphan))
                orphan.clear()

        for child in container:
            tag = str(child.tag).lower() if isinstance(child.tag, str) else ""
            if tag in SECTIONS:
                flush()
                consume(child)
            elif tag == "tr":
                flush()
                rows.append(
                    [
                        cell
                        for cell in child
                        if isinstance(cell.tag, str)
                        and str(cell.tag).lower() in CELLS
                    ]
                )
            elif tag in CELLS:
                orphan.append(child)
            else:
                flush()
        flush()

    consume(table)
    return rows


def normalized_span(cell: html.HtmlElement, name: str) -> int:
    try:
        return max(1, int(cell.get(name, "1")))
    except (TypeError, ValueError):
        return 1


def table_shape(table: html.HtmlElement) -> tuple[int, int]:
    occupied: set[tuple[int, int]] = set()
    max_row = 0
    max_column = 0
    for row_index, row in enumerate(direct_rows_and_cells(table)):
        column_index = 0
        for cell in row:
            while (row_index, column_index) in occupied:
                column_index += 1
            row_span = normalized_span(cell, "rowspan")
            column_span = normalized_span(cell, "colspan")
            for row_offset in range(row_span):
                for column_offset in range(column_span):
                    occupied.add((row_index + row_offset, column_index + column_offset))
            max_row = max(max_row, row_index + row_span)
            max_column = max(max_column, column_index + column_span)
            column_index += column_span
    return max_row, max_column


def table_rank(table: html.HtmlElement) -> tuple[int, int, int]:
    rows, columns = table_shape(table)
    return (
        rows * columns,
        sum(len(row) for row in direct_rows_and_cells(table)),
        len("".join(table.itertext()).strip()),
    )


def select_largest(tables: list[html.HtmlElement]) -> html.HtmlElement | None:
    return max(tables, key=table_rank) if tables else None


def serialize_table(table: html.HtmlElement) -> str:
    return etree.tostring(table, encoding="unicode", method="html")


def adapt_ocr_table(table: html.HtmlElement) -> html.HtmlElement:
    """Render visible formula text and safe glyph repairs without score folding."""
    output = copy.deepcopy(table)
    for cell in output.xpath(".//td|.//th"):
        source_text = "".join(cell.itertext())
        text = (
            latex_to_visible_text(source_text)
            if LATEX_SIGNAL.search(source_text)
            else source_text
        )
        cell.text = normalize_display_text(text)
        for descendant in cell.iterdescendants():
            descendant.text = None
            descendant.tail = None
    return output


def normalize_table(table: html.HtmlElement) -> html.HtmlElement:
    """Return a canonical table→tr→td tree with valid span attributes."""
    output = etree.Element("table")
    for cells in direct_rows_and_cells(table):
        row = etree.SubElement(output, "tr")
        for source_cell in cells:
            cell = etree.SubElement(row, "td")
            row_span = normalized_span(source_cell, "rowspan")
            column_span = normalized_span(source_cell, "colspan")
            if row_span > 1:
                cell.set("rowspan", str(row_span))
            if column_span > 1:
                cell.set("colspan", str(column_span))
            cell.text = "".join(source_cell.itertext()).strip()
    return output


def merge_compatible_tables(
    tables: list[html.HtmlElement],
) -> html.HtmlElement | None:
    """Merge row- or column-compatible fragments without consulting ground truth."""
    if len(tables) < 2:
        return copy.deepcopy(tables[0]) if tables else None
    shapes = [table_shape(table) for table in tables]
    same_columns = len({shape[1] for shape in shapes}) == 1 and shapes[0][1] > 0
    same_rows = len({shape[0] for shape in shapes}) == 1 and shapes[0][0] > 0
    if not same_columns and not same_rows:
        return None

    output = etree.Element("table")
    row_sets = [direct_rows_and_cells(table) for table in tables]
    if same_columns:
        for rows in row_sets:
            for cells in rows:
                row = etree.SubElement(output, "tr")
                for cell in cells:
                    row.append(copy.deepcopy(cell))
        return output

    if len({len(rows) for rows in row_sets}) != 1:
        return None
    for row_index in range(len(row_sets[0])):
        row = etree.SubElement(output, "tr")
        for rows in row_sets:
            for cell in rows[row_index]:
                row.append(copy.deepcopy(cell))
    return output


def html_metric_outputs(
    source: str,
    *,
    merge_fragments: bool = False,
) -> dict[str, str]:
    """Build the five unique artifacts consumed by the six metric stages."""
    tables = parse_top_level_tables(source)
    if not tables:
        raise ValueError("The provider response contains no HTML table.")

    raw_largest = select_largest(tables)
    assert raw_largest is not None
    adapted = [adapt_ocr_table(table) for table in tables]
    ocr_largest = select_largest(adapted)
    assert ocr_largest is not None
    ocr_merged = merge_compatible_tables(adapted) if merge_fragments else None
    if ocr_merged is None:
        ocr_merged = ocr_largest

    normalized = [normalize_table(table) for table in tables]
    html_largest = select_largest(normalized)
    assert html_largest is not None
    html_merged = merge_compatible_tables(normalized) if merge_fragments else None
    if html_merged is None:
        html_merged = html_largest

    return {
        "raw_largest": serialize_table(raw_largest),
        "ocr_largest": serialize_table(ocr_largest),
        "ocr_merged": serialize_table(ocr_merged),
        "html_largest": serialize_table(html_largest),
        "html_merged": serialize_table(html_merged),
    }
