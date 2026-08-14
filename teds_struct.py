"""PubTabNet-style APTED TEDS-Struct evaluation."""

from __future__ import annotations

from collections import deque

from apted import APTED, Config
from apted.helpers import Tree
from Levenshtein import distance as levenshtein_distance
from lxml import html


class TableTree(Tree):
    def __init__(
        self,
        tag: str,
        column_span: int | None = None,
        row_span: int | None = None,
        content: str | None = None,
        *children: Tree,
    ) -> None:
        self.tag = tag
        self.colspan = column_span
        self.rowspan = row_span
        self.content = content
        self.children = list(children)


class TableTreeConfig(Config):
    @staticmethod
    def normalized_distance(left: str, right: str) -> float:
        maximum = max(len(left), len(right))
        return float(levenshtein_distance(left, right)) / maximum if maximum else 0.0

    def rename(self, node1: TableTree, node2: TableTree) -> float:
        if (
            node1.tag != node2.tag
            or node1.colspan != node2.colspan
            or node1.rowspan != node2.rowspan
        ):
            return 1.0
        if node1.tag in {"td", "th"} and (node1.content or node2.content):
            return self.normalized_distance(node1.content or "", node2.content or "")
        return 0.0


class TEDS:
    def __init__(self, structure_only: bool = False, keep_th: bool = False) -> None:
        self.structure_only = structure_only
        self.keep_th = keep_th

    def _load_tree(
        self,
        node: html.HtmlElement,
        parent: TableTree | None = None,
    ) -> TableTree:
        if node.tag in {"td", "th"}:
            content = "" if self.structure_only else "".join(node.itertext()).strip()
            current = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                content,
                *deque(),
            )
        else:
            current = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(current)
        if node.tag not in {"td", "th"}:
            for child in node.getchildren():
                self._load_tree(child, current)
        return current

    @staticmethod
    def _table(document: html.HtmlElement) -> html.HtmlElement | None:
        if document.tag == "table":
            return document
        tables = document.xpath(".//table")
        return tables[0] if tables else None

    def evaluate(self, prediction: str, ground_truth: str) -> float:
        if not prediction or not ground_truth:
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        prediction_document = html.fromstring(prediction, parser=parser)
        truth_document = html.fromstring(ground_truth, parser=parser)
        if not self.keep_th:
            for document in (prediction_document, truth_document):
                for header in document.xpath(".//th"):
                    header.tag = "td"

        prediction_table = self._table(prediction_document)
        truth_table = self._table(truth_document)
        if prediction_table is None or truth_table is None:
            return 0.0
        if _node_signature(prediction_table) == _node_signature(truth_table):
            return 1.0
        node_count = max(
            len(prediction_table.xpath(".//*")),
            len(truth_table.xpath(".//*")),
        )
        if node_count == 0:
            return 1.0
        distance = APTED(
            self._load_tree(prediction_table),
            self._load_tree(truth_table),
            TableTreeConfig(),
        ).compute_edit_distance()
        return 1.0 - float(distance) / node_count


_TEDS_STRUCT = TEDS(structure_only=True)


def _node_signature(node: html.HtmlElement) -> tuple:
    tag = "td" if node.tag == "th" else node.tag
    if tag == "td":
        return (
            tag,
            int(node.attrib.get("colspan", "1")),
            int(node.attrib.get("rowspan", "1")),
        )
    return (tag, tuple(_node_signature(child) for child in node.getchildren()))


def structure_signature(source: str) -> tuple:
    """Return the exact structure consumed by structure-only TEDS."""
    if not source:
        return ()
    document = html.fromstring(
        source,
        parser=html.HTMLParser(remove_comments=True, encoding="utf-8"),
    )
    table = TEDS._table(document)
    return _node_signature(table) if table is not None else ()


def teds_struct_score(ground_truth: str, prediction: str) -> float:
    """Return standard structure-only TEDS in the inclusive range [0, 1]."""
    return max(0.0, min(1.0, _TEDS_STRUCT.evaluate(prediction, ground_truth)))
