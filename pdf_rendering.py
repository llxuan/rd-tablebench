"""Deterministic single-page PDF rendering for provider input."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pymupdf

RENDERER_NAME = "PyMuPDF"


def renderer_version() -> str:
    return str(pymupdf.VersionBind)


def validate_single_page_pdf(path: Path) -> None:
    with pymupdf.open(path) as document:
        if document.page_count != 1:
            raise ValueError(
                f"RD-TableBench rendered PDF input requires exactly one page: "
                f"{path} has {document.page_count}."
            )


def render_single_page_png(source: Path, destination: Path, dpi: int) -> dict[str, object]:
    if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi <= 0:
        raise ValueError("PDF render DPI must be a positive integer.")
    with pymupdf.open(source) as document:
        if document.page_count != 1:
            raise ValueError(
                f"RD-TableBench rendered PDF input requires exactly one page: "
                f"{source} has {document.page_count}."
            )
        pixmap = document[0].get_pixmap(
            dpi=dpi,
            colorspace=pymupdf.csRGB,
            alpha=False,
        )
        content = pixmap.tobytes("png")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "renderer": RENDERER_NAME,
        "renderer_version": renderer_version(),
        "dpi": dpi,
        "media_type": "image/png",
        "width": pixmap.width,
        "height": pixmap.height,
    }
