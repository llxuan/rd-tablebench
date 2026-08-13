"""Mistral OCR provider for RD-TableBench."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any


def retry_delay(error: Exception, attempt: int) -> float:
    headers = getattr(error, "headers", {}) or {}
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            delay = float(headers.get(name, 0)) * scale
            if delay > 0:
                return delay
        except (TypeError, ValueError):
            pass
    return min(2**attempt, 30)


def process(client: Any, model: str, document: dict[str, str]) -> Any:
    for attempt in range(20):
        try:
            return client.ocr.process(
                model=model,
                document=document,
                timeout_ms=int(os.environ.get("MISTRAL_TIMEOUT_MS", "180000")),
            )
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            if status_code not in {429, 500, 502, 503, 504} or attempt == 19:
                raise
            time.sleep(retry_delay(error, attempt))
    raise RuntimeError("Mistral OCR exhausted its retry budget.")


def analyze(pdf_path: Path, provider: str, model: str) -> dict[str, Any]:
    """Analyze one released PDF and return the JSON-serializable raw response."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is required.")
    if provider == "azure":
        from mistralai.azure.client import MistralAzure

        endpoint = os.environ.get("MISTRAL_API_ENDPOINT")
        if not endpoint:
            raise RuntimeError(
                "MISTRAL_API_ENDPOINT is required for Azure-hosted Mistral."
            )
        client = MistralAzure(api_key=api_key, server_url=endpoint)
    elif provider == "mistral":
        from mistralai import Mistral

        client = Mistral(api_key=api_key)
    else:
        raise ValueError(f"Unsupported Mistral provider: {provider}")

    encoded = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    response = process(
        client,
        model,
        {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{encoded}",
        },
    )
    raw = response.model_dump(mode="json")
    if not isinstance(raw, dict):
        raise TypeError("Mistral OCR returned a non-object response.")
    return raw
