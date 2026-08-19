"""Azure-hosted Mistral OCR provider for RD-TableBench."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

from providers.errors import ProviderRequestError

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 20


def _retry_delay(error: Exception, attempt: int) -> float:
    headers = getattr(error, "headers", {}) or {}
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            delay = float(headers.get(name, 0)) * scale
            if delay > 0:
                return delay
        except (TypeError, ValueError):
            pass
    return min(2**attempt, 30)


def _input_document(input_path: Path) -> dict[str, str]:
    encoded = base64.b64encode(input_path.read_bytes()).decode("ascii")
    return {
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{encoded}",
    }


def _process(
    client: Any,
    model: str,
    document: dict[str, str],
    timeout_ms: int,
) -> Any:
    for attempt in range(MAX_ATTEMPTS):
        try:
            return client.ocr.process(
                model=model,
                document=document,
                table_format="html",
                timeout_ms=timeout_ms,
            )
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            if status_code not in RETRYABLE_STATUS_CODES or attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(_retry_delay(error, attempt))
    raise RuntimeError("Mistral OCR exhausted its retry budget.")


def analyze(input_path: Path, model: str) -> dict[str, Any]:
    """Analyze one PDF with HTML table output and return the raw JSON object."""
    endpoint = os.environ.get("MISTRAL_API_ENDPOINT")
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("MISTRAL_API_ENDPOINT and MISTRAL_API_KEY are required.")

    from mistralai.azure.client import MistralAzure

    client = MistralAzure(api_key=api_key, server_url=endpoint)
    document = _input_document(input_path)
    timeout_ms = int(os.environ.get("MISTRAL_TIMEOUT_MS", "180000"))
    try:
        response = _process(client, model, document, timeout_ms)
    except Exception as error:
        raise ProviderRequestError("Mistral OCR request failed.") from error
    raw = response.model_dump(mode="json")
    if not isinstance(raw, dict):
        raise TypeError("Mistral OCR returned a non-object response.")
    return raw
