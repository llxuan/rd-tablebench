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


def process(
    client: Any,
    model: str,
    document: dict[str, str],
    table_format: str | None = None,
) -> Any:
    for attempt in range(20):
        try:
            options: dict[str, Any] = {
                "model": model,
                "document": document,
                "timeout_ms": int(os.environ.get("MISTRAL_TIMEOUT_MS", "180000")),
            }
            if table_format is not None:
                options["table_format"] = table_format
            return client.ocr.process(
                **options,
            )
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            if status_code not in {429, 500, 502, 503, 504} or attempt == 19:
                raise
            time.sleep(retry_delay(error, attempt))
    raise RuntimeError("Mistral OCR exhausted its retry budget.")


def input_document(
    input_path: Path,
    input_mode: str,
    media_type: str | None = None,
) -> dict[str, str]:
    encoded = base64.b64encode(input_path.read_bytes()).decode("ascii")
    if media_type is None:
        if input_mode == "pdf":
            media_type = "application/pdf"
        elif input_mode == "image":
            media_type = "image/jpeg"
        else:
            raise ValueError(f"Unsupported RD-TableBench input mode: {input_mode}")
    return {
        "type": "document_url",
        "document_url": f"data:{media_type};base64,{encoded}",
    }


def analyze(
    input_path: Path,
    provider: str,
    model: str,
    input_mode: str,
    table_format: str,
    media_type: str | None = None,
) -> dict[str, Any]:
    """Analyze one released PDF or JPG and return the raw response."""
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

    response = process(
        client,
        model,
        input_document(input_path, input_mode, media_type),
        None if table_format == "inline" else table_format,
    )
    raw = response.model_dump(mode="json")
    if not isinstance(raw, dict):
        raise TypeError("Mistral OCR returned a non-object response.")
    return raw
