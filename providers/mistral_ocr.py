"""Mistral OCR provider for RD-TableBench."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any


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
    response = None
    for attempt in range(5):
        try:
            response = client.ocr.process(
                model=model,
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{encoded}",
                },
            )
            break
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            if status_code not in {429, 500, 502, 503, 504} or attempt == 4:
                raise
            headers = getattr(error, "headers", {})
            try:
                delay = float(headers.get("retry-after", 0))
            except (TypeError, ValueError):
                delay = 0
            time.sleep(max(delay, min(2**attempt, 60)))
    if response is None:
        raise RuntimeError("Mistral OCR did not return a response.")
    raw = response.model_dump(mode="json")
    if not isinstance(raw, dict):
        raise TypeError("Mistral OCR returned a non-object response.")
    return raw
