"""Azure Content Understanding provider for RD-TableBench."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def analyze(pdf_path: Path, analyzer_id: str) -> dict[str, Any]:
    """Analyze one released PDF and return the JSON-serializable raw response."""
    from azure.ai.contentunderstanding import ContentUnderstandingClient
    from azure.core.credentials import AzureKeyCredential

    endpoint = os.environ.get("AZURE_CONTENT_UNDERSTANDING_ENDPOINT")
    key = os.environ.get("AZURE_CONTENT_UNDERSTANDING_KEY")
    if not endpoint or not key:
        raise RuntimeError(
            "AZURE_CONTENT_UNDERSTANDING_ENDPOINT and "
            "AZURE_CONTENT_UNDERSTANDING_KEY are required."
        )
    client = ContentUnderstandingClient(endpoint, AzureKeyCredential(key))
    result = client.begin_analyze_binary(
        analyzer_id=analyzer_id,
        binary_input=pdf_path.read_bytes(),
    ).result()
    raw = result.as_dict()
    if not isinstance(raw, dict):
        raise TypeError("Azure Content Understanding returned a non-object response.")
    return raw
