"""Azure Content Understanding provider for RD-TableBench."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from providers.errors import ProviderRequestError


def analyze(input_path: Path, analyzer_id: str) -> dict[str, Any]:
    """Analyze one PDF and return the raw Azure response as a JSON object."""
    endpoint = os.environ.get("AZURE_CONTENT_UNDERSTANDING_ENDPOINT")
    key = os.environ.get("AZURE_CONTENT_UNDERSTANDING_KEY")
    if not endpoint or not key:
        raise RuntimeError(
            "AZURE_CONTENT_UNDERSTANDING_ENDPOINT and "
            "AZURE_CONTENT_UNDERSTANDING_KEY are required."
        )

    from azure.ai.contentunderstanding import ContentUnderstandingClient
    from azure.core.credentials import AzureKeyCredential

    client = ContentUnderstandingClient(endpoint, AzureKeyCredential(key))
    binary_input = input_path.read_bytes()
    try:
        result = client.begin_analyze_binary(
            analyzer_id=analyzer_id,
            binary_input=binary_input,
            content_type="application/pdf",
        ).result()
    except Exception as error:
        raise ProviderRequestError("Azure Content Understanding request failed.") from error
    raw = result.as_dict()
    if not isinstance(raw, dict):
        raise TypeError("Azure Content Understanding returned a non-object response.")
    return raw
