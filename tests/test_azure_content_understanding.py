from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

UPSTREAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UPSTREAM_ROOT))

from providers.azure_content_understanding import analyze  # noqa: E402
from providers.errors import ProviderRequestError  # noqa: E402


class AzureContentUnderstandingTests(unittest.TestCase):
    @staticmethod
    def _sdk_modules(
        client_factory: Mock,
        credential_factory: Mock,
    ) -> dict[str, types.ModuleType]:
        azure = types.ModuleType("azure")
        azure.__path__ = []
        azure_ai = types.ModuleType("azure.ai")
        azure_ai.__path__ = []
        content_understanding = types.ModuleType("azure.ai.contentunderstanding")
        setattr(content_understanding, "ContentUnderstandingClient", client_factory)
        azure_core = types.ModuleType("azure.core")
        azure_core.__path__ = []
        credentials = types.ModuleType("azure.core.credentials")
        setattr(credentials, "AzureKeyCredential", credential_factory)
        return {
            "azure": azure,
            "azure.ai": azure_ai,
            "azure.ai.contentunderstanding": content_understanding,
            "azure.core": azure_core,
            "azure.core.credentials": credentials,
        }

    def test_analyze_sends_pdf_and_returns_raw_response(self):
        raw = {"status": "Succeeded", "contents": []}
        sdk_result = Mock()
        sdk_result.as_dict.return_value = raw
        poller = Mock()
        poller.result.return_value = sdk_result
        client = Mock()
        client.begin_analyze_binary.return_value = poller
        client_factory = Mock(return_value=client)
        credential = object()
        credential_factory = Mock(return_value=credential)

        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "case.pdf"
            pdf.write_bytes(b"pdf bytes")
            with (
                patch.dict(
                    os.environ,
                    {
                        "AZURE_CONTENT_UNDERSTANDING_ENDPOINT": "https://example.test",
                        "AZURE_CONTENT_UNDERSTANDING_KEY": "secret",
                    },
                    clear=True,
                ),
                patch.dict(
                    sys.modules,
                    self._sdk_modules(client_factory, credential_factory),
                ),
            ):
                actual = analyze(pdf, "prebuilt-layout")

        self.assertIs(actual, raw)
        credential_factory.assert_called_once_with("secret")
        client_factory.assert_called_once_with("https://example.test", credential)
        client.begin_analyze_binary.assert_called_once_with(
            analyzer_id="prebuilt-layout",
            binary_input=b"pdf bytes",
            content_type="application/pdf",
        )
        poller.result.assert_called_once_with()
        sdk_result.as_dict.assert_called_once_with()

    def test_analyze_requires_endpoint_and_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "AZURE_CONTENT_UNDERSTANDING_ENDPOINT.*AZURE_CONTENT_UNDERSTANDING_KEY",
            ):
                analyze(Path("case.pdf"), "prebuilt-layout")

    def test_analyze_rejects_non_object_response(self):
        sdk_result = Mock()
        sdk_result.as_dict.return_value = []
        poller = Mock()
        poller.result.return_value = sdk_result
        client = Mock()
        client.begin_analyze_binary.return_value = poller
        client_factory = Mock(return_value=client)

        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "case.pdf"
            pdf.write_bytes(b"pdf bytes")
            with (
                patch.dict(
                    os.environ,
                    {
                        "AZURE_CONTENT_UNDERSTANDING_ENDPOINT": "https://example.test",
                        "AZURE_CONTENT_UNDERSTANDING_KEY": "secret",
                    },
                    clear=True,
                ),
                patch.dict(
                    sys.modules,
                    self._sdk_modules(client_factory, Mock()),
                ),
            ):
                with self.assertRaisesRegex(TypeError, "non-object response"):
                    analyze(pdf, "prebuilt-layout")

    def test_analyze_wraps_remote_request_failure(self):
        client = Mock()
        client.begin_analyze_binary.side_effect = RuntimeError("service failed")
        client_factory = Mock(return_value=client)

        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "case.pdf"
            pdf.write_bytes(b"pdf bytes")
            with (
                patch.dict(
                    os.environ,
                    {
                        "AZURE_CONTENT_UNDERSTANDING_ENDPOINT": "https://example.test",
                        "AZURE_CONTENT_UNDERSTANDING_KEY": "secret",
                    },
                    clear=True,
                ),
                patch.dict(
                    sys.modules,
                    self._sdk_modules(client_factory, Mock()),
                ),
            ):
                with self.assertRaisesRegex(
                    ProviderRequestError,
                    "Azure Content Understanding request failed",
                ):
                    analyze(pdf, "prebuilt-layout")


if __name__ == "__main__":
    unittest.main()