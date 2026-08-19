from __future__ import annotations

import base64
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

UPSTREAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UPSTREAM_ROOT))

from providers.mistral_ocr import analyze  # noqa: E402
from providers.errors import ProviderRequestError  # noqa: E402


class RateLimitError(RuntimeError):
    status_code = 429
    headers = {"retry-after-ms": "5000"}


class MistralOcrTests(unittest.TestCase):
    @staticmethod
    def _sdk_modules(client_factory: Mock) -> dict[str, types.ModuleType]:
        mistralai = types.ModuleType("mistralai")
        mistralai.__path__ = []
        azure = types.ModuleType("mistralai.azure")
        azure.__path__ = []
        client_module = types.ModuleType("mistralai.azure.client")
        setattr(client_module, "MistralAzure", client_factory)
        return {
            "mistralai": mistralai,
            "mistralai.azure": azure,
            "mistralai.azure.client": client_module,
        }

    @staticmethod
    def _environment() -> dict[str, str]:
        return {
            "MISTRAL_API_ENDPOINT": "https://example.test",
            "MISTRAL_API_KEY": "secret",
        }

    def test_analyze_requests_html_tables_and_returns_raw_response(self):
        raw = {
            "pages": [
                {
                    "tables": [
                        {
                            "format": "html",
                            "content": "<table><tr><td>A</td></tr></table>",
                        }
                    ]
                }
            ]
        }
        response = Mock()
        response.model_dump.return_value = raw
        client = Mock()
        client.ocr.process.return_value = response
        client_factory = Mock(return_value=client)

        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "case.pdf"
            pdf.write_bytes(b"pdf bytes")
            with (
                patch.dict(os.environ, self._environment(), clear=True),
                patch.dict(sys.modules, self._sdk_modules(client_factory)),
            ):
                actual = analyze(pdf, "mistral-ocr-4-0")

        self.assertIs(actual, raw)
        client_factory.assert_called_once_with(
            api_key="secret",
            server_url="https://example.test",
        )
        options = client.ocr.process.call_args.kwargs
        self.assertEqual(options["model"], "mistral-ocr-4-0")
        self.assertEqual(options["table_format"], "html")
        self.assertEqual(options["timeout_ms"], 180000)
        self.assertEqual(options["document"]["type"], "document_url")
        encoded = options["document"]["document_url"].split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), b"pdf bytes")
        self.assertEqual(
            actual["pages"][0]["tables"][0]["content"],
            "<table><tr><td>A</td></tr></table>",
        )
        response.model_dump.assert_called_once_with(mode="json")

    def test_analyze_retries_with_service_delay(self):
        response = Mock()
        response.model_dump.return_value = {"pages": []}
        client = Mock()
        client.ocr.process.side_effect = [RateLimitError(), response]
        client_factory = Mock(return_value=client)

        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "case.pdf"
            pdf.write_bytes(b"pdf bytes")
            with (
                patch.dict(os.environ, self._environment(), clear=True),
                patch.dict(sys.modules, self._sdk_modules(client_factory)),
                patch("providers.mistral_ocr.time.sleep") as sleep,
            ):
                analyze(pdf, "mistral-ocr-4-0")

        self.assertEqual(client.ocr.process.call_count, 2)
        sleep.assert_called_once_with(5.0)

    def test_analyze_requires_endpoint_and_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "MISTRAL_API_ENDPOINT.*MISTRAL_API_KEY",
            ):
                analyze(Path("case.pdf"), "mistral-ocr-4-0")

    def test_analyze_rejects_non_object_response(self):
        response = Mock()
        response.model_dump.return_value = []
        client = Mock()
        client.ocr.process.return_value = response
        client_factory = Mock(return_value=client)

        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "case.pdf"
            pdf.write_bytes(b"pdf bytes")
            with (
                patch.dict(os.environ, self._environment(), clear=True),
                patch.dict(sys.modules, self._sdk_modules(client_factory)),
            ):
                with self.assertRaisesRegex(TypeError, "non-object response"):
                    analyze(pdf, "mistral-ocr-4-0")

    def test_analyze_wraps_remote_request_failure(self):
        error = RuntimeError("service failed")
        error.status_code = 400  # type: ignore[attr-defined]
        client = Mock()
        client.ocr.process.side_effect = error
        client_factory = Mock(return_value=client)

        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "case.pdf"
            pdf.write_bytes(b"pdf bytes")
            with (
                patch.dict(os.environ, self._environment(), clear=True),
                patch.dict(sys.modules, self._sdk_modules(client_factory)),
            ):
                with self.assertRaisesRegex(
                    ProviderRequestError,
                    "Mistral OCR request failed",
                ):
                    analyze(pdf, "mistral-ocr-4-0")

    def test_analyze_does_not_wrap_local_input_failure(self):
        client = Mock()
        client_factory = Mock(return_value=client)

        with (
            patch.dict(os.environ, self._environment(), clear=True),
            patch.dict(sys.modules, self._sdk_modules(client_factory)),
        ):
            with self.assertRaises(FileNotFoundError):
                analyze(Path("missing.pdf"), "mistral-ocr-4-0")

        client.ocr.process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
