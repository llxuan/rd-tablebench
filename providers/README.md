# Table Extraction Providers

This directory contains the invocation scripts used to run each provider through the [RD-TableBench](https://reducto.ai/blog/rd-tablebench) evaluation. Each script handles PDF ingestion, API calls, and response storage. Responses are parsed by `parsing.py` and scored by `grading.py` in the root of the repository.

## Reducto

**File:** `reducto.py`
**API:** [Reducto Document Parsing API](https://reducto.ai)
**Concurrency:** Async with 10 concurrent uploads and 10 concurrent parse jobs
**Auth:** `REDUCTO_API_KEY` environment variable

[Reducto](https://reducto.ai) is a hybrid document parsing API that combines OCR and layout analysis to extract structured data from complex documents. Its approach to table extraction handles challenging scenarios — scanned documents, handwritten content, merged cells, and multilingual text — by fusing multiple extraction signals rather than relying on a single method.

The invocation script uses a three-step async workflow: upload the PDF, submit an async parse job with combined OCR mode, and poll for completion.

```bash
export REDUCTO_API_KEY="your-key"
python providers/reducto.py
```

**Dependencies:** `aiohttp`, `tqdm`

To get a Reducto API key, sign up at [reducto.ai](https://reducto.ai).

## Baseline Providers

The following providers are included as baselines for comparison. Each script invokes the provider's API and stores the raw response for parsing and grading.

| Provider | File | Auth Environment Variables |
|---|---|---|
| Azure Document Intelligence | `azure_docintelligence.py` | `AZURE_ENDPOINT`, `AZURE_KEY` |
| AWS Textract | `textract.py` | AWS credentials (boto3 default chain) |
| GPT-4o | `gpt4o.py` | `OPENAI_API_KEY` |
| Google Cloud Document AI | `gcloud.py` | `GCP_PROJECT_ID`, `GCP_PROCESSOR_ID` |
| Unstructured | `unstructured.py` | `UNSTRUCTURED_API_KEY` |
| Chunkr | `chunkr.py` | `CHUNKR_API_KEY` |
| Azure Content Understanding | `azure_content_understanding.py` | `AZURE_CONTENT_UNDERSTANDING_ENDPOINT`, `AZURE_CONTENT_UNDERSTANDING_KEY` |
| Mistral OCR | `mistral_ocr.py` | `MISTRAL_API_KEY`; Azure-hosted deployments also require `MISTRAL_API_ENDPOINT` |

Each baseline script follows a similar pattern: read PDFs from the dataset directory, call the provider API with rate-limit handling, and write JSON responses to a provider-specific output directory. See individual files for provider-specific configuration details.

## Adding a New Provider

To add support for a new table extraction service:

1. Create a new Python file in this directory (e.g., `your_provider.py`)
2. Implement a function that processes a PDF and saves the API response as JSON
3. Add a corresponding parser function in `parsing.py` that extracts the HTML table from your provider's response format
4. Run the provider against the benchmark dataset and grade the results using `grading.py`

See any existing provider script for the expected pattern.

## Resumable benchmark CLI

`benchmark_cli.py` provides a stable command that runs Azure Content
Understanding or Mistral OCR over any complete RD-TableBench dataset directory,
parses one HTML table per case, invokes the native `table_similarity` evaluator,
and writes machine-readable native artifacts:

```text
python benchmark_cli.py run \
	--dataset-root /path/to/rd-tablebench \
	--output-root /path/to/run \
	--provider azure-cu \
	--analyzer-id prebuilt-layout \
	--parallel 4
```

For Azure-hosted Mistral OCR, use `--provider mistral`,
`--mistral-provider azure`, and `--mistral-model mistral-ocr-4-0`.

The benchmark CLI sends each released padded PDF as `application/pdf`. Mistral
uses the OCR API's `document_url` variant. For structured table output, add
`--mistral-table-format html`; the parser consumes ordered
`pages[].tables[].content` entries because page Markdown contains only table
placeholders in this mode. Table format is included in the resume configuration
identity, so incompatible artifacts cannot be reused.

The output root contains `manifest.json`, per-case `raw/`, `outputs/`, and
`status/` directories, plus `evaluation/results.jsonl`, `failures.jsonl`, and
`summary.json`. Provider and evaluation failures remain distinct and contribute
zero to the fixed-denominator aggregate without being represented as scored
zero cases. Matching successful cases resume only when input, configuration,
raw response, and output hashes still agree.
