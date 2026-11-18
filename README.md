# FinVis

FinVis is a local, evidence-first financial document intelligence application.
It extracts financial tables and narrative text from annual and quarterly PDFs,
preserves source provenance, calculates supported comparisons and ratios, builds
a reusable MiniLM/FAISS retrieval index, and answers questions with citations.

## Install

Use Python 3.11-3.13 on Windows:

```powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

Optional OCR: install Tesseract and add it to `PATH`. FinVis avoids OCR for
pages that already contain sufficient native text.

## Run

Streamlit application:

```powershell
streamlit run app.py
```

FastAPI service:

```powershell
uvicorn api:app --reload
```

API endpoints:

- `GET /health`
- `POST /index` multipart PDF
- `POST /ask` JSON `{"question":"What was revenue?","top_k":5}`

## Tests

```powershell
python -m pytest -q
```

## Document workflow

The pipeline is:

`PDF validation -> SHA-256 document ID -> cache lookup -> fast page scan ->
targeted native extraction -> selective OCR -> table/header repair -> cleaning ->
row-specific chunks -> structured records -> lexical/FAISS retrieval ->
deterministic calculations -> dashboard -> cited Q&A`

Large PDFs are processed page by page. The request narrows full-fidelity page
selection, adjacent context pages are retained, and repeated document/request
processing reuses the `.finvis_cache` cache.

## Dataset and source description

FinVis accepts user-owned financial PDF reports. No paid dataset or cloud API is
required. `data/sample_data.csv` is a placeholder and contains no fabricated
financial ground truth. `benchmark_outputs/` contains benchmark artifacts only
when a supplied PDF and real question dataset have been evaluated.

Create evaluation cases with question text, answerability, reference answers,
and expected evidence chunk IDs. Do not claim benchmark results without real
ground truth.

## Dashboard

- **Overview:** available KPI cards and financial profile.
- **Financial Analysis:** metric/scope filters, Plotly trends, YoY changes,
  and profit margins.
- **Segment Analysis:** segment evidence only when present.
- **Evidence Explorer:** raw evidence with page, table, chunk, scope, unit,
  and reporting period metadata.
- **Ask FinVis:** table-grounded/direct answers or extractive QA with citations.

Missing values are not invented. Standalone and consolidated rows are kept
separate. Units must match before period calculations.

## Evaluation and benchmarks

Run reusable ingestion/cache benchmark:

```powershell
python scripts/benchmark_ingestion.py --pdf report.pdf
```

Run before/after page-selection comparison:

```powershell
python scripts/benchmark_performance.py --pdf report.pdf
```

Run a ground-truth retrieval/QA benchmark:

```powershell
python scripts/run_benchmark.py --pdf report.pdf --input benchmark_inputs.json
```

The evaluation reports Hit@K, MRR, answer correctness, citation accuracy,
abstention accuracy, and latency. Complex layouts, extraction quality, and
scanned pages need human/source-backed error analysis.

## Architecture documentation

- [Problem statement](docs/PROBLEM_STATEMENT.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Presentation outline](docs/PRESENTATION_OUTLINE.md)
- [Demo script](docs/DEMO_SCRIPT.md)
- [Implementation status](IMPLEMENTATION_STATUS.md)

## Limitations

- PyMuPDF table extraction is layout dependent.
- OCR quality depends on Tesseract and source image quality.
- The pretrained extractive QA model is not a calibrated confidence model.
- The API uses an in-memory index per process.
- A 500-page PDF is supported through page selection, but actual latency and OCR
  usage depend on report layout and hardware.
