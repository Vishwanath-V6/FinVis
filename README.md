# FinVis — Financial Intelligence Dashboard

FinVis is an evidence-first financial document workspace. It extracts page text
and tables from financial PDFs, preserves page and chunk provenance, indexes the
evidence with MiniLM and FAISS, and presents validated table values through a
focused Streamlit dashboard. Question answering is available as a secondary
evidence-cited workflow.

## Setup

Use Python 3.11-3.13 where possible. From the repository root:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first embedding and QA model use downloads from Hugging Face:
`sentence-transformers/all-MiniLM-L6-v2` and
`distilbert-base-uncased-distilled-squad`. Do not commit downloaded model weights.

## Command-line pipeline

```python
from src.preprocessing import extract_text, clean_pages, chunk_pages
from src.retrieval import RetrievalIndex
from src.inference import load_qa_model, answer_question

pages = clean_pages(extract_text("financial_report.pdf"))
index = RetrievalIndex().build(chunk_pages(pages))
index.save("models/saved_model")
tokenizer, model = load_qa_model()
result = answer_question("What was revenue?", index, tokenizer, model)
print(result["answer"], result["sources"])
```

Each chunk includes `document_id`, `source_filename`, one-based `page_number`,
sequential `chunk_id`, `text`, `word_count`, and `character_count`. Retrieval
results retain this metadata, so citations are tied to the exact evidence chunk.

## Interfaces

Run Streamlit:

```powershell
streamlit run app.py
```

The dashboard provides:

- **Overview** — extracted KPI cards and a validated financial profile.
- **Financial Analysis** — metric/scope filters, period comparisons, Plotly trends,
  and deterministic year-over-year calculations.
- **Segment Analysis** — segment evidence only when supported by extracted tables.
- **Evidence Explorer** — metric, period, unit, scope, page, table, and chunk
  provenance with the original searchable evidence.
- **Ask FinVis** — the existing retrieval-grounded QA flow.

The initial workflow accepts a PDF and a natural-language analysis request. It
does not fabricate missing metrics: empty or unsupported views explain why data
is unavailable. OCR requires the external Tesseract installation described below.

Run the API:

```powershell
uvicorn api:app --reload
```

The API provides `GET /health`, `POST /index` with a PDF multipart upload, and
`POST /ask` with `{"question": "...", "top_k": 5}`.

## Notebooks and evaluation

The notebooks call the shared modules rather than maintaining a second pipeline:

1. `01_data_preprocessing.ipynb` validates extraction, cleaning, and chunking.
2. `02_model_training.ipynb` demonstrates pretrained embedding/index construction;
   no embedding model is trained from scratch.
3. `03_model_evaluation.ipynb` computes retrieval and answer metrics from supplied
   cases. `data/sample_data.csv` deliberately contains unanswerable placeholders,
   not fabricated financial facts.
4. `04_inference_demo.ipynb` demonstrates indexed retrieval and citations.

Evaluation results are only meaningful when real PDFs and ground-truth cases are
provided. Scanned PDFs require an external OCR installation; the ingestion layer
fails clearly rather than silently returning fabricated text.

## Limitations

The current answerer is extractive. It returns the best valid span by model
logit ranking; raw logits are not treated as calibrated confidence. Callers that
have validation data can provide an explicit abstention policy to
`answer_question`. Tables and multi-column layouts depend on PyMuPDF's extraction
order. The API keeps one in-memory index per process; production deployments
should use a managed lifecycle and authentication.

PDF ingestion uses PyMuPDF table detection when available. Tables remain separate
structured records and are indexed as table chunks with header-to-cell
relationships. Pages with little or no text can use PyMuPDF's Tesseract OCR
bridge (`extract_text(..., enable_ocr=True)`); the Tesseract executable and
language data must be installed separately. OCR is not used to invent missing
values, and extraction quality for scans, handwriting, and complex layouts must
be validated against the source document.
