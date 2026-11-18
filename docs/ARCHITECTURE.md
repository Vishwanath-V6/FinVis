# FinVis Architecture

## Pipeline

`PDF upload -> validation -> SHA-256 fingerprint -> targeted page scan ->
native text/table extraction -> selective OCR -> cleaning -> row-based chunks ->
structured records -> lexical/FAISS retrieval -> deterministic calculations ->
dashboard/evidence explorer -> grounded Q&A`

## Modules

- `src/ingestion.py`: PDF fingerprinting, document cache, page selection,
  staged extraction, cache reuse.
- `src/preprocessing.py`: native text extraction, table detection, header
  reconstruction, selective OCR, cleaning, and row-specific chunking.
- `src/dashboard.py`: conservative financial parsing, record construction,
  year-over-year changes, and profit ratios.
- `src/retrieval.py`: MiniLM vector embeddings, FAISS, metadata, lexical
  fallback, table-aware ranking, and persistent index loading.
- `src/model_utils.py`: reusable CPU embedding model, batching, normalization,
  and thread configuration.
- `src/inference.py`: table evidence resolution, extractive QA, citations,
  and abstention.
- `src/evaluation.py`: retrieval metrics and answer grounding metrics.
- `app.py`: Streamlit dashboard and evidence workflow.
- `api.py`: optional FastAPI upload/query service.

## Grounding rules

Every structured record carries metric, raw value, value, unit, period, scope,
page, table ID, chunk ID, source filename, and evidence text. Standalone and
consolidated scopes are kept separate. Calculations require matching metric,
scope, and unit and use a zero-denominator guard.
