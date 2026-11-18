# Implementation Status

## Verified

- 37-test suite passes.
- Streamlit 1.66-compatible app structure.
- PyMuPDF native text and table extraction.
- Selective OCR bridge.
- SHA-256 document cache and staged page selection.
- Row-specific table chunks with page/table/chunk provenance.
- Lexical and FAISS retrieval.
- Deterministic YoY and profit-margin calculations.
- Grounded table QA and explicit missing-evidence abstention.
- FastAPI upload/query interface.
- Reproducible benchmark scripts and evaluation helpers.

## External verification still requires user-supplied documents

- Complex 500-page PDFs should be benchmarked against the target hardware.
- OCR quality requires Tesseract and language data installed by the deployment
  host.
- Retrieval/QA accuracy requires real ground-truth cases and a target PDF.
- The QA model may download pretrained weights on first use.
