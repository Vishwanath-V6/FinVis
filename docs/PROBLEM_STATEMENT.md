# FinVis Problem Statement

Financial reports are long, layout-heavy PDFs. Readers need reliable access to
values, reporting periods, units, accounting scope, and the original source
passage or table. Manual search is slow and copy/paste often loses sign, unit,
or standalone/consolidated context.

FinVis is a local financial document intelligence application that:

1. validates and fingerprints uploaded PDFs;
2. extracts narrative text and page-level tables;
3. selectively OCRs scanned pages;
4. normalizes only supported financial values and periods;
5. preserves table, page, and chunk evidence;
6. builds a reusable lexical/FAISS retrieval index;
7. calculates comparisons and ratios deterministically; and
8. answers questions with source citations or explicit abstention.

The system does not use an LLM for arithmetic and does not fabricate missing
scope, units, or periods.
