"""Streamlit interface for the Financial RAG Research Assistant."""

from pathlib import Path
import tempfile

import streamlit as st

from src.inference import answer_question, load_qa_model
from src.preprocessing import chunk_pages, clean_pages, extract_text
from src.retrieval import RetrievalIndex


st.set_page_config(page_title="Financial RAG Research Assistant", layout="wide")
st.title("Financial RAG Research Assistant")
st.caption("Upload a financial PDF, index it locally, and ask evidence-backed questions.")

if "index" not in st.session_state:
    st.session_state.index = None
if "qa" not in st.session_state:
    st.session_state.qa = None

uploaded = st.file_uploader("Financial PDF", type=["pdf"])
if uploaded and st.button("Index document"):
    try:
        content = uploaded.getvalue()
        if not content or len(content) > 50 * 1024 * 1024:
            raise ValueError("PDF must be non-empty and at most 50 MB")
        if content[:5] != b"%PDF-":
            raise ValueError("uploaded content is not a valid PDF")
        safe_name = Path(uploaded.name or "document.pdf").name
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.pdf"
            path.write_bytes(content)
            pages = clean_pages(extract_text(path, document_id=safe_name))
            chunks = chunk_pages(pages)
            with st.spinner("Loading embeddings and building FAISS index..."):
                new_index = RetrievalIndex().build(chunks)
        st.session_state.index = new_index
        st.success(f"Indexed {len(chunks)} chunks from {safe_name}.")
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"Unable to index document: {exc}")

question = st.text_input("Question")
if st.button("Ask") and question:
    if st.session_state.index is None:
        st.error("Index a PDF before asking a question.")
    else:
        with st.spinner("Retrieving evidence and extracting an answer..."):
            try:
                if st.session_state.qa is None:
                    st.session_state.qa = load_qa_model()
                result = answer_question(question, st.session_state.index, *st.session_state.qa)
            except (ValueError, RuntimeError) as exc:
                st.error(f"Unable to answer question: {exc}")
                result = None
        if result is None:
            st.stop()
        if result["answer"]:
            st.subheader(result["answer"])
        else:
            st.warning("The indexed evidence does not support a reliable answer.")
        for source in result["sources"]:
            st.write(
                f"{source['source_filename']} — page {source['page_number']}, "
                f"chunk {source['chunk_id']} (score {source['score']:.3f})"
            )
        for evidence in result["evidence"]:
            with st.expander("Supporting evidence"):
                st.write(evidence["text"])
