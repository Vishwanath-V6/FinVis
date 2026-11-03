"""FinVis Streamlit dashboard for grounded financial-document analysis."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard import add_families, calculate_yoy, financial_records
from src.inference import answer_question, load_qa_model
from src.preprocessing import PreprocessingError, chunk_pages, clean_pages, extract_text
from src.retrieval import RetrievalIndex


MAX_UPLOAD_BYTES = 50 * 1024 * 1024

st.set_page_config(page_title="FinVis", page_icon="F", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root { --ink: #172033; --muted: #657084; --line: #e5e8ee; --paper: #fbfcfe; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #f3f5f8; border-right: 1px solid var(--line); }
    .block-container { max-width: 1440px; padding-top: 2rem; }
    h1, h2, h3 { letter-spacing: -0.025em; color: var(--ink); }
    .eyebrow { color: #64748b; font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .metric-card { background: white; border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.1rem; min-height: 108px; }
    .metric-label { color: var(--muted); font-size: .8rem; margin-bottom: .4rem; }
    .metric-value { color: var(--ink); font-size: 1.45rem; font-weight: 700; font-variant-numeric: tabular-nums; }
    .metric-note { color: var(--muted); font-size: .75rem; margin-top: .35rem; }
    .status { border-left: 3px solid #6b8f71; background: #f1f7f2; padding: .8rem 1rem; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    defaults = {
        "index": None,
        "chunks": [],
        "pages": [],
        "records": pd.DataFrame(),
        "qa": None,
        "document_name": None,
        "analysis_request": "",
        "status": "Upload a report to begin.",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _format_value(value: float, unit: str) -> str:
    return f"{value:,.2f} {unit}".strip()


def _metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def _index_upload(uploaded_file: Any, request: str) -> None:
    content = uploaded_file.getvalue()
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("PDF must be non-empty and at most 50 MB.")
    if content[:5] != b"%PDF-":
        raise ValueError("The uploaded file is not a valid PDF.")
    safe_name = Path(uploaded_file.name or "financial-report.pdf").name
    progress = st.status("Preparing your report...", expanded=True)
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.pdf"
            path.write_bytes(content)
            progress.write("Reading pages and extracting tables...")
            pages = clean_pages(extract_text(path, document_id=safe_name, enable_ocr=True))
            progress.write(f"Extracted {len(pages):,} pages. Creating evidence chunks...")
            chunks = chunk_pages(pages)
            progress.write(f"Created {len(chunks):,} chunks. Loading embeddings and building the index...")
            index = RetrievalIndex().build(chunks)
        progress.update(
            label=f"Report ready: {len(chunks):,} evidence chunks indexed.",
            state="complete",
            expanded=False,
        )
    except Exception:
        progress.update(label="Report processing failed.", state="error", expanded=True)
        raise
    st.session_state.update(
        index=index,
        pages=pages,
        chunks=chunks,
        records=add_families(financial_records(chunks)),
        document_name=safe_name,
        analysis_request=request.strip(),
        status=f"Ready: {len(chunks):,} evidence chunks indexed.",
        qa=None,
    )


def _sidebar() -> None:
    with st.sidebar:
        st.markdown("## FinVis")
        st.caption("Evidence-first financial intelligence")
        uploaded = st.file_uploader("Upload annual report", type=["pdf"])
        request = st.text_area(
            "What should FinVis analyse?",
            value=st.session_state.analysis_request,
            placeholder="Revenue, operating profit and debt over the last three years...",
            height=100,
        )
        workflow = st.selectbox("Workflow", ["Summary dashboard", "Targeted extraction"])
        st.caption("After clicking Analyse report, processing may take several minutes for a large annual report.")
        if st.button("Analyse report", type="primary", use_container_width=True):
            if uploaded is None:
                st.error("Choose a PDF before starting analysis.")
            elif workflow == "Targeted extraction" and not request.strip():
                st.error("Describe the metrics you want extracted.")
            else:
                try:
                    _index_upload(uploaded, request)
                    st.rerun()
                except (OSError, ValueError, RuntimeError, PreprocessingError) as exc:
                    st.session_state.status = "Analysis failed."
                    st.error(f"Unable to analyse the report: {exc}")
        st.divider()
        st.markdown(f"**Status**  \n{st.session_state.status}")
        if st.session_state.document_name:
            st.caption(st.session_state.document_name)


def _overview(records: pd.DataFrame) -> None:
    st.markdown('<div class="eyebrow">Overview</div>', unsafe_allow_html=True)
    st.title("Financial intelligence, grounded in the report")
    st.caption("Only values extracted from validated table evidence are shown.")
    if records.empty:
        st.info("No structured financial table values were extracted. Open Evidence Explorer to inspect text evidence.")
        return
    metrics = records["Metric"].drop_duplicates().tolist()
    preferred = [
        next((m for m in metrics if "revenue from operations" in m.casefold()), None),
        next((m for m in metrics if "operating profit" in m.casefold()), None),
        next((m for m in metrics if "profit after tax" in m.casefold() or "net profit" in m.casefold()), None),
        next((m for m in metrics if "debt" in m.casefold()), None),
    ]
    cards = [m for m in preferred if m]
    if not cards:
        cards = metrics[:4]
    columns = st.columns(min(4, len(cards)))
    for column, metric in zip(columns, cards):
        latest = records[records["Metric"] == metric].iloc[-1]
        with column:
            _metric_card(metric, _format_value(latest["Value"], latest["Unit"]), f"{latest['Scope']} · {latest['Year']}")
    st.divider()
    st.subheader("Extracted financial profile")
    summary = records.groupby(["Metric", "Scope"], as_index=False)["Value"].count().rename(columns={"Value": "Evidence points"})
    st.dataframe(summary, use_container_width=True, hide_index=True)


def _analysis(records: pd.DataFrame) -> None:
    st.markdown('<div class="eyebrow">Financial analysis</div>', unsafe_allow_html=True)
    st.title("Compare validated figures")
    if records.empty:
        st.info("Structured analysis becomes available when the report contains extractable financial tables.")
        return
    left, right = st.columns(2)
    with left:
        metric = st.selectbox("Metric", sorted(records["Metric"].unique()))
    with right:
        scope = st.selectbox("Reporting scope", ["All"] + sorted(records["Scope"].unique()))
    filtered = records[records["Metric"] == metric]
    if scope != "All":
        filtered = filtered[filtered["Scope"] == scope]
    st.dataframe(
        filtered[["Metric", "Year", "Scope", "Value", "Unit", "Source page", "Chunk ID"]],
        use_container_width=True,
        hide_index=True,
    )
    try:
        import plotly.express as px
        if not filtered.empty:
            chart = px.line(
                filtered.sort_values("Year"),
                x="Year",
                y="Value",
                color="Scope",
                markers=True,
                title=f"{metric} by reporting period",
                labels={"Value": f"Value ({filtered['Unit'].iloc[0]})"},
            )
            chart.update_layout(hovermode="x unified", margin={"t": 55, "b": 10})
            st.plotly_chart(chart, use_container_width=True)
    except ImportError:
        st.warning("Install Plotly from requirements.txt to display interactive charts.")
    yoy = calculate_yoy(filtered)
    if not yoy.empty:
        st.subheader("Period-over-period change")
        st.dataframe(yoy[["Year", "Scope", "Value", "Change", "Change %", "Unit"]], use_container_width=True, hide_index=True)


def _segments(records: pd.DataFrame) -> None:
    st.markdown('<div class="eyebrow">Segment analysis</div>', unsafe_allow_html=True)
    st.title("Business segments")
    segment_records = records[
        records["Metric"].str.contains("segment|automotive|jaguar|commercial|passenger", case=False, regex=True, na=False)
    ]
    if segment_records.empty:
        st.info("No segment-level figures were identified in the extracted tables. FinVis will not invent a segment view.")
        return
    st.dataframe(segment_records, use_container_width=True, hide_index=True)


def _evidence(records: pd.DataFrame, chunks: list[dict]) -> None:
    st.markdown('<div class="eyebrow">Evidence explorer</div>', unsafe_allow_html=True)
    st.title("Every figure has a source")
    if records.empty:
        st.info("No structured figures are available; showing raw indexed evidence instead.")
        evidence = pd.DataFrame(chunks)
        if not evidence.empty:
            st.dataframe(evidence[["page_number", "chunk_id", "content_type", "preview"]], use_container_width=True, hide_index=True)
        return
    selected = st.multiselect("Filter metrics", sorted(records["Metric"].unique()))
    filtered = records[records["Metric"].isin(selected)] if selected else records
    st.dataframe(
        filtered[["Metric", "Year", "Scope", "Value", "Unit", "Source page", "Source table", "Chunk ID"]],
        use_container_width=True,
        hide_index=True,
    )
    for _, row in filtered.head(20).iterrows():
        with st.expander(f"{row['Metric']} · {row['Year']} · page {row['Source page']}"):
            st.caption(f"{row['Scope']} · {row['Unit']} · chunk {row['Chunk ID']}")
            st.write(row["Evidence"])


def _ask() -> None:
    st.markdown('<div class="eyebrow">Ask FinVis</div>', unsafe_allow_html=True)
    st.title("Ask about the report")
    if st.session_state.index is None:
        st.info("Index a report first.")
        return
    question = st.text_area("Question", placeholder="What was revenue from operations in FY25?", height=90)
    if st.button("Ask", type="primary") and question.strip():
        try:
            with st.spinner("Retrieving evidence..."):
                if st.session_state.qa is None:
                    st.session_state.qa = load_qa_model()
                result = answer_question(question, st.session_state.index, *st.session_state.qa)
            if result.get("answer"):
                st.subheader(result["answer"])
            else:
                st.warning("The indexed evidence does not support a reliable answer.")
            for source in result.get("sources", []):
                st.caption(
                    f"{source.get('source_filename')} · page {source.get('page_number')} · "
                    f"chunk {source.get('chunk_id')}"
                )
            for evidence in result.get("evidence", []):
                with st.expander("Supporting evidence"):
                    st.write(evidence.get("text", ""))
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(f"Unable to answer the question: {exc}")


def main() -> None:
    _init_state()
    _sidebar()
    pages = ["Overview", "Financial Analysis", "Segment Analysis", "Evidence Explorer", "Ask FinVis"]
    selected = st.sidebar.radio("Workspace", pages, label_visibility="collapsed")
    records = st.session_state.records
    if selected == "Overview":
        _overview(records)
    elif selected == "Financial Analysis":
        _analysis(records)
    elif selected == "Segment Analysis":
        _segments(records)
    elif selected == "Evidence Explorer":
        _evidence(records, st.session_state.chunks)
    else:
        _ask()


if __name__ == "__main__":
    main()
