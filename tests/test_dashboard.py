import pandas as pd

from src.dashboard import calculate_yoy, financial_records, parse_financial_value


def test_parse_financial_value_preserves_parenthesized_sign():
    assert parse_financial_value("₹1,234.50") == 1234.5
    assert parse_financial_value("(922)") == -922
    assert parse_financial_value("not a number") is None


def test_financial_records_preserve_table_evidence_metadata():
    records = financial_records(
        [
            {
                "content_type": "table",
                "table_row_label": "Revenue from operations",
                "table_values": {"Standalone FY25": "69,419", "Standalone FY24": "60,000"},
                "table_unit": "₹ crore",
                "page_number": 150,
                "table_id": "doc:150:0",
                "chunk_id": 42,
                "text": "Revenue from operations; Standalone FY25 = 69,419",
                "source_filename": "report.pdf",
            }
        ]
    )
    assert list(records["Year"]) == ["FY25", "FY24"]
    assert list(records["Value"]) == [69419.0, 60000.0]
    assert set(records["Chunk ID"]) == {42}
    assert set(records["Source page"]) == {150}


def test_calculate_yoy_uses_prior_period_within_same_scope():
    records = pd.DataFrame(
        [
            {"Metric": "Revenue", "Year": "FY2024", "Scope": "Standalone", "Unit": "₹ crore", "Value": 100.0},
            {"Metric": "Revenue", "Year": "FY2025", "Scope": "Standalone", "Unit": "₹ crore", "Value": 125.0},
        ]
    )
    result = calculate_yoy(records)
    assert len(result) == 1
    assert result.iloc[0]["Change"] == 25.0
    assert result.iloc[0]["Change %"] == 25.0


def test_calculate_yoy_normalizes_short_financial_years():
    records = pd.DataFrame(
        [
            {"Metric": "Revenue", "Year": "FY24", "Scope": "Standalone", "Unit": "₹ crore", "Value": 100.0},
            {"Metric": "Revenue", "Year": "FY25", "Scope": "Standalone", "Unit": "₹ crore", "Value": 125.0},
        ]
    )
    result = calculate_yoy(records)
    assert len(result) == 1
    assert result.iloc[0]["Change"] == 25.0
