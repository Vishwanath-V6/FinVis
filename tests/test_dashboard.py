import pandas as pd

from src.dashboard import calculate_ratio, calculate_yoy, financial_records, parse_financial_value, profit_margin


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


def test_parse_indian_and_currency_units():
    assert parse_financial_value("₹12,34,567") == 1234567
    assert parse_financial_value("12.5 crore") == 125000000
    assert parse_financial_value("2 million") == 2000000
    assert parse_financial_value("-(922)") is None


def test_ratio_rejects_missing_and_zero_denominators():
    assert calculate_ratio(25, 100) == 25
    assert calculate_ratio(25, 0) is None
    assert calculate_ratio(None, 100) is None


def test_profit_margin_matches_scope_year_and_unit():
    records = pd.DataFrame([
        {"Metric": "Revenue", "Year": "FY2024", "Scope": "Consolidated", "Unit": "₹ crore", "Value": 100},
        {"Metric": "Net profit", "Year": "FY2024", "Scope": "Consolidated", "Unit": "₹ crore", "Value": 25},
        {"Metric": "Net profit", "Year": "FY2024", "Scope": "Standalone", "Unit": "₹ crore", "Value": 20},
    ])
    result = profit_margin(records, "Net profit", "Revenue")
    assert len(result) == 1
    assert result.iloc[0]["Ratio %"] == 25
