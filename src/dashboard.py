"""Structured financial records and deterministic dashboard calculations."""

from __future__ import annotations

import re
import math
from typing import Any, Iterable

import pandas as pd


_NUMBER_RE = re.compile(
    r"^\s*(?:[₹$€£]\s*)?"
    r"(-?(?:\(?\d[\d,]*(?:\.\d+)?\)?))"
    r"\s*(%|crores?|lakhs?|million|millions|billion|billions)?\s*$",
    re.IGNORECASE,
)

_SCALE_MULTIPLIERS = {
    "lakh": 100_000,
    "lakhs": 100_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
    "million": 1_000_000,
    "millions": 1_000_000,
    "billion": 1_000_000_000,
    "billions": 1_000_000_000,
}


def parse_financial_value(value: Any) -> float | None:
    """Parse a displayed financial cell without changing its sign."""
    if not isinstance(value, str):
        return None
    match = _NUMBER_RE.fullmatch(value)
    if not match:
        return None
    text = match.group(1).replace(",", "").strip()
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    if text.startswith("-"):
        negative = True
        text = text[1:]
    scale = match.group(2).casefold() if match.group(2) else ""
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    result = -number if negative else number
    if scale and scale != "%":
        result *= _SCALE_MULTIPLIERS.get(scale, 1)
    return result


def financial_records(chunks: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Convert table chunks into evidence-preserving normalized records."""
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk.get("content_type") != "table":
            continue
        values = chunk.get("table_values")
        if not isinstance(values, dict):
            continue
        metric = str(chunk.get("table_row_label", "")).strip()
        if not metric:
            continue
        for header, raw_value in values.items():
            numeric_value = parse_financial_value(raw_value)
            if numeric_value is None:
                continue
            header_text = str(header).strip()
            scope = next(
                (item.title() for item in ("standalone", "consolidated") if item in header_text.casefold()),
                "Unspecified",
            )
            year_match = re.search(r"(?:FY)?(20\d{2}|[12]\d)", header_text, re.IGNORECASE)
            year = year_match.group(0).upper() if year_match else "Unspecified"
            rows.append(
                {
                    "Metric": metric,
                    "Year": year,
                    "Scope": scope,
                    "Value": numeric_value,
                    "Raw value": str(raw_value),
                    "Unit": str(chunk.get("table_unit", "")).strip() or "Not specified",
                    "Source page": chunk.get("page_number"),
                    "Source table": chunk.get("table_id", ""),
                    "Chunk ID": chunk.get("chunk_id"),
                    "Evidence": chunk.get("text", ""),
                    "Source document": chunk.get("source_filename", ""),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "Metric", "Year", "Scope", "Value", "Raw value", "Unit",
                "Source page", "Source table", "Chunk ID", "Evidence", "Source document",
            ]
        )
    return pd.DataFrame(rows).drop_duplicates(
        subset=["Metric", "Year", "Scope", "Value", "Chunk ID"]
    )


def metric_family(metric: str) -> str:
    lowered = metric.casefold()
    if "revenue" in lowered or "income" in lowered:
        return "Revenue"
    if "profit" in lowered or "loss" in lowered or "ebit" in lowered:
        return "Profitability"
    if "expense" in lowered or "cost" in lowered:
        return "Expenses"
    if "debt" in lowered or "liabilit" in lowered or "asset" in lowered:
        return "Balance sheet"
    if "cash" in lowered:
        return "Cash flow"
    return "Other"


def add_families(records: pd.DataFrame) -> pd.DataFrame:
    result = records.copy()
    if not result.empty:
        result["Category"] = result["Metric"].map(metric_family)
    return result


def calculate_yoy(records: pd.DataFrame) -> pd.DataFrame:
    """Calculate changes only when two numeric periods share metric/scope/unit."""
    if records.empty:
        return pd.DataFrame()
    result = records.copy()
    extracted_year = result["Year"].str.extract(r"(20\d{2}|[12]\d)", expand=False)
    result["Year number"] = pd.to_numeric(extracted_year, errors="coerce")
    result.loc[result["Year number"] < 100, "Year number"] += 2000
    result = result.dropna(subset=["Year number"])
    result = result.sort_values(["Metric", "Scope", "Unit", "Year number"])
    result["Change"] = result.groupby(["Metric", "Scope", "Unit"])["Value"].diff()
    prior_value = result.groupby(["Metric", "Scope", "Unit"])["Value"].shift(1)
    result["Change %"] = (result["Change"] / prior_value.abs().replace(0, pd.NA)) * 100
    return result.dropna(subset=["Change"])


def calculate_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return a percentage ratio, or None for missing or zero denominator values."""
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return None
    if denominator == 0:
        return None
    return float(numerator) / float(denominator) * 100


def profit_margin(records: pd.DataFrame, profit_metric: str, revenue_metric: str) -> pd.DataFrame:
    """Calculate a profit margin only within matching year, scope, and unit."""
    if records.empty:
        return pd.DataFrame()
    profits = records[records["Metric"].str.casefold().eq(profit_metric.casefold())]
    revenues = records[records["Metric"].str.casefold().eq(revenue_metric.casefold())]
    if profits.empty or revenues.empty:
        return pd.DataFrame()
    merged = profits.merge(
        revenues,
        on=["Year", "Scope", "Unit"],
        suffixes=("_profit", "_revenue"),
    )
    if merged.empty:
        return pd.DataFrame()
    output = merged[["Year", "Scope", "Unit"]].copy()
    output["Ratio %"] = [
        calculate_ratio(profit, revenue)
        for profit, revenue in zip(merged["Value_profit"], merged["Value_revenue"])
    ]
    return output.dropna(subset=["Ratio %"])
