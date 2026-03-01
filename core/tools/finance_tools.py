#finance_tools.py

import os 
import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
from core.rag.rag import search,list_indexes

load_dotenv()

@tool
def calculate_metrics(
    revenue:            float,
    net_income:         float,
    total_assets:       float,
    total_equity:       float,
    shares_outstanding: float,
    current_price:      float,
) -> str:
    """
    Calculate key financial ratios from raw financial data.
    Use this when the user provides financial numbers and wants
    to understand profitability, valuation, or efficiency metrics.
    All inputs should be in the same currency unit.
    """

    eps           = (net_income / shares_outstanding
                     if shares_outstanding else 0)

    pe_ratio      = (current_price / eps
                     if eps > 0 else None)

    profit_margin = (net_income / revenue * 100
                     if revenue else 0)

    roe           = (net_income / total_equity * 100
                     if total_equity else 0)

    roa           = (net_income / total_assets * 100
                     if total_assets else 0)

    pe_display = f"{pe_ratio:.2f}x" if pe_ratio else "N/A (negative earnings)"

    margin_signal = (
        "✅ Healthy (>10%)" if profit_margin > 10
        else "⚠️ Thin (<10%)"
    )
    roe_signal = (
        "✅ Strong (>15%)" if roe > 15
        else "⚠️ Below average"
    )
    roa_signal = (
        "✅ Good (>5%)" if roa > 5
        else "⚠️ Low (<5%)"
    )
    pe_signal = (
        "✅ Reasonable" if pe_ratio and 5 < pe_ratio < 30
        else "⚠️ Check valuation"
    )

    return (
        f"Financial Metrics Analysis\n"
        f"{'─' * 35}\n"
        f"EPS (Earnings Per Share) : ${eps:,.2f}\n"
        f"P/E Ratio                : {pe_display} {pe_signal}\n"
        f"Profit Margin            : {profit_margin:.1f}% {margin_signal}\n"
        f"Return on Equity (ROE)   : {roe:.1f}% {roe_signal}\n"
        f"Return on Assets (ROA)   : {roa:.1f}% {roa_signal}\n"
        f"{'─' * 35}\n"
        f"Revenue    : ${revenue:,.0f}\n"
        f"Net Income : ${net_income:,.0f}\n"
    )
