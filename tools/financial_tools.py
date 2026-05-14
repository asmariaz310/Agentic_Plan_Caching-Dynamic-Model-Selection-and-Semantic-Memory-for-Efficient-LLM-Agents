from langchain_core.tools import tool
import pandas as pd
import re

@tool
def extract_financial_metrics(text: str) -> dict:
    """Extract key financial metrics from text."""
    metrics = {}

    # Extract current assets
    ca_match = re.search(r'current assets[^0-9]*\$?([0-9,]+)', text, re.IGNORECASE)
    if ca_match:
        metrics['current_assets'] = int(ca_match.group(1).replace(',', ''))

    # Extract current liabilities
    cl_match = re.search(r'current liabilities[^0-9]*\$?([0-9,]+)', text, re.IGNORECASE)
    if cl_match:
        metrics['current_liabilities'] = int(cl_match.group(1).replace(',', ''))

    # Extract total assets
    ta_match = re.search(r'total assets[^0-9]*\$?([0-9,]+)', text, re.IGNORECASE)
    if ta_match:
        metrics['total_assets'] = int(ta_match.group(1).replace(',', ''))

    # Extract total liabilities
    tl_match = re.search(r'total liabilities[^0-9]*\$?([0-9,]+)', text, re.IGNORECASE)
    if tl_match:
        metrics['total_liabilities'] = int(tl_match.group(1).replace(',', ''))

    return metrics

@tool
def calculate_working_capital(current_assets: int, current_liabilities: int) -> float:
    """Calculate working capital ratio."""
    if current_liabilities == 0:
        return 0.0
    return current_assets / current_liabilities

@tool
def analyze_balance_sheet(csv_path: str) -> dict:
    """Analyze balance sheet data from CSV."""
    try:
        df = pd.read_csv(csv_path)
        analysis = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "summary_stats": df.describe().to_dict()
        }
        return analysis
    except Exception as e:
        return {"error": str(e)}

# Financial tools list
FINANCIAL_TOOLS = [extract_financial_metrics, calculate_working_capital, analyze_balance_sheet]