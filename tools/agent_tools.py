from langchain_core.tools import tool
import numexpr
import pandas as pd
import os
from tavily import TavilyClient

# Initialize Tavily client (requires TAVILY_API_KEY env var)
tavily_api_key = os.getenv("TAVILY_API_KEY", "")
if tavily_api_key:
    tavily_client = TavilyClient(api_key=tavily_api_key)
else:
    tavily_client = None

@tool
def web_search(query: str) -> str:
    """Search the web for information using Tavily API."""
    if not tavily_client:
        return "Web search unavailable - no API key configured."
    try:
        results = tavily_client.search(query=query, max_results=3)
        if results and results.get("results"):
            return "\n".join([f"{r['title']}: {r['content']}" for r in results["results"]])
        return "No search results found."
    except Exception as e:
        return f"Search failed: {e}"

@tool
def calculate(expression: str) -> str:
    """Evaluate mathematical expressions safely."""
    try:
        result = numexpr.evaluate(expression)
        return str(result)
    except Exception as e:
        return f"Calculation failed: {e}"

@tool
def read_csv(filepath: str) -> str:
    """Read and preview CSV data."""
    try:
        df = pd.read_csv(filepath)
        preview = df.head(10).to_string()
        return f"CSV Preview:\n{preview}"
    except Exception as e:
        return f"Failed to read CSV: {e}"

@tool
def python_executor(code: str) -> str:
    """Execute Python code for complex computations."""
    try:
        # Safe execution with limited globals
        allowed_globals = {"__builtins__": {"len": len, "sum": sum, "max": max, "min": min}}
        exec_globals = {}
        exec(code, allowed_globals, exec_globals)
        return str(exec_globals.get("result", "Code executed successfully"))
    except Exception as e:
        return f"Code execution failed: {e}"

# List of all available tools
AGENT_TOOLS = [web_search, calculate, read_csv, python_executor]