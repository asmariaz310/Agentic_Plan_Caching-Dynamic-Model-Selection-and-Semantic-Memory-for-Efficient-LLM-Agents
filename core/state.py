from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """State management for the agentic plan caching workflow."""
    query: str
    keyword: Optional[str]
    context: str
    cached_template: Optional[Dict[str, Any]]
    cache_hit: bool
    fallback_to_large: bool
    final_answer: Optional[str]
    execution_log: List[Dict[str, Any]]
    messages: List[BaseMessage]
    current_step: str
    max_iterations: int
    iteration_count: int
    query_domain: Optional[str]
    cache_confidence: Optional[float]
    selected_model: Optional[str]
    # Cache statistics tracking
    cache_stats: Optional[Dict[str, Any]]