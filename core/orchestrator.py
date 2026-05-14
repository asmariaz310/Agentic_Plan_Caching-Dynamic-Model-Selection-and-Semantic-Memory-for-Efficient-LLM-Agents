from langgraph.graph import StateGraph, END
from core import state
from core.state import AgentState
from agents.planner import PlannerAgent
from agents.actor import ActorAgent
from agents.scientific_agent import ScientificAgent
from agents.sanitization import SanitizationAgent
from memory.plan_cache import PlanCache
from langchain_core.messages import AIMessage
import json
import os
import time
import hashlib
from datetime import datetime


class TokenTracker:
    """Track token usage across API calls to stay within daily limits."""

    def __init__(self, daily_limit=1000, cache_file="token_usage.json"):
        self.daily_limit = daily_limit
        self.cache_file = cache_file
        self.tokens_used = 0
        self._load_usage()

    def _load_usage(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.tokens_used = data.get("tokens_used", 0)
            except:
                pass

    def _save_usage(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({"tokens_used": self.tokens_used, "timestamp": datetime.now().isoformat()}, f)
        except:
            pass

    def estimate_tokens(self, text):
        """Estimate tokens: ~4 chars = 1 token"""
        return max(1, len(text) // 4)

    def add_tokens(self, count, operation="api"):
        self.tokens_used += count
        self._save_usage()
        remaining = self.daily_limit - self.tokens_used
        percent = (self.tokens_used / self.daily_limit) * 100
        print(f"Tokens: {self.tokens_used}/{self.daily_limit} ({percent:.1f}%) | Remaining: {remaining}")
        return remaining > 0

    def can_use(self, estimated_tokens):
        return estimated_tokens <= (self.daily_limit - self.tokens_used)

    def get_stats(self):
        remaining = self.daily_limit - self.tokens_used
        return {
            "used": self.tokens_used,
            "limit": self.daily_limit,
            "remaining": remaining,
            "percent": (self.tokens_used / self.daily_limit) * 100
        }

    def reset(self):
        self.tokens_used = 0
        self._save_usage()


class ResponseCache:
    """Aggressive caching to minimize token usage."""

    def __init__(self, cache_file="response_cache.json"):
        self.cache_file = cache_file
        self.cache = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except:
                pass

    def _save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except:
            pass

    def _get_key(self, query, context=""):
        combined = f"{query}|{context}".lower().strip()
        return hashlib.md5(combined.encode()).hexdigest()

    def get(self, query, context=""):
        key = self._get_key(query, context)
        cached = self.cache.get(key)
        return cached.get("response") if cached else None

    def set(self, query, response, context=""):
        key = self._get_key(query, context)
        self.cache[key] = {"response": response, "query": query, "timestamp": datetime.now().isoformat()}
        self._save_cache()

    def stats(self):
        return {"entries": len(self.cache)}


class AgentOrchestrator:
    def __init__(self, plan_cache: PlanCache, large_model="llama-3.3-70b-versatile", small_model="llama-3.1-8b-instant"):
        """Initialize orchestrator with Groq models and domain-specific agents."""
        self.plan_cache = plan_cache
        self.large_model = large_model
        self.small_model = small_model
        self.planner_agent = PlannerAgent(large_model=large_model, small_model=small_model)
        self.actor_agent = ActorAgent(model_name=small_model)
        self.scientific_agent = ScientificAgent(model_name=large_model)  # Use large model for scientific analysis
        self.sanitization_agent = SanitizationAgent()
        self.cache_confidence_threshold = 0.80
        self.domain_metrics = {}
        # Add token management
        self.token_tracker = TokenTracker(daily_limit=1000)
        self.response_cache = ResponseCache()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)

        # Define Nodes
        builder.add_node("extract_keyword", self.extract_keyword_node)
        builder.add_node("cache_lookup", self.cache_lookup_node)
        builder.add_node("large_planner", self.large_planner_node)
        builder.add_node("small_planner", self.small_planner_node)
        builder.add_node("actor", self.actor_node)
        builder.add_node("scientific_analyzer", self.scientific_analyzer_node)
        builder.add_node("cache_generator", self.cache_generator_node)

        # Define Edges
        builder.set_entry_point("extract_keyword")
        builder.add_edge("extract_keyword", "cache_lookup")

        # Conditional edge from cache_lookup
        def route_after_lookup(state: AgentState):
            # Check if this is a scientific query
            if self._is_scientific_query(state["query"]):
                return "scientific_analyzer"
            elif state.get("cache_hit") and state.get("cached_template"):
                if state.get("cache_confidence", 0.0) >= self.cache_confidence_threshold:
                    return "small_planner"
                return "large_planner"
            return "large_planner"

        builder.add_conditional_edges("cache_lookup", route_after_lookup)

        # Conditional edge from small_planner
        def route_after_small_planner(state: AgentState):
            if state.get("fallback_to_large"):
                return "large_planner"
            if state.get("final_answer"):
                return END
            return "actor"

        builder.add_conditional_edges("small_planner", route_after_small_planner)

        # Conditional edge from large_planner
        def route_after_large_planner(state: AgentState):
            if state.get("final_answer"):
                return "cache_generator"
            return "actor"

        builder.add_conditional_edges("large_planner", route_after_large_planner)

        # Conditional edge from actor
        def route_after_actor(state: AgentState):
            if state.get("final_answer"):
                return "cache_generator"
            # Continue planning
            if state.get("cache_hit"):
                return "small_planner"
            return "large_planner"

        builder.add_conditional_edges("actor", route_after_actor)

        builder.add_edge("cache_generator", END)

        return builder.compile()

    def extract_keyword_node(self, state: AgentState) -> AgentState:
        """Extract keyword and domain from query."""
        keyword = self.plan_cache.extract_keyword(state["query"])
        state["keyword"] = keyword
        state["query_domain"] = self._get_query_domain(state["query"])
        return state

    def cache_lookup_node(self, state: AgentState) -> AgentState:
        """Check cache for existing template and track statistics."""
        cache_result = self.plan_cache.lookup(state["query"], state["keyword"])
        state["cache_hit"] = cache_result["template"] is not None
        state["cached_template"] = cache_result["template"]
        state["cache_confidence"] = cache_result["confidence"]
        
        # Capture cache statistics at lookup time
        state["cache_stats"] = self.plan_cache.get_cache_statistics()
        
        return state

    def large_planner_node(self, state: AgentState) -> AgentState:
        """Use the selected planner to generate a plan."""
        selected_model = self._select_optimal_model(
            state["query"],
            state.get("query_domain"),
            state["context"],
            state.get("cache_confidence", 0.0)
        )
        state["selected_model"] = selected_model

        if selected_model == self.small_model:
            plan = self.planner_agent.fast_plan(
                state["query"],
                state["context"],
                state["messages"]
            )
        else:
            plan = self.planner_agent.large_plan(
                state["query"],
                state["context"],
                state["messages"]
            )

        state["messages"].append(AIMessage(content=plan))

        # Check if this is a final answer
        if "FINAL ANSWER:" in plan:
            state["final_answer"] = plan.split("FINAL ANSWER:")[-1].strip()

        return state

    def small_planner_node(self, state: AgentState) -> AgentState:
        """Use small planner to adapt cached template."""
        state["selected_model"] = self.small_model
        response, fallback = self.planner_agent.small_plan(
            state["query"],
            state["context"],
            state["messages"],
            state["cached_template"]
        )

        if fallback:
            state["fallback_to_large"] = True
            return state

        state["messages"].append(AIMessage(content=response))

        # Check if this is a final answer
        if "FINAL ANSWER:" in response:
            state["final_answer"] = response.split("FINAL ANSWER:")[-1].strip()

        return state
    
    def actor_node(self, state: AgentState) -> AgentState:
        """Execute action step using ActorAgent."""
        
        # Get last message (latest plan step)
        if not state["messages"]:
            return state

        last_message = state["messages"][-1].content

        # Execute action
        result = self.actor_agent.act(last_message)

        # Store result in messages
        state["messages"].append(AIMessage(content=result))

        # Log execution (important for cache/template generation)
        state["execution_log"].append({
            "action": last_message,
            "result": result
        })

        # Check if actor produced final answer
        if isinstance(result, str) and "FINAL ANSWER:" in result:
            state["final_answer"] = result.split("FINAL ANSWER:")[-1].strip()

        return state

    def scientific_analyzer_node(self, state: AgentState) -> AgentState:
        """Use scientific agent for domain-specific analysis."""
        answer = self.scientific_agent.analyze_paper_query(state["query"], state["context"])
        state["final_answer"] = answer
        return state

    def cache_generator_node(self, state: AgentState) -> AgentState:
        """Generate and store cache template."""
        if not state.get("cache_hit"):  # Only generate for cache misses
            # Sanitize execution log before generating template
            sanitized_log = self.sanitization_agent.sanitize_log(state["execution_log"])
            template = self.plan_cache.generate_template(
                sanitized_log,
                state["keyword"],
                state["query"]
            )
        return state

    def run(self, query: str, context: str = "") -> AgentState:
        """Run the agentic workflow with optimizations and cache tracking."""
        initial_state = AgentState(
            query=query,
            keyword=None,
            context=context,
            cached_template=None,
            cache_hit=False,
            fallback_to_large=False,
            final_answer=None,
            execution_log=[],
            messages=[],
            current_step="start",
            max_iterations=5,  # Reduced from 10
            iteration_count=0,
            cache_stats=None,  # Will be populated during execution
            query_domain=None,
            cache_confidence=None,
            selected_model=None
        )

        # Quick check for simple math queries - bypass full pipeline
        if self._is_simple_math(query):
            answer = self._solve_simple_math(query)
            initial_state["final_answer"] = answer
            initial_state["query_domain"] = self._get_query_domain(query)
            initial_state["selected_model"] = self.small_model
            # Still capture cache stats for metrics
            initial_state["cache_stats"] = self.plan_cache.get_cache_statistics()
            return initial_state

        return self.graph.invoke(initial_state)

    def _is_simple_math(self, query: str) -> bool:
        """Check if query is simple math that can be solved directly."""
        query_lower = query.lower()
        has_math_ops = any(op in query for op in ['+', '-', '*', '/', 'sqrt'])
        has_square_root = 'square root' in query_lower
        return (has_math_ops or has_square_root) and len(query.split()) < 12

    def _is_scientific_query(self, query: str) -> bool:
        """Check if query is scientific paper analysis."""
        query_lower = query.lower()

        # Scientific paper indicators
        scientific_indicators = [
            'paper', 'contribution', 'methodology', 'evaluation', 'architecture',
            'baseline', 'dataset', 'performance', 'result', 'experiment',
            'novelty', 'innovation', 'framework', 'approach', 'technique'
        ]

        # Check for scientific keywords
        has_scientific_terms = any(term in query_lower for term in scientific_indicators)

        # Check for question patterns typical of scientific analysis
        question_patterns = [
            'what is the main', 'what are the key', 'how does', 'what is the',
            'what are the', 'how do', 'what does'
        ]

        has_question_pattern = any(pattern in query_lower for pattern in question_patterns)

        return has_scientific_terms and has_question_pattern

    def _get_query_domain(self, query: str) -> str:
        """Classify the query domain for dynamic model selection."""
        query_lower = query.lower()

        if self._is_scientific_query(query):
            return 'scientific'
        if any(term in query_lower for term in ['finance', 'financial', 'stock', 'market', 'investment', 'interest rate']):
            return 'finance'
        if any(term in query_lower for term in ['calculate', 'compute', 'solve', 'sum', 'average', 'ratio', 'percentage', 'sqrt']):
            return 'math'
        if any(term in query_lower for term in ['who', 'what', 'where', 'when', 'why', 'how']):
            return 'factual'
        return 'general'

    def _select_optimal_model(self, query: str, domain: str = 'general', context: str = "", cache_confidence: float = 0.0) -> str:
        """Select the optimal model based on query complexity, domain, and cache confidence."""
        query_lower = query.lower()
        query_length = len(query.split())

        if domain == 'scientific':
            return self.large_model

        if domain == 'finance':
            return self.large_model

        if cache_confidence >= self.cache_confidence_threshold:
            return self.small_model

        if self._is_simple_math(query):
            return self.small_model

        if query_length > 25 or '?' in query or len(context.split()) > 40:
            return self.large_model

        if domain == 'math' and query_length <= 18:
            return self.small_model

        if domain == 'factual':
            return self.small_model

        return self.small_model

    def record_domain_metrics(self, domain: str, latency: float, is_correct: bool, selected_model: str):
        """Record per-domain performance metrics for later analysis."""
        if domain not in self.domain_metrics:
            self.domain_metrics[domain] = {
                'queries': 0,
                'correct': 0,
                'latency': 0.0,
                'model_usage': {}
            }

        metrics = self.domain_metrics[domain]
        metrics['queries'] += 1
        metrics['correct'] += 1 if is_correct else 0
        metrics['latency'] += latency
        metrics['model_usage'][selected_model] = metrics['model_usage'].get(selected_model, 0) + 1

    def _solve_simple_math(self, query: str) -> str:
        """Solve simple math problems directly."""
        try:
            import re
            import math
            
            # Extract numbers and operations
            if 'sqrt' in query.lower() or 'square root' in query.lower():
                # Handle both "sqrt(16)" and "square root of 16"
                match = re.search(r'sqrt\((\d+)\)', query)
                if not match:
                    match = re.search(r'square root of (\d+)', query.lower())
                if match:
                    return str(math.sqrt(int(match.group(1))))
            
            # Basic arithmetic
            match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', query)
            if match:
                a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
                if op == '+':
                    return str(a + b)
                elif op == '-':
                    return str(a - b)
                elif op == '*':
                    return str(a * b)
                elif op == '/' and b != 0:
                    return str(a / b)
            
            # Sum of integers
            if 'sum' in query.lower() and 'to' in query.lower():
                match = re.search(r'sum.*(\d+).*to.*(\d+)', query)
                if match:
                    start, end = int(match.group(1)), int(match.group(2))
                    return str(sum(range(start, end + 1)))
                    
        except:
            pass
        
        # Fallback to actor
        return self.actor_agent.act(f"Solve: {query}")

    def run_token_efficient(self, query: str, context: str = "") -> dict:
        """Run with token efficiency: cache → check limit → API → mock fallback"""
        
        # STEP 1: Check cache first (0 tokens)
        cached = self.response_cache.get(query, context)
        if cached:
            return {
                "query": query,
                "answer": cached,
                "source": "cache",
                "tokens_used": 0,
                "cached": True
            }
        
        # STEP 2: Estimate token cost
        est_tokens = self.token_tracker.estimate_tokens(query) + \
                     self.token_tracker.estimate_tokens(context) + 150  # +150 for response
        
        # STEP 3: Check if we have tokens
        if not self.token_tracker.can_use(est_tokens):
            answer = self._generate_mock_answer(query, context)
            self.response_cache.set(query, answer, context)
            return {
                "query": query,
                "answer": answer,
                "source": "mock",
                "tokens_used": 0,
                "cached": False
            }
        
        # STEP 4: Make actual API call with error handling
        # STEP 4: Make actual API call with error handling
        try:
            final_state = self.run(query, context)
            answer = final_state.get("final_answer", "No answer")
            
            # STEP 5: Track actual token usage and cache result
            actual_tokens = self.token_tracker.estimate_tokens(answer)
            self.token_tracker.add_tokens(actual_tokens)
            self.response_cache.set(query, answer, context)
            
            return {
                "query": query,
                "answer": answer,
                "source": "api",
                "tokens_used": actual_tokens,
                "cached": False
            }
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg or "quota" in error_msg:
                answer = self._generate_mock_answer(query, context)
                self.response_cache.set(query, answer, context)
                return {
                    "query": query,
                    "answer": answer,
                    "source": "mock",
                    "tokens_used": 0,
                    "cached": False
                }
            else:
                # Re-raise other errors
                raise e
    
    def _generate_mock_answer(self, query: str, context: str):
        """Generate intelligent mock answer based on context and query type"""
        query_lower = query.lower()
        
        # Math questions
        if any(op in query for op in ['+', '-', '*', '/', 'sqrt']):
            try:
                return self._solve_simple_math(query)
            except:
                return "Math queries should be answered via direct calculation"
        
        # Scientific questions
        if any(term in query_lower for term in ['contribution', 'methodology', 'evaluation', 'architecture']):
            try:
                return self.scientific_agent.analyze_paper_query(query, context)
            except:
                return "Context-based answer for scientific query"
        
        # Default fallback with context awareness
        if context:
            return f"Based on context: {context[:100]}..."
        return f"Context-based answer for: {query[:50]}..."

    def get_cache_statistics(self) -> dict:
        """Get and return cache hit rate statistics."""
        return self.plan_cache.get_cache_statistics()

    def report_cache_performance(self):
        """Print detailed cache performance report."""
        stats = self.get_cache_statistics()
        
        print("\n" + "="*60)
        print("CACHE PERFORMANCE REPORT")
        print("="*60)
        
        # Template Cache Stats
        print("\n[Template Cache]")
        print(f"  Hits:     {stats['template_cache']['hits']}")
        print(f"  Misses:   {stats['template_cache']['misses']}")
        print(f"  Total:    {stats['template_cache']['total']}")
        print(f"  Hit Rate: {stats['template_cache']['hit_rate']:.2f}%")
        
        # LLM Response Cache Stats
        print("\n[LLM Response Cache]")
        print(f"  Hits:     {stats['llm_response_cache']['hits']}")
        print(f"  Misses:   {stats['llm_response_cache']['misses']}")
        print(f"  Total:    {stats['llm_response_cache']['total']}")
        print(f"  Hit Rate: {stats['llm_response_cache']['hit_rate']:.2f}%")
        
        # Overall Stats
        print("\n[Overall Performance]")
        print(f"  Total Hits:      {stats['overall']['total_hits']}")
        print(f"  Total Requests:  {stats['overall']['total_requests']}")
        print(f"  Overall Hit Rate: {stats['overall']['overall_hit_rate']:.2f}%")
        print("="*60 + "\n")