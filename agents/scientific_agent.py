from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.rate_limiter import GroqAPIManager
import re

class ScientificAgent:
    """Specialized agent for scientific paper analysis and QASPER queries."""

    def __init__(self, model_name="llama-3.3-70b-versatile"):
        """Initialize scientific agent with specialized model."""
        self.llm = ChatGroq(model=model_name, temperature=0.1)  # Slightly higher temperature for creative scientific reasoning
        self.api_manager = GroqAPIManager(requests_per_minute=20)  # Conservative for scientific analysis

    def analyze_paper_query(self, query: str, context: str = "") -> str:
        """Analyze scientific paper queries with domain expertise."""

        # Detect query type and apply specialized reasoning
        query_type = self._classify_query_type(query)

        if query_type == "contribution":
            return self._analyze_contribution(query, context)
        elif query_type == "methodology":
            return self._analyze_methodology(query, context)
        elif query_type == "evaluation":
            return self._analyze_evaluation(query, context)
        elif query_type == "architecture":
            return self._analyze_architecture(query, context)
        elif query_type == "comparison":
            return self._analyze_comparison(query, context)
        else:
            return self._general_scientific_analysis(query, context)

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of scientific query."""
        query_lower = query.lower()

        if any(word in query_lower for word in ['contribution', 'main finding', 'key result', 'propose', 'introduce']):
            return "contribution"
        elif any(word in query_lower for word in ['method', 'approach', 'technique', 'algorithm', 'framework']):
            return "methodology"
        elif any(word in query_lower for word in ['evaluate', 'performance', 'result', 'experiment', 'dataset']):
            return "evaluation"
        elif any(word in query_lower for word in ['architecture', 'system', 'design', 'structure']):
            return "architecture"
        elif any(word in query_lower for word in ['compare', 'baseline', 'versus', 'vs', 'better than']):
            return "comparison"
        else:
            return "general"

    def _analyze_contribution(self, query: str, context: str) -> str:
        """Analyze paper contributions with scientific reasoning."""
        sys_prompt = """You are a scientific paper analyst specializing in identifying key contributions.
Focus on:
1. Novelty: What is new about this work?
2. Impact: How does this advance the field?
3. Technical depth: What specific technical innovation is introduced?
4. Validation: How is the contribution validated?

Provide a concise but comprehensive answer based on the paper's content."""

        full_prompt = f"{sys_prompt}\n\nQuery: {query}\nContext: {context}"
        msgs = [SystemMessage(content=full_prompt), HumanMessage(content=query)]

        response = self.llm.invoke(msgs)
        return response.content.strip()

    def _analyze_methodology(self, query: str, context: str) -> str:
        """Analyze methodology with technical precision."""
        sys_prompt = """You are analyzing the methodology of a scientific paper.
Focus on:
1. Technical approach: What specific methods/algorithms are used?
2. Implementation details: How is the method implemented?
3. Rationale: Why was this approach chosen?
4. Technical soundness: Are the methods appropriate and well-executed?

Be precise about technical details and implementation choices."""

        full_prompt = f"{sys_prompt}\n\nQuery: {query}\nContext: {context}"
        msgs = [SystemMessage(content=full_prompt), HumanMessage(content=query)]

        response = self.api_manager.make_request(self.llm.invoke, msgs)
        return response.content.strip()

    def _analyze_evaluation(self, query: str, context: str) -> str:
        """Analyze evaluation methods and results."""
        sys_prompt = """You are evaluating the evaluation methodology of a scientific paper.
Focus on:
1. Evaluation metrics: What metrics are used and why?
2. Baselines: What comparisons are made?
3. Datasets: Are the evaluation datasets appropriate?
4. Statistical rigor: Is the evaluation statistically sound?
5. Result interpretation: What do the results actually show?

Provide critical analysis of the evaluation quality."""

        full_prompt = f"{sys_prompt}\n\nQuery: {query}\nContext: {context}"
        msgs = [SystemMessage(content=full_prompt), HumanMessage(content=query)]

        response = self.llm.invoke(msgs)
        return response.content.strip()

    def _analyze_architecture(self, query: str, context: str) -> str:
        """Analyze system architecture and design."""
        sys_prompt = """You are analyzing the system architecture described in a scientific paper.
Focus on:
1. System components: What are the main architectural elements?
2. Data flow: How do components interact?
3. Design decisions: Why was this architecture chosen?
4. Scalability: How does the design handle scale?
5. Technical innovation: What architectural innovations are introduced?

Describe the architecture clearly and technically."""

        full_prompt = f"{sys_prompt}\n\nQuery: {query}\nContext: {context}"
        msgs = [SystemMessage(content=full_prompt), HumanMessage(content=query)]

        response = self.api_manager.make_request(self.llm.invoke, msgs)
        return response.content.strip()

    def _analyze_comparison(self, query: str, context: str) -> str:
        """Analyze comparisons with baselines and related work."""
        sys_prompt = """You are analyzing how this paper compares to existing work.
Focus on:
1. Baseline methods: What methods are compared against?
2. Fair comparison: Are comparisons fair and appropriate?
3. Performance gains: What improvements are demonstrated?
4. Limitations: What are the limitations compared to alternatives?
5. Positioning: How does this work fit in the broader field?

Provide balanced analysis of strengths and weaknesses relative to alternatives."""

        full_prompt = f"{sys_prompt}\n\nQuery: {query}\nContext: {context}"
        msgs = [SystemMessage(content=full_prompt), HumanMessage(content=query)]

        response = self.api_manager.make_request(self.llm.invoke, msgs)
        return response.content.strip()

    def _general_scientific_analysis(self, query: str, context: str) -> str:
        """General scientific analysis for uncategorized queries."""
        sys_prompt = """You are a scientific paper analysis expert.
Apply rigorous scientific reasoning to answer the query based on the paper's content.
Consider the broader scientific context and implications.
Be precise, evidence-based, and technically accurate."""

        full_prompt = f"{sys_prompt}\n\nQuery: {query}\nContext: {context}"
        msgs = [SystemMessage(content=full_prompt), HumanMessage(content=query)]

        response = self.api_manager.make_request(self.llm.invoke, msgs)
        return response.content.strip()

    def extract_scientific_keywords(self, query: str) -> list:
        """Extract domain-specific keywords for scientific queries."""
        keywords = []

        # Scientific method keywords
        method_terms = ['method', 'approach', 'algorithm', 'technique', 'framework', 'architecture']
        for term in method_terms:
            if term in query.lower():
                keywords.append(f"scientific_{term}")

        # Evaluation keywords
        eval_terms = ['evaluation', 'performance', 'result', 'experiment', 'dataset', 'baseline']
        for term in eval_terms:
            if term in query.lower():
                keywords.append(f"scientific_{term}")

        # Contribution keywords
        contrib_terms = ['contribution', 'novel', 'innovation', 'advance', 'improvement']
        for term in contrib_terms:
            if term in query.lower():
                keywords.append(f"scientific_{term}")

        # Add general scientific domain marker
        keywords.append("scientific_paper_analysis")

        return keywords if keywords else ["scientific_query"]