import json
import os
import time
import hashlib
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from agents.sanitization import SanitizationAgent
from core.rate_limiter import GroqAPIManager

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_EMBEDDING = True
except ImportError:
    HAS_EMBEDDING = False
    cosine_similarity = None


class PlanCache:
    def __init__(self, small_model_name="groq/compound-mini", similarity_threshold: float = 0.82):
        """Initialize cache with lightweight Groq model for keyword extraction and semantic lookup."""
        self.entries = []
        self.keyword_index = {}
        self.llm_cache = {}  # Cache for LLM responses
        self.small_lm = ChatGroq(model=small_model_name, temperature=0)
        self.sanitizer = SanitizationAgent()
        self.api_manager = GroqAPIManager(model_name=small_model_name)
        self.similarity_threshold = similarity_threshold
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') if HAS_EMBEDDING else None
        # Cache hit/miss tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.llm_cache_hits = 0
        self.llm_cache_misses = 0
        self.semantic_hits = 0
        self.semantic_misses = 0

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _encode(self, text: str):
        """Encode text into an embedding vector if supported."""
        if not self.embedding_model:
            return None
        try:
            return self.embedding_model.encode(text)
        except Exception:
            return None

    def _get_similarity(self, emb1, emb2) -> float:
        """Compute cosine similarity or fallback to exact text match."""
        if emb1 is None or emb2 is None:
            return 1.0 if emb1 == emb2 else 0.0
        try:
            return float(cosine_similarity([emb1], [emb2])[0][0])
        except Exception:
            return 0.0

    def _cached_llm_call(self, messages: list) -> str:
        cache_key = self._get_cache_key(str(messages))
        if cache_key in self.llm_cache:
            self.llm_cache_hits += 1
            return self.llm_cache[cache_key]

        self.llm_cache_misses += 1
        retries = 4  # Increased from 3 to 4 for better accuracy under rate limiting
        for i in range(retries):
            try:
                response = self.api_manager.make_request(self.small_lm.invoke, messages)
                self.llm_cache[cache_key] = response.content
                return response.content
            except Exception as e:
                wait_time = (i + 1) * 10
                print(f"[Retry {i+1}/{retries}] Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise Exception(f"Max retries ({retries}) exceeded")

    def extract_keyword(self, query: str) -> str:
        """Extract task keyword from query using LLM with domain awareness."""
        query_lower = query.lower()

        # Scientific paper analysis keywords
        if any(word in query_lower for word in ['contribution', 'main finding', 'key result', 'propose', 'introduce']):
            return 'scientific_contribution'
        elif any(word in query_lower for word in ['method', 'approach', 'technique', 'algorithm', 'framework']):
            return 'scientific_methodology'
        elif any(word in query_lower for word in ['evaluate', 'performance', 'result', 'experiment', 'dataset']):
            return 'scientific_evaluation'
        elif any(word in query_lower for word in ['architecture', 'system', 'design', 'structure']):
            return 'scientific_architecture'
        elif any(word in query_lower for word in ['compare', 'baseline', 'versus', 'vs', 'better than']):
            return 'scientific_comparison'

        # Math keywords
        if any(word in query_lower for word in ['calculate', 'compute', 'solve', 'what is']):
            if 'ratio' in query_lower or 'percentage' in query_lower:
                return 'ratio_calculation'
            elif 'average' in query_lower or 'mean' in query_lower:
                return 'average_calculation'
            elif any(op in query for op in ['+', '-', '*', '/']):
                return 'arithmetic'
            else:
                return 'math_calculation'

        elif any(query_lower.startswith(word) for word in ['capital', 'who', 'what', 'where', 'when']):
            return 'factual_question'

        else:
            # Fallback to LLM for complex queries
            prompt = f"""Extract a short keyword (2-3 words) that captures the core task type of this query.
Examples: "math calculation", "factual question", "ratio calculation", "scientific analysis"

Query: {query}
Keyword:"""

            messages = [
                SystemMessage(content="You are a keyword extraction expert."),
                HumanMessage(content=prompt)
            ]

            response = self._cached_llm_call(messages)
            return response.strip().lower().replace(' ', '_')

    def lookup(self, query: str, keyword: str) -> dict:
        """Lookup cached template by semantic similarity and keyword relevance."""
        query_embedding = self._encode(query)
        best_entry = None
        best_score = -1.0

        if query_embedding is None:
            # Fall back to exact keyword match when embeddings are unavailable.
            candidates = self.keyword_index.get(keyword, [])
            if candidates:
                best_entry = candidates[0]
                best_score = 1.0
        else:
            for entry in self.entries:
                score = self._get_similarity(query_embedding, entry["query_embedding"])
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_entry is None or best_score < self.similarity_threshold:
            self.cache_misses += 1
            self.semantic_misses += 1
            return {
                "template": None,
                "confidence": best_score if best_score >= 0 else 0.0,
                "entry": None
            }

        self.cache_hits += 1
        self.semantic_hits += 1
        return {
            "template": best_entry["template"],
            "confidence": best_score,
            "entry": best_entry
        }

    def add_cache_entry(self, query: str, keyword: str, template: dict, execution_log: list):
        """Store a cache entry with semantic metadata and intermediate reasoning."""
        query_embedding = self._encode(query)
        template_embedding = self._encode(json.dumps(template)) if self.embedding_model else None
        entry = {
            "keyword": keyword,
            "query_text": query,
            "query_embedding": query_embedding,
            "template": template,
            "template_embedding": template_embedding,
            "execution_log": execution_log,
            "stored_at": time.time()
        }
        self.entries.append(entry)
        self.keyword_index.setdefault(keyword, []).append(entry)

    def generate_template(self, execution_log: list, keyword: str, query: str) -> dict:
        """Generate reusable template from execution log and add it to the semantic cache."""
        sanitized_log = self.sanitizer.sanitize_log(execution_log)
        template = self.sanitizer.generate_template(sanitized_log)
        self.add_cache_entry(query, keyword, template, execution_log)
        return template

    def get_cache_statistics(self) -> dict:
        """Get comprehensive cache hit rate and statistics."""
        template_total = self.cache_hits + self.cache_misses
        llm_total = self.llm_cache_hits + self.llm_cache_misses
        semantic_total = self.semantic_hits + self.semantic_misses

        return {
            "template_cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "total": template_total,
                "hit_rate": (self.cache_hits / template_total * 100) if template_total > 0 else 0.0
            },
            "semantic_cache": {
                "hits": self.semantic_hits,
                "misses": self.semantic_misses,
                "total": semantic_total,
                "hit_rate": (self.semantic_hits / semantic_total * 100) if semantic_total > 0 else 0.0
            },
            "llm_response_cache": {
                "hits": self.llm_cache_hits,
                "misses": self.llm_cache_misses,
                "total": llm_total,
                "hit_rate": (self.llm_cache_hits / llm_total * 100) if llm_total > 0 else 0.0
            },
            "overall": {
                "total_hits": self.cache_hits + self.llm_cache_hits,
                "total_requests": template_total + llm_total,
                "overall_hit_rate": ((self.cache_hits + self.llm_cache_hits) / (template_total + llm_total) * 100)
                                   if (template_total + llm_total) > 0 else 0.0
            }
        }

    def reset_cache_statistics(self):
        """Reset cache hit/miss counters while keeping cached data."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.llm_cache_hits = 0
        self.llm_cache_misses = 0