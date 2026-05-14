import json
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.rate_limiter import GroqAPIManager

class PlannerAgent:
    def __init__(self, large_model="llama-3.3-70b-versatile", small_model="llama-3.1-8b-instant"):
        """Initialize planner with Groq models and rate limiting."""
        # Using Groq API through environment variables
        self.large_lm = ChatGroq(model=large_model, temperature=0)
        self.small_lm = ChatGroq(model=small_model, temperature=0)
        self.api_manager = GroqAPIManager(requests_per_minute=25)  # Conservative limit

    def large_plan(self, query: str, context: str, past_messages: list) -> str:
        """The expensive, highly capable planner (GPT-4)."""
        sys_prompt = f"""You are an advanced planning agent.
Your task is to decompose the user's query and generate a step-by-step action plan to solve it using the available tools.
If you have enough information, provide the final answer starting with 'FINAL ANSWER:'.
Available context: {context}"""

        msgs = [SystemMessage(content=sys_prompt), HumanMessage(content=query)] + past_messages
        response = self.api_manager.make_request(self.large_lm.invoke, msgs)
        return response.content

    def fast_plan(self, query: str, context: str, past_messages: list) -> str:
        """A faster planner that uses the small model for lower-cost planning."""
        sys_prompt = f"""You are a fast planning agent.
Use the available tools and context to generate a concise plan for the query.
If you have enough information, provide the final answer starting with 'FINAL ANSWER:'.
Available context: {context}"""

        msgs = [SystemMessage(content=sys_prompt), HumanMessage(content=query)] + past_messages
        response = self.api_manager.make_request(self.small_lm.invoke, msgs)
        return response.content

    def small_plan(self, query: str, context: str, past_messages: list, cached_template: dict) -> tuple[str, bool]:
        """The fast, cheap planner that adapts a cached template. Returns (response, fallback_needed)."""
        # If the template seems to be failing (e.g. error from tool, or we did too many steps without finding an answer),
        # we trigger a fallback to the large model.
        if len(past_messages) > 6:
            return "Fallback required due to loop length.", True

        sys_prompt = f"""You are a fast planning agent. You have a cached reference template for a similar task.
Adapt the reference template to the current query and context.
If the cached template doesn't seem to apply or you encounter unexpected errors, respond with exactly 'FALLBACK'.
If you have enough information to answer the query, provide the final answer starting with 'FINAL ANSWER:'.

Cached Template:
{json.dumps(cached_template, indent=2)}

Current Context: {context}"""

        msgs = [SystemMessage(content=sys_prompt), HumanMessage(content=query)] + past_messages
        response = self.api_manager.make_request(self.small_lm.invoke, msgs)
        content = response.content.strip()

        if content == "FALLBACK":
            return "Fallback triggered.", True

        return content, False