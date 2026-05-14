from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tools.agent_tools import AGENT_TOOLS, calculate
from core.rate_limiter import GroqAPIManager
import json

class ActorAgent:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        """Initialize actor with Groq model and rate limiting."""
        # The actor binds to tools - using Groq for tool calling
        self.llm = ChatGroq(model=model_name, temperature=0).bind_tools(AGENT_TOOLS)
        self.direct_llm = ChatGroq(model=model_name, temperature=0)  # For direct responses
        self.api_manager = GroqAPIManager(requests_per_minute=25)  # Conservative limit

    def act(self, plan_instruction: str) -> str:
        """Executes a step in the plan using tools if necessary."""
        # First try direct execution for simple math
        if any(word in plan_instruction.lower() for word in ['calculate', 'compute', 'solve', 'what is']):
            # Try to extract and execute math directly
            try:
                # Simple pattern matching for basic math
                import re
                # Look for expressions like "2 + 2", "10 * 5", etc.
                math_patterns = [
                    r'(\d+)\s*\+\s*(\d+)',  # addition
                    r'(\d+)\s*\-\s*(\d+)',  # subtraction
                    r'(\d+)\s*\*\s*(\d+)',  # multiplication
                    r'(\d+)\s*\/\s*(\d+)',  # division
                    r'sqrt\((\d+)\)',       # square root
                ]
                
                for pattern in math_patterns:
                    match = re.search(pattern, plan_instruction)
                    if match:
                        if len(match.groups()) == 2:
                            a, b = map(float, match.groups())
                            if '+' in plan_instruction:
                                result = a + b
                            elif '-' in plan_instruction:
                                result = a - b
                            elif '*' in plan_instruction:
                                result = a * b
                            elif '/' in plan_instruction and b != 0:
                                result = a / b
                            return str(result)
                        elif 'sqrt' in plan_instruction:
                            import math
                            return str(math.sqrt(float(match.group(1))))
            except:
                pass
        
        # For more complex tasks, use the LLM
        sys_prompt = "You are an actor agent. Execute the given instruction. For math problems, calculate the result directly. Be concise."
        msgs = [SystemMessage(content=sys_prompt), HumanMessage(content=plan_instruction)]
        
        response = self.api_manager.make_request(self.direct_llm.invoke, msgs)
        return response.content.strip()