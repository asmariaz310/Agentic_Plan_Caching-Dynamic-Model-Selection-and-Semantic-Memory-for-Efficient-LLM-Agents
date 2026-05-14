from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import re
import json

class SanitizationAgent:
    def __init__(self, model_name="groq/compound-mini"):
        """Initialize sanitization agent with lightweight Groq model."""
        self.llm = ChatGroq(model=model_name, temperature=0)

    def sanitize_log(self, execution_log: list) -> list:
        """Sanitize execution log to remove sensitive information and create reusable template."""
        # Rule-based filtering first
        sanitized = []
        for entry in execution_log:
            # Remove verbose reasoning, keep key actions and decisions
            if "verbose" in entry.get("type", "").lower():
                continue
            # Remove specific numbers, names, etc.
            content = entry.get("content", "")
            content = re.sub(r'\b\d{4}\b', '<YEAR>', content)  # Fiscal years
            content = re.sub(r'\$\d+(?:\.\d+)?', '<AMOUNT>', content)  # Monetary amounts
            content = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '<PERSON>', content)  # Names
            content = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Corporation\b', '<COMPANY>', content)  # Company names

            sanitized.append({"type": entry.get("type"), "content": content})

        return sanitized

    def generate_template(self, sanitized_log: list) -> dict:
        """Generate reusable template from sanitized log using LLM."""
        log_str = "\n".join([f"{entry['type']}: {entry['content']}" for entry in sanitized_log])

        prompt = f"""Convert this execution log into a reusable template for similar tasks.
Remove all specific details and replace with placeholders. Focus on the workflow pattern.

Execution Log:
{log_str}

Return a JSON template with 'workflow' array containing steps like:
{{"workflow": [
  {{"type": "message", "content": "General planning step"}},
  {{"type": "output", "content": "General action result"}},
  {{"type": "answer", "content": "General final answer"}}
]}}

Output only valid JSON:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            return json.loads(response.content.strip())
        except:
            # Fallback template
            return {"workflow": [
                {"type": "message", "content": "Analyze the query and plan actions"},
                {"type": "output", "content": "Execute planned actions"},
                {"type": "answer", "content": "Provide final answer"}
            ]}