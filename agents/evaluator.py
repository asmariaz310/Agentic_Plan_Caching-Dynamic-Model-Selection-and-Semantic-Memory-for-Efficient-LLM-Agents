from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

class EvaluationAgent:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        """Initialize evaluation agent with capable Groq model."""
        self.llm = ChatGroq(model=model_name, temperature=0)

    def evaluate(self, query: str, ground_truth: str, predicted_answer: str) -> bool:
        """Evaluate if the predicted answer is correct compared to ground truth."""
        prompt = f"""Compare the predicted answer with the ground truth for this query.
Query: {query}
Ground Truth: {ground_truth}
Predicted Answer: {predicted_answer}

Is the predicted answer correct? Consider semantic equivalence, not exact string match.
Answer with only 'YES' or 'NO'."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip().upper() == "YES"