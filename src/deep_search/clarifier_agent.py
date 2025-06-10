from typing import List
from pydantic import BaseModel
from agents import Agent

class ClarificationData(BaseModel):
    questions: List[str]

CLARIFY_INSTRUCTIONS = """
You are a Research Clarifier. Given a user’s research query, generate exactly 3 clarifying questions 
that will help focus and refine that query. Return only JSON matching the ClarificationData model.
"""

clarifier_agent = Agent(
    name="ClarifierAgent",
    instructions=CLARIFY_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=ClarificationData,
)
