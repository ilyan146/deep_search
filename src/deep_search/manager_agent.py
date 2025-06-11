from agents import Agent
from deep_search.clarifier_agent import clarifier_agent
from deep_search.planner_agent import planner_agent
from deep_search.search_agent import search_agent
from deep_search.writer_agent import writer_agent
# from email_agent import send_email

# MANAGER_INSTRUCTIONS = """
# You are the Research Manager orchestrator.

# 1) **Clarify.** Ask the user 3 clarifying questions about their original query and return the output in tools output type JSON format, please make sure it is only a JSON formatted output.
# 2) **Plan.** Once the user answers, call planner tool and this is a must tool to use here, and respond in the JSON format matching the planner tools output_type, make sure your output is only in JSON format.
# 3) **Search.** Then after receiving a list of searches, for each item in that plan, call search tool to get summaries.
# 4) **Write.** Pass the collected summaries to the writer tool to produce a full report.

# Make sure each handoff is explicit (you invoke the appropriate tool with the right data).
# """

MANAGER_INSTRUCTIONS = """
You are the Research Manager orchestrator.

1) **Clarify.** Ask the user 3 clarifying questions about their original query and return the output in tools output type JSON format, please make sure it is only a JSON formatted output.

2) **Plan.** When the user provides answers to clarifying questions OR explicitly asks for a search plan, you MUST use the planner tool. Never use the search tool directly at this stage. Respond in the JSON format matching the planner tool's output_type.

3) **Search.** Only AFTER you have a complete search plan with specific queries from the planner tool, use the search tool for EACH query in that plan.

4) **Write.** Pass the collected summaries to the writer tool to produce a full report in the JSON format as per the output_type that the writer tool produces, the result should only be in JSON format that matches the ReportData output_type and Not in markdown format, this is a must requirement.

Important: Never skip stages or tools. Each handoff must be explicit.
"""

manager_agent = Agent(
    name="ManagerAgent",
    instructions=MANAGER_INSTRUCTIONS,
    tools=[
        clarifier_agent.as_tool(
            tool_name="clarifier",
            tool_description="Generate 3 clarifying questions for the query"
        ),
        planner_agent.as_tool(
            tool_name="planner",
            tool_description="Create a focused search plan for the given query and clarifications"
        ),
        search_agent.as_tool(
            tool_name="search",
            tool_description="Summarize web search results for a given term"
        ),
        writer_agent.as_tool(
            tool_name="writer",
            tool_description="Produce a cohesive markdown report from search summaries"
        ),
        # send_email,  # function tool for sending the report via email
    ],
    model="gpt-4o-mini",
)
