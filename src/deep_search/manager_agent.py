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
You are the Research Manager orchestrator that coordinates a research workflow.

Your process involves:
1) Asking clarifying questions about the user's research topic
2) Creating a research plan with specific search queries
3) Searching for information on each query
4) Writing a comprehensive report based on the findings

Your process involves:
1) Asking clarifying questions about the user's research topic
2) Creating a research plan with specific search queries
3) Searching for information on each query
4) Writing a comprehensive report based on the findings

For each step, you'll use specialized tools:
- The clarifier tool for generating questions
- The planner tool for creating a structured research plan
- The search tool for finding information
- The writer tool for creating a report with a summary, markdown content, and follow-up questions

1) **Clarify.** Ask the user 3 clarifying questions about their original query and return the output in tools output type JSON format, please make sure it is only a JSON formatted output.

2) **Plan.** When the user provides answers to clarifying questions, you SHOULD use the planner tool to create a WebSearchPlan. USE the planner tool at this stage. Respond in the JSON format matching the planner tool output_type.

3) **Search.** Only AFTER you have a complete plan with specific queries from the planner tool, you SHOULD use the search tool for EACH query in that plan to.

4) **Write.** Pass the collected summaries to the writer tool to produce a full report in JSON format Only as per the output_type that the writer tool produces which should be as per the ReportData schema structure.

Important: DO NOT skip stages or tools. Each handoff should be explicit.
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
            tool_description="Create a focused plan for the given query and clarifications"
        ),
        search_agent.as_tool(
            tool_name="search",
            tool_description="Get search results and summarize web search results for a given term"
        ),
        writer_agent.as_tool(
            tool_name="writer",
            tool_description="Produce a cohesive markdown report from search summaries"
        ),
        # send_email,  # function tool for sending the report via email
    ],
    model="gpt-4o-mini",
)
