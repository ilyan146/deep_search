from deep_search.manager_agent import manager_agent
from deep_search.clarifier_agent import ClarificationData
from deep_search.planner_agent import WebSearchPlan, planner_agent
from deep_search.writer_agent import ReportData
from dotenv import load_dotenv
from agents import Runner, set_default_openai_api, set_default_openai_client, set_tracing_disabled
from loguru import logger
import gradio as gr
from openai import AsyncAzureOpenAI
from deep_search.client import AzureAIClient

# Load environment variables (e.g., SENDGRID_API_KEY)
load_dotenv(override=True)

# Set client settings
client = AzureAIClient()
openai_client = AsyncAzureOpenAI(
    azure_endpoint=client.azure_endpoint,
    api_version=client.api_version,
    azure_ad_token_provider=client.token_provider,
)
set_default_openai_client(openai_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

async def run(query: str, answers: str, state: list[str]):
    """
    Two-phase Gradio workflow:
      1) If `state` is empty, ask clarifying questions.
      2) Once answered, plan → search → write → email → report.
    """
    # Phase 1: Clarify
    if not state:
        clar = await Runner.run(manager_agent, query)
        # questions = clar.final_output.questions
        # Attempt parsing of the manager agents output
        clar_agent_output = clar.final_output
        parsed_questions = ClarificationData.model_validate_json(clar_agent_output)
        questions = parsed_questions.questions
        logger.debug(f"Clarifier agent output type: {type(parsed_questions)}")
        logger.debug(f"Clarification agent output .questions: {questions}")

        qtext = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        return qtext, gr.update(visible=True), questions

    # Phase 2: Full pipeline
    # 1) Bundle user answers for planner
    answered = [
        f"{i+1}. {state[i]} answered: {ans.strip()}"
        for i, ans in enumerate(answers.splitlines())
    ]

    planner_input = f"Now with the clarifiaction input, STAGE: **Plan** - please create a plan using the planner tool only to make WebSearchPlan and DO NOT use the search tool at this stage and make sure the results are in JSON format, this is a must requirement.\n\nOriginal query: {query}\nClarifications:\n\n"+ "\n".join(answered)

    logger.info(f"Input for the planner agent tool: {planner_input}")
    # 2) Generate search plan
    plan_res = await Runner.run(manager_agent, planner_input)

    plan_res_output = plan_res.final_output
    logger.debug(f"Planner agent tools output: {plan_res_output}")

    # Parsing back to WebSearchPlan model
    parsed_plan = WebSearchPlan.model_validate_json(plan_res_output)
    logger.debug(f"Planner agent output after loading to pydantic Type: {type(parsed_plan)}")
    # searches = plan_res.final_output.searches
    searches = parsed_plan.searches
    # searches = plan_res.final_output
    logger.debug(f"Generated search plan from .searches in the pydantic object: {searches}")

    # 3) Run each search and collect summaries
    summaries = []
    for item in searches:
        search_prompt = f"STAGE: **Search** - Please carry out searches with the search tool available to you for the query : {item.query}\n\nDo not use the clarifier tool here. Return a summary for the search results that you have obtained."
        # search_res = await Runner.run(manager_agent, item.query)
        logger.info(f"Running search for query selected: {search_prompt}")
        search_res = await Runner.run(manager_agent, search_prompt)
        logger.success(f"Search completed for query: {item.query}")
        logger.debug(f"Search result: {search_res.final_output}")
        summaries.append(str(search_res.final_output))

    # 4) Write the full report 
    # writer_input = f"Original query: {query}\nSummaries: {summaries}"
    writer_input = (
    f"STAGE: WRITING - Please use the writer tool, this is a must requirement, to create a report in JSON format and NOT markdown format.\n\n"
    f"Original query: {query}\n"
    f"Search summaries: {summaries}"
    )
    logger.info(f"Writing report with input: {writer_input}")
    write_res = await Runner.run(manager_agent, writer_input)
    report_data = write_res.final_output
    logger.info(f"Generated report: {report_data}")
    report_data_res = ReportData.model_validate_json(report_data)
    logger.debug(f"Report data: {report_data_res}")
    # report_md = report_data.markdown_report
    report_md = report_data_res.markdown_report
    logger.debug(f"Markdown report: {report_md}")

    return report_md, gr.update(visible=False), []


# Gradio UI
with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Deep Research")
    query_textbox = gr.Textbox(label="What topic would you like to research?")
    answers_box = gr.Textbox(
        label="Answer clarifying questions (one per line)",
        visible=False,
        lines=3,
        placeholder="1. …\n2. …\n3. …"
    )
    state = gr.State([])
    report = gr.Markdown(label="Output")
    run_button = gr.Button("Run")

    run_button.click(
        fn=run,
        inputs=[query_textbox, answers_box, state],
        outputs=[report, answers_box, state]
    )

ui.launch(share=False)

