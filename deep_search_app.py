from deep_search.manager_agent import manager_agent
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
        questions = clar.final_output
        logger.info(f"Clarifying questions: {questions}")
        # qtext = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        qtext = questions
        return qtext, gr.update(visible=True), questions

    # Phase 2: Full pipeline
    # 1) Bundle user answers for planner
    # answered = [
    #     f"{i+1}. {state[i]} answered: {ans.strip()}"
    #     for i, ans in enumerate(answers.splitlines())
    # ]
    answered = [
        f"{i+1}. answered: {ans.strip()}"
        for i, ans in enumerate(answers.splitlines())
    ]
    logger.info(f"User answers: {answered}")

    planner_input = f"Using the following context: Original query: {query}\nClarifications:\n" + "\n".join(answered)
    planner_input += "\nCreate a focused search plan using the planner tool"

    logger.info(f"Planner input: {planner_input}")

    # 2) Generate search plan
    plan_res = await Runner.run(manager_agent, planner_input)
    # searches = plan_res.final_output.searches
    searches = plan_res.final_output

    logger.info(f"Generated search plan: {searches}")

    # 3) Run each search and collect summaries
    summaries = []
    for item in searches:
        # search_res = await Runner.run(manager_agent, item.query)
        search_res = await Runner.run(manager_agent, item)
        summaries.append(str(search_res.final_output))

    # 4) Write the full report
    writer_input = f"Original query: {query}\nSummaries: {summaries}"
    logger.info(f"Writing report with input: {writer_input}")
    write_res = await Runner.run(manager_agent, writer_input)
    report_data = write_res.final_output
    logger.info(f"Generated report: {report_data}")
    # report_md = report_data.markdown_report
    report_md = report_data


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

