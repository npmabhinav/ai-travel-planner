import os
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
load_dotenv()

# LangChain & LangGraph Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from ddgs import DDGS

# -------- Agent State --------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "Conversation history"]

# -------- Tools --------
@tool
def safe_duckduckgo_search(query: str) -> str:
    """
    DuckDuckGo search that always returns text.
    """
    try:
        results_text = []
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            for r in results:
                if r.get("body"):
                    results_text.append(r["body"])

        if not results_text:
            return "No relevant search results found."

        return "\n".join(results_text)

    except Exception as e:
        return f"Search failed: {e}"

def get_wiki_tool():
    wiki_wrapper = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=500
    )
    return WikipediaQueryRun(api_wrapper=wiki_wrapper)

# -------- Agent Setup --------
def setup_travel_agent(api_key: str):

    # Set API key safely
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        convert_system_message_to_human=True
    )

    tools = [safe_duckduckgo_search, get_wiki_tool()]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = """
    You are an expert AI Travel Agent.

    HARD CONSTRAINTS:
    - NEVER change the user's destination.
    - NEVER change the number of days.
    - NEVER exceed the given budget.
    - NEVER invent cities, countries, or dates.

    STRATEGY:
    1. Understand user preferences.
    2. Use web search for real-time info.
    3. Use Wikipedia for cultural context.
    4. Respect all constraints strictly.
    """

    # -------- Reasoning Node --------
    def planner(state: AgentState):
        system_msg = SystemMessage(content=system_prompt)

        state_messages = state.get("messages", [])

        if not state_messages:
            state_messages = [
                HumanMessage(content="Continue planning the itinerary.")
        ]

        response = llm.invoke([system_msg] + state_messages)

        return {
            "messages": state_messages + [response]
        }






    # -------- Final Node --------
    def final_response(state: AgentState):
        messages = state.get("messages", [])

        # Always extract the first human message (form input)
        base_request = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                base_request = msg
                break

        if base_request is None:
            base_request = HumanMessage(
                content="Generate a complete travel itinerary."
            )

        system_instruction = SystemMessage(
            content="""
                You are an itinerary generator.

                Rules:
                - All trip details are already provided
                - Do NOT ask questions
                - Do NOT request clarification
                - Do NOT explain assumptions
                - Generate a detailed, day-wise itinerary
                - Respect destination, dates, budget, travelers, and style
                - Output only the itinerary in Markdown
                """
            )

        response = llm.invoke([system_instruction, base_request])

        return {
         "messages": [response]
        }



    # -------- LangGraph --------
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("final", final_response)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "final")
    graph.add_edge("final", END)

    return graph.compile()

