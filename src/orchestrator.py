from src.agent_tools import query_telemetry_db, fetch_corridor_conditions, search_compliance_sop
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. SETUP & PATH RESOLUTION
# ==========================================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[0]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ==========================================
# 2. FACTORY INITIALIZATION: AGENT REASONER LLM
# ==========================================


AGENT_LLM_SETTING = os.getenv("Agent_llm", "OLLAMA").strip().upper()

if AGENT_LLM_SETTING == "OPENAI":
    print("🤖 Brain Mode: Utilizing Cloud OpenAI Reasoner (gpt-4o)...")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

elif AGENT_LLM_SETTING == "DEEPSEEK":
    print("🐳 Brain Mode: Utilizing Flagship DeepSeek Cloud Reasoner (deepseek-v4-pro)...")
    from langchain_openai import ChatOpenAI

    # Fully updated to match 2026 DeepSeek API parameters and endpoint contracts
    llm = ChatOpenAI(
        # deepseek-v4-flash, deepseek-v4-pro
        model="deepseek-v4-flash",
        temperature=0,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",     # Fixed connection string url endpoint
        # Gives the deep reasoner plenty of output runway
        max_tokens=2048,
        # extra_body={
        #     "thinking": {"type": "enabled"},              # Activates DeepSeek Deep-Thinking mode
        #     "reasoning_effort": "high"                     # Drives maximal reasoning depth for logic maps
        # }
    )

elif AGENT_LLM_SETTING == "GROQ":
    print("🐳 Brain Mode: Utilizing Flagship GROQ_API_KEY Cloud Reasoner (qwen/qwen3.8-27b)...")
    from langchain_groq import ChatGroq

    # Fully updated to match 2026 DeepSeek API parameters and endpoint contracts
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=groq_api_key,
                   model_name="qwen/qwen3.8-27b", temperature=0, max_tokens=800)


else:  # FALLBACK / DEFAULT RUNNER MODE
    print("🤗 Brain Mode: Local Fallback Activated. Binding Local Ollama (qwen2.5:7b)...")
    # from langchain_community.chat_models import ChatOllama
    from langchain_ollama import ChatOllama
   # llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_predict=1024)

    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0,
        model_kwargs={
            "num_predict": 1024  # Correct way to pass Ollama-specific parameters
        }
    )

fde_tools = [query_telemetry_db,
             fetch_corridor_conditions, search_compliance_sop]
llm_with_tools = llm.bind_tools(fde_tools)

# ==========================================
# 3. GRAPH ARCHITECTURE ASSEMBLY
# ==========================================


def reasoning_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


print("⚙️ Compiling LangGraph FDE Orchestrator...")
graph_builder = StateGraph(AgentState)
graph_builder.add_node("reasoner", reasoning_node)
graph_builder.add_node("tools", ToolNode(fde_tools))

# adding edge between start and reasoner.
graph_builder.add_edge(START, "reasoner")
# ReAct loop conditioning between reasoner and tool_node
graph_builder.add_conditional_edges("reasoner", tools_condition)
# add edge between reasoner and tool
graph_builder.add_edge("tools", "reasoner")

# compile the graph Memory Saver needed when every time agent runs it remebers the previous run
fde_agent = graph_builder.compile(checkpointer=MemorySaver())

# fde use fde_agent in streamlit


# ==========================================
# 4. CHAT LOOP TESTING PANEL
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*100)
    print("🚀 FDE Supply Chain Orchestrator State Machine Online Using Langgraph::Similar to Amazon Strands ..Stateful Agentic Framework")
    print(
        f"   Configured Execution: [LLM: {AGENT_LLM_SETTING}] -> [Embeddings: {os.getenv('Embeddings_model', 'LOCAL')}]")
    print("="*55 + "\n")

    # Load the business-structured system prompt from the external file
    prompt_path = project_root / "src" / "prompt-templates" / "systemPrompt.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_instructions = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find {prompt_path}")
        system_instructions = "You are a helpful AI assistant."  # Basic fallback

    system_prompt = SystemMessage(content=system_instructions)

    thread_config = {"configurable": {"thread_id": "production_test_1"}}
   # commented for groq constraints fde_agent.invoke({"messages": [system_prompt]}, config=thread_config)

    while True:
        user_input = input("\nDispatcher > ")
        if user_input.lower() in ['exit', 'quit']:
            break
         # Check if this is the very first message in the thread
        current_state = fde_agent.get_state(thread_config)

        if not current_state.values or not current_state.values.get("messages"):
            # First turn: Inject both the system prompt rules AND the user message together
            input_messages = [system_prompt, HumanMessage(content=user_input)]
        else:
            # Subsequent turns: Append only the new user message
            input_messages = [HumanMessage(content=user_input)]

        events = fde_agent.stream(
            {"messages": input_messages}, config=thread_config, stream_mode="updates")
        for event in events:
            for node_name, node_state in event.items():
                if node_name == "tools":
                    print(
                        "   [System] 🔄 Retrieving external data elements via ToolNode...")
                elif node_name == "reasoner":
                    latest_msg = node_state["messages"][-1]
                    if latest_msg.content:
                        print(f"\n🤖 FDE Agent:\n{latest_msg.content}")
