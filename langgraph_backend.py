import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
)

# ── Message trimmer (keeps last ~10 messages to stay within context) ──────────
trimmer = trim_messages(
    max_tokens=4096,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful, friendly, and knowledgeable AI assistant. "
        "Answer clearly and concisely. If you don't know something, say so."
    )
)


# ── Graph node ────────────────────────────────────────────────────────────────
def call_model(state: MessagesState, config: RunnableConfig) -> dict:
    """Trim history, prepend system prompt, invoke the LLM."""
    trimmed = trimmer.invoke(state["messages"])
    messages = [SYSTEM_PROMPT] + trimmed
    response = llm.invoke(messages, config)
    return {"messages": [response]}


# ── Build graph ───────────────────────────────────────────────────────────────
workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
chatbot = workflow.compile(checkpointer=memory)