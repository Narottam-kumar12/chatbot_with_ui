"""
Lumina - LangGraph chatbot backend

Features:
    - Gemini 2.5 Flash
    - LangGraph StateGraph with tool calling
    - Human-in-the-Loop tool approval (interrupt / resume)
    - Tool rejection handling
    - SQLite persistent checkpoints and thread metadata
    - SQLite human-feedback storage
    - LangSmith tracing
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, ToolMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command

# Existing tools (kept unchanged)
from tool_calculator import calculator
from tool_search import web_search, news_search
from tool_stock import get_stock_price, get_stock_history, compare_stocks
from tool_weather import get_current_weather, get_weather_forecast, compare_weather

load_dotenv()

# ---------------------------------------------------------------------
# LangSmith tracing
# ---------------------------------------------------------------------
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv(
    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv(
    "LANGCHAIN_PROJECT", "lumina-chatbot"
)

# ---------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------
DB_PATH = "chat_history.db"

# ---------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------
ALL_TOOLS = [
    calculator,
    web_search,
    news_search,
    get_stock_price,
    get_stock_history,
    compare_stocks,
    get_current_weather,
    get_weather_forecast,
    compare_weather,
]

# ---------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
)
llm_with_tools = llm.bind_tools(ALL_TOOLS)

# ---------------------------------------------------------------------
# Message trimming
# ---------------------------------------------------------------------
trimmer = trim_messages(
    max_tokens=4096,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# ---------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------
SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are Lumina, a helpful, friendly, and knowledgeable AI assistant. "
        "You have access to these tools:\n"
        "- calculator: evaluate mathematical expressions\n"
        "- web_search: search the web\n"
        "- news_search: search for recent news\n"
        "- get_stock_price: retrieve stock prices\n"
        "- get_stock_history: retrieve historical stock data\n"
        "- compare_stocks: compare stocks\n"
        "- get_current_weather: retrieve current weather\n"
        "- get_weather_forecast: retrieve weather forecasts\n"
        "- compare_weather: compare weather across cities\n\n"
        "Use tools when appropriate, especially for real-time information. "
        "If a user rejects a tool execution, respect that decision and do not "
        "retry the rejected tool in the same response. Be concise and clear."
    )
)

# ---------------------------------------------------------------------
# Graph state: MessagesState plus approval decision
# ---------------------------------------------------------------------
class ChatState(MessagesState):
    approval: Optional[dict]


# ---------------------------------------------------------------------
# Model node
# ---------------------------------------------------------------------
def call_model(state: ChatState, config: RunnableConfig) -> dict:
    """Call Gemini; it may return a final answer or tool calls."""
    trimmed_messages = trimmer.invoke(state["messages"])
    messages = [SYSTEM_PROMPT] + trimmed_messages
    response = llm_with_tools.invoke(messages, config)
    return {"messages": [response]}


# ---------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------
def should_continue(state: ChatState) -> str:
    """Route tool requests to approval; ordinary answers end the graph."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "approval"
    return END


# ---------------------------------------------------------------------
# Human approval node (LangGraph interrupt)
# ---------------------------------------------------------------------
def human_approval(state: ChatState) -> dict:
    """
    Pause execution and surface tool-call details to the frontend.
    The graph resumes with Command(resume={"approved": True/False}).
    """
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    request = {
        "type": "tool_approval",
        "message": "Lumina requests permission to execute these tool(s).",
        "tool_calls": [
            {
                "id": call["id"],
                "name": call["name"],
                "args": call.get("args", {}),
            }
            for call in tool_calls
        ],
    }

    # interrupt() pauses here. When resumed, its return value is
    # the human-provided decision. Keep side effects out of this node.
    decision = interrupt(request)
    return {"approval": decision}


# ---------------------------------------------------------------------
# Approval routing
# ---------------------------------------------------------------------
def approval_router(state: ChatState) -> str:
    decision = state.get("approval") or {}
    if decision.get("approved") is True:
        return "tools"
    return "rejection"


# ---------------------------------------------------------------------
# Rejection node
# ---------------------------------------------------------------------
def reject_tool_calls(state: ChatState) -> dict:
    """
    Each requested tool call needs a matching ToolMessage so the model
    can continue. These messages report rejection; no tool is run.
    """
    last_message = state["messages"][-1]
    rejection_messages = []

    for call in getattr(last_message, "tool_calls", []):
        rejection_messages.append(
            ToolMessage(
                content=(
                    f"The user declined execution of '{call['name']}'. "
                    "The tool was not executed. Respond without using this "
                    "rejected tool, and respect the user's decision."
                ),
                tool_call_id=call["id"],
                name=call["name"],
            )
        )

    return {"messages": rejection_messages}


# ---------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------
tool_node = ToolNode(ALL_TOOLS)

workflow = StateGraph(state_schema=ChatState)
workflow.add_node("model", call_model)
workflow.add_node("approval", human_approval)
workflow.add_node("tools", tool_node)
workflow.add_node("rejection", reject_tool_calls)

workflow.add_edge(START, "model")

workflow.add_conditional_edges(
    "model",
    should_continue,
    {
        "approval": "approval",
        END: END,
    },
)

workflow.add_conditional_edges(
    "approval",
    approval_router,
    {
        "tools": "tools",
        "rejection": "rejection",
    },
)

workflow.add_edge("tools", "model")
workflow.add_edge("rejection", "model")

# ---------------------------------------------------------------------
# SQLite checkpointer and graph compilation
# ---------------------------------------------------------------------
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(_conn)
chatbot = workflow.compile(checkpointer=memory)


# ---------------------------------------------------------------------
# Resume an interrupted graph
# ---------------------------------------------------------------------
def resume_chat(thread_id: str, approved: bool):
    """
    Resume an interrupted approval for the same thread.

    Returns the LangGraph result. Frontend should inspect result["__interrupt__"]
    to determine whether another approval is pending.
    """
    config = {"configurable": {"thread_id": thread_id}}
    return chatbot.invoke(
        Command(resume={"approved": bool(approved)}),
        config=config,
    )


# ---------------------------------------------------------------------
# Thread metadata helpers
# ---------------------------------------------------------------------
def _meta_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_meta (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT 'New chat',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def save_thread_meta(
    thread_id: str,
    title: str,
    created_at: datetime,
) -> None:
    conn = _meta_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO thread_meta (thread_id, title, created_at)
        VALUES (?, ?, ?)
        """,
        (thread_id, title, created_at.isoformat()),
    )
    conn.commit()
    conn.close()


def update_thread_title(thread_id: str, title: str) -> None:
    conn = _meta_conn()
    conn.execute(
        "UPDATE thread_meta SET title = ? WHERE thread_id = ?",
        (title, thread_id),
    )
    conn.commit()
    conn.close()


def delete_thread_meta(thread_id: str) -> None:
    conn = _meta_conn()
    conn.execute(
        "DELETE FROM thread_meta WHERE thread_id = ?",
        (thread_id,),
    )
    conn.commit()
    conn.close()


def load_all_threads() -> list[dict]:
    conn = _meta_conn()
    rows = conn.execute(
        """
        SELECT thread_id, title, created_at
        FROM thread_meta
        ORDER BY created_at DESC
        """
    ).fetchall()
    conn.close()

    return [
        {
            "thread_id": row[0],
            "title": row[1],
            "created_at": datetime.fromisoformat(row[2]),
        }
        for row in rows
    ]


def load_thread_messages(thread_id: str) -> list[dict]:
    """
    Return display-safe user/assistant history from LangGraph checkpoints.
    Internal tool-call and ToolMessage entries are omitted.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = chatbot.get_state(config)

        if not state or not state.values:
            return []

        history = []
        for msg in state.values.get("messages", []):
            cls = msg.__class__.__name__

            if cls == "HumanMessage":
                history.append({"role": "user", "content": msg.content})

            elif cls == "AIMessage":
                if isinstance(msg.content, str) and msg.content.strip():
                    history.append(
                        {"role": "assistant", "content": msg.content}
                    )
                elif isinstance(msg.content, list):
                    text = " ".join(
                        part["text"]
                        for part in msg.content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ).strip()
                    if text:
                        history.append(
                            {"role": "assistant", "content": text}
                        )

            # ToolMessage is internal graph plumbing; not displayed.

        return history

    except Exception as exc:
        print(f"Error loading thread messages: {exc}")
        return []


# ---------------------------------------------------------------------
# Human feedback persistence (used by the upcoming Streamlit update)
# ---------------------------------------------------------------------
def _feedback_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS human_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            message TEXT NOT NULL,
            rating TEXT NOT NULL,
            feedback_text TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def save_human_feedback(
    thread_id: str,
    message: str,
    rating: str,
    feedback_text: str = "",
) -> None:
    """Save thumbs-up/down and optional written feedback to SQLite."""
    if rating not in {"positive", "negative"}:
        raise ValueError("rating must be 'positive' or 'negative'")

    conn = _feedback_conn()
    conn.execute(
        """
        INSERT INTO human_feedback
        (thread_id, message, rating, feedback_text, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            thread_id,
            message,
            rating,
            feedback_text,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
