"""
langgraph_backend.py
─────────────────────
LangGraph chatbot backend with:
  • Gemini 2.0 Flash LLM
  • SQLite persistent memory (SqliteSaver)
  • LangSmith tracing
  • Tools: calculator, web search, news search, stock price, weather
"""

import os
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# Import all tools
from tool_calculator import calculator
from tool_search     import web_search, news_search
from tool_stock      import get_stock_price, get_stock_history, compare_stocks
from tool_weather    import get_current_weather, get_weather_forecast, compare_weather

load_dotenv()

# ── LangSmith Tracing ─────────────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "lumina-chatbot")

# ── SQLite path ───────────────────────────────────────────────────────────────
DB_PATH = "chat_history.db"

# ── Tool registry ─────────────────────────────────────────────────────────────
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

# ── LLM bound with tools ──────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
)
llm_with_tools = llm.bind_tools(ALL_TOOLS)

# ── Message trimmer ───────────────────────────────────────────────────────────
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
        "You are Lumina, a helpful, friendly, and knowledgeable AI assistant. "
        "You have access to the following tools — use them whenever relevant:\n\n"
        "🧮 calculator       — evaluate any math expression\n"
        "🔍 web_search       — search the web for general knowledge\n"
        "📰 news_search      — find latest news on any topic\n"
        "📈 get_stock_price  — current price & stats for any stock ticker\n"
        "📊 get_stock_history — historical stock performance over a period\n"
        "📉 compare_stocks   — compare multiple stocks side by side\n"
        "🌤️ get_current_weather  — current weather for any city\n"
        "📅 get_weather_forecast — multi-day forecast for any city\n"
        "🌍 compare_weather  — compare weather across multiple cities\n\n"
        "Always use tools for real-time data (weather, stocks, news, math). "
        "Be concise, clear, and friendly."
    )
)


# ── Graph nodes ───────────────────────────────────────────────────────────────

def call_model(state: MessagesState, config: RunnableConfig) -> dict:
    """Invoke LLM with tools. Trims history to fit context window."""
    trimmed  = trimmer.invoke(state["messages"])
    messages = [SYSTEM_PROMPT] + trimmed
    response = llm_with_tools.invoke(messages, config)
    return {"messages": [response]}


def should_continue(state: MessagesState) -> str:
    """Route: if LLM called a tool go to tools node, else end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Build graph ───────────────────────────────────────────────────────────────
tool_node = ToolNode(ALL_TOOLS)

workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("model", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "model")
workflow.add_conditional_edges(
    "model", should_continue, {"tools": "tools", END: END}
)
workflow.add_edge("tools", "model")   # after tools → back to model for final answer

# SqliteSaver: all checkpoints persist to disk
_conn   = sqlite3.connect(DB_PATH, check_same_thread=False)
memory  = SqliteSaver(_conn)
chatbot = workflow.compile(checkpointer=memory)


# ── Thread metadata helpers ───────────────────────────────────────────────────

def _meta_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_meta (
            thread_id   TEXT PRIMARY KEY,
            title       TEXT NOT NULL DEFAULT 'New chat',
            created_at  TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def save_thread_meta(thread_id: str, title: str, created_at: datetime) -> None:
    conn = _meta_conn()
    conn.execute(
        "INSERT OR REPLACE INTO thread_meta (thread_id, title, created_at) VALUES (?, ?, ?)",
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
    conn.execute("DELETE FROM thread_meta WHERE thread_id = ?", (thread_id,))
    conn.commit()
    conn.close()


def load_all_threads() -> list[dict]:
    """Return all threads sorted newest-first."""
    conn = _meta_conn()
    rows = conn.execute(
        "SELECT thread_id, title, created_at FROM thread_meta ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [
        {
            "thread_id":  r[0],
            "title":      r[1],
            "created_at": datetime.fromisoformat(r[2]),
        }
        for r in rows
    ]


def load_thread_messages(thread_id: str) -> list[dict]:
    """
    Reconstruct display-safe chat history from LangGraph SQLite checkpoints.
    Skips internal tool-call/tool-result messages — returns only user & assistant text.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state  = chatbot.get_state(config)
        if not state or not state.values:
            return []

        history = []
        for msg in state.values.get("messages", []):
            cls = msg.__class__.__name__

            if cls == "HumanMessage":
                history.append({"role": "user", "content": msg.content})

            elif cls == "AIMessage":
                # Only display messages that have real text (not pure tool-call stubs)
                if isinstance(msg.content, str) and msg.content.strip():
                    history.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg.content, list):
                    text = " ".join(
                        p["text"] for p in msg.content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ).strip()
                    if text:
                        history.append({"role": "assistant", "content": text})

            # ToolMessage intentionally skipped (internal plumbing)

        return history
    except Exception:
        return []