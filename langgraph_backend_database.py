import os
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()

# ── SQLite path ───────────────────────────────────────────────────────────────
DB_PATH = "chat_history.db"

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
)

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
        "You are a helpful, friendly, and knowledgeable AI assistant. "
        "Answer clearly and concisely. If you don't know something, say so."
    )
)


# ── Graph node ────────────────────────────────────────────────────────────────
def call_model(state: MessagesState, config: RunnableConfig) -> dict:
    trimmed = trimmer.invoke(state["messages"])
    messages = [SYSTEM_PROMPT] + trimmed
    response = llm.invoke(messages, config)
    return {"messages": [response]}


# ── Build graph with SqliteSaver ──────────────────────────────────────────────
workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# SqliteSaver persists all checkpoints to disk
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(_conn)
chatbot = workflow.compile(checkpointer=memory)


# ── Thread metadata helpers (separate metadata table) ────────────────────────
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
            "thread_id": r[0],
            "title": r[1],
            "created_at": datetime.fromisoformat(r[2]),
        }
        for r in rows
    ]


def load_thread_messages(thread_id: str) -> list[dict]:
    """
    Reconstruct chat history from LangGraph checkpoints stored in SQLite.
    Returns list of {role, content} dicts.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = chatbot.get_state(config)
        if not state or not state.values:
            return []
        messages = state.values.get("messages", [])
        history = []
        for msg in messages:
            role = "assistant" if msg.__class__.__name__ == "AIMessage" else "user"
            history.append({"role": role, "content": msg.content})
        return history
    except Exception:
        return []