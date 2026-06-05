import time
import uuid
from datetime import datetime

import streamlit as st
from langchain_core.messages import HumanMessage

from langgraph_backend import chatbot

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  – clean dark sidebar, ChatGPT-like feel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    [data-testid="stAppViewContainer"] { background: #0f0f0f; }
    [data-testid="stMain"] { background: #0f0f0f; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #171717 !important;
        border-right: 1px solid #2a2a2a;
        min-width: 260px !important;
        max-width: 260px !important;
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* New Chat button */
    div[data-testid="stSidebar"] div.new-chat-btn > button {
        background: #2a2a2a !important;
        border: 1px solid #3a3a3a !important;
        border-radius: 8px !important;
        color: #fff !important;
        font-weight: 600 !important;
        width: 100% !important;
        padding: 0.55rem 1rem !important;
        margin-bottom: 1rem;
        transition: background 0.15s;
    }
    div[data-testid="stSidebar"] div.new-chat-btn > button:hover {
        background: #333 !important;
    }

    /* Thread list buttons */
    div[data-testid="stSidebar"] div.thread-btn > button {
        background: transparent !important;
        border: none !important;
        border-radius: 6px !important;
        color: #ccc !important;
        text-align: left !important;
        width: 100% !important;
        padding: 0.45rem 0.75rem !important;
        font-size: 0.85rem !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        transition: background 0.12s;
    }
    div[data-testid="stSidebar"] div.thread-btn > button:hover {
        background: #252525 !important;
        color: #fff !important;
    }
    div[data-testid="stSidebar"] div.thread-btn-active > button {
        background: #2d2d2d !important;
        color: #fff !important;
        font-weight: 600 !important;
    }

    /* ── Chat area ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
    }
    [data-testid="stChatMessageContent"] p { color: #e0e0e0 !important; }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background: #1e1e1e !important;
        border: 1px solid #333 !important;
        color: #e0e0e0 !important;
        border-radius: 12px !important;
    }

    /* Title */
    h1 { color: #f0f0f0 !important; font-size: 1.4rem !important; }

    /* Thin section label */
    .section-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #555 !important;
        padding: 0.6rem 0.75rem 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def initialize_session() -> None:
    """
    all_threads  : dict[thread_id → {title, created_at, messages[]}]
    active_thread: currently viewed thread_id
    """
    if "all_threads" not in st.session_state:
        first_id = str(uuid.uuid4())
        st.session_state.all_threads = {
            first_id: {
                "title": "New chat",
                "created_at": datetime.now(),
                "messages": [],
            }
        }
        st.session_state.active_thread = first_id

# ─────────────────────────────────────────────────────────────────────────────
# Thread helpers
# ─────────────────────────────────────────────────────────────────────────────

def current_thread() -> dict:
    return st.session_state.all_threads[st.session_state.active_thread]


def create_new_thread() -> None:
    tid = str(uuid.uuid4())
    st.session_state.all_threads[tid] = {
        "title": "New chat",
        "created_at": datetime.now(),
        "messages": [],
    }
    st.session_state.active_thread = tid
    st.rerun()


def switch_thread(tid: str) -> None:
    st.session_state.active_thread = tid
    st.rerun()


def delete_thread(tid: str) -> None:
    del st.session_state.all_threads[tid]
    if st.session_state.active_thread == tid:
        if st.session_state.all_threads:
            st.session_state.active_thread = next(iter(st.session_state.all_threads))
        else:
            create_new_thread()
            return
    st.rerun()


def auto_title(user_text: str) -> str:
    """Generate a short title from the first user message."""
    words = user_text.strip().split()
    title = " ".join(words[:6])
    return title[:40] + ("…" if len(title) > 40 else "")

# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.active_thread}}


def get_bot_response(user_input: str) -> str:
    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=get_config(),
    )
    return response["messages"][-1].content


def stream_response(response_text: str) -> None:
    placeholder = st.empty()
    streamed = ""
    for word in response_text.split():
        streamed += word + " "
        placeholder.markdown(streamed + "▌")
        time.sleep(0.03)
    placeholder.markdown(streamed.strip())

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        # App name
        st.markdown("### 🤖 LangGraph Chat")
        st.markdown("---")

        # New Chat button
        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        if st.button("＋  New Chat", key="new_chat_btn", use_container_width=True):
            create_new_thread()
        st.markdown("</div>", unsafe_allow_html=True)

        # Thread list header
        st.markdown('<div class="section-label">Recent conversations</div>', unsafe_allow_html=True)

        # Sort newest first
        sorted_threads = sorted(
            st.session_state.all_threads.items(),
            key=lambda x: x[1]["created_at"],
            reverse=True,
        )

        for tid, meta in sorted_threads:
            is_active = tid == st.session_state.active_thread
            btn_class = "thread-btn-active" if is_active else "thread-btn"
            label = ("💬 " if is_active else "   ") + meta["title"]

            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f'<div class="{btn_class}">', unsafe_allow_html=True)
                if st.button(label, key=f"thread_{tid}", use_container_width=True):
                    switch_thread(tid)
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                if st.button("✕", key=f"del_{tid}", help="Delete this chat"):
                    delete_thread(tid)

        # Footer
        st.markdown("---")
        st.caption("Built with LangGraph + Gemini + Streamlit")

# ─────────────────────────────────────────────────────────────────────────────
# Chat area
# ─────────────────────────────────────────────────────────────────────────────

def display_chat_history() -> None:
    thread = current_thread()
    if not thread["messages"]:
        st.markdown(
            "<div style='text-align:center; color:#444; margin-top:6rem; font-size:1.1rem;'>"
            "Start a conversation…</div>",
            unsafe_allow_html=True,
        )
        return
    for msg in thread["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def handle_user_input(user_input: str) -> None:
    thread = current_thread()

    # Auto-title on first message
    if not thread["messages"]:
        thread["title"] = auto_title(user_input)

    # User bubble
    thread["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant bubble
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = get_bot_response(user_input)
            except Exception as e:
                response = f"⚠️ Error: {e}"
        stream_response(response)

    thread["messages"].append({"role": "assistant", "content": response})

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    initialize_session()
    render_sidebar()

    thread = current_thread()
    st.markdown(f"### {thread['title']}")

    display_chat_history()

    user_input = st.chat_input("Type your message…")
    if user_input:
        handle_user_input(user_input)


if __name__ == "__main__":
    main()