import time
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from langgraph_backend import chatbot


# ── Session helpers ───────────────────────────────────────────────────────────

def initialize_session() -> None:
    """Initialize chat history and per-session thread ID."""
    if "message_history" not in st.session_state:
        st.session_state.message_history = []
    if "thread_id" not in st.session_state:
        # Unique thread per browser session → separate LangGraph memory
        st.session_state.thread_id = str(uuid.uuid4())


def get_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


# ── Chat helpers ──────────────────────────────────────────────────────────────

def display_chat_history() -> None:
    """Render all previous messages."""
    for message in st.session_state.message_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def get_bot_response(user_input: str) -> str:
    """Invoke LangGraph chatbot and return the assistant reply."""
    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=get_config(),
    )
    return response["messages"][-1].content


def stream_response(response_text: str) -> None:
    """Word-by-word typing animation with a blinking cursor."""
    placeholder = st.empty()
    streamed_text = ""
    for word in response_text.split():
        streamed_text += word + " "
        placeholder.markdown(streamed_text + "▌")
        time.sleep(0.03)
    placeholder.markdown(streamed_text.strip())


def handle_user_input(user_input: str) -> None:
    """Full chat round-trip: display user msg → get response → stream it."""
    # --- user bubble ---
    st.session_state.message_history.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- assistant bubble ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = get_bot_response(user_input)
            except Exception as e:
                response = f"⚠️ Something went wrong: {e}"
        stream_response(response)

    st.session_state.message_history.append(
        {"role": "assistant", "content": response}
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Settings")
        st.caption(f"Session ID: `{st.session_state.get('thread_id', '…')}`")

        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.message_history = []
            # New thread = fresh LangGraph memory
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        st.divider()
        st.markdown(
            "Built with [LangGraph](https://github.com/langchain-ai/langgraph) "
            "+ [Gemini](https://ai.google.dev/) "
            "+ [Streamlit](https://streamlit.io)"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="LangGraph Chatbot",
        page_icon="🤖",
        layout="centered",
    )

    st.title("🤖 LangGraph + Gemini Chatbot")

    initialize_session()
    render_sidebar()
    display_chat_history()

    user_input = st.chat_input("Type your message…")
    if user_input:
        handle_user_input(user_input)


if __name__ == "__main__":
    main()