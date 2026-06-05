import time
import uuid
from datetime import datetime

import streamlit as st
from langchain_core.messages import HumanMessage

from langgraph_backend_database import (
    chatbot,
    save_thread_meta,
    update_thread_title,
    delete_thread_meta,
    load_all_threads,
    load_thread_messages,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lumina Chat",
    page_icon="🌹",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Romantic CSS — rose-gold, candlelight, velvet dark
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Raleway:wght@300;400;500&display=swap');

    /* ── Root palette ── */
    :root {
        --rose:       #c9748a;
        --rose-light: #e8a5b5;
        --rose-dark:  #8b4a5a;
        --gold:       #c9a96e;
        --gold-light: #e8d5a3;
        --velvet:     #12090d;
        --velvet-mid: #1c1018;
        --velvet-card:#221520;
        --velvet-line:#3a2535;
        --cream:      #f5ede8;
        --muted:      #8a7080;
    }

    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: var(--velvet) !important;
        font-family: 'Raleway', sans-serif;
    }

    /* Floating rose petals background */
    [data-testid="stMain"]::before {
        content: '';
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse 60% 40% at 20% 20%, rgba(201,116,138,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 50% 60% at 80% 80%, rgba(201,169,110,0.06) 0%, transparent 60%),
            radial-gradient(ellipse 30% 30% at 50% 50%, rgba(139,74,90,0.04) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--velvet-mid) !important;
        border-right: 1px solid var(--velvet-line) !important;
        min-width: 270px !important;
        max-width: 270px !important;
    }
    [data-testid="stSidebar"] > div { padding: 0 !important; }

    /* Sidebar inner padding */
    [data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1rem !important;
    }

    /* App title in sidebar */
    .sidebar-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.55rem;
        font-weight: 600;
        font-style: italic;
        background: linear-gradient(135deg, var(--rose-light), var(--gold));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0 0.2rem;
        letter-spacing: 0.03em;
    }
    .sidebar-tagline {
        font-size: 0.68rem;
        color: var(--muted) !important;
        text-align: center;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }

    /* New Chat button */
    div.new-chat-wrap > button {
        background: linear-gradient(135deg, var(--rose-dark), #6b3045) !important;
        border: 1px solid var(--rose) !important;
        border-radius: 10px !important;
        color: var(--cream) !important;
        font-family: 'Raleway', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em !important;
        width: 100% !important;
        padding: 0.6rem !important;
        margin-bottom: 1.4rem !important;
        box-shadow: 0 2px 12px rgba(201,116,138,0.25) !important;
        transition: all 0.2s ease !important;
    }
    div.new-chat-wrap > button:hover {
        background: linear-gradient(135deg, var(--rose), var(--rose-dark)) !important;
        box-shadow: 0 4px 20px rgba(201,116,138,0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* Section label */
    .section-label {
        font-size: 0.65rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted) !important;
        padding: 0 0.5rem 0.5rem;
        border-bottom: 1px solid var(--velvet-line);
        margin-bottom: 0.6rem;
    }

    /* Thread buttons */
    div.thread-item > button {
        background: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        color: #c0a8b0 !important;
        text-align: left !important;
        width: 100% !important;
        padding: 0.5rem 0.75rem !important;
        font-family: 'Raleway', sans-serif !important;
        font-size: 0.82rem !important;
        transition: all 0.15s ease !important;
        line-height: 1.3 !important;
    }
    div.thread-item > button:hover {
        background: rgba(201,116,138,0.12) !important;
        color: var(--cream) !important;
    }
    div.thread-item-active > button {
        background: rgba(201,116,138,0.18) !important;
        color: var(--cream) !important;
        border-left: 2px solid var(--rose) !important;
        font-weight: 500 !important;
    }

    /* Delete button */
    div.del-btn > button {
        background: transparent !important;
        border: none !important;
        color: #5a3a45 !important;
        font-size: 0.75rem !important;
        padding: 0.4rem !important;
        border-radius: 4px !important;
        transition: color 0.15s !important;
    }
    div.del-btn > button:hover {
        color: var(--rose) !important;
        background: rgba(201,116,138,0.1) !important;
    }

    /* ── Chat area ── */
    .chat-header {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.6rem;
        font-style: italic;
        font-weight: 300;
        color: var(--cream);
        text-align: center;
        padding: 1rem 0 0.3rem;
        letter-spacing: 0.04em;
    }
    .chat-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, var(--rose), transparent);
        margin: 0 auto 1.5rem;
        width: 60%;
        opacity: 0.5;
    }

    /* Message bubbles */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.3rem 0 !important;
    }

    /* User message */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, rgba(139,74,90,0.35), rgba(201,116,138,0.2)) !important;
        border: 1px solid rgba(201,116,138,0.25) !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 0.75rem 1rem !important;
        color: var(--cream) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }

    /* Assistant message */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background: rgba(34,21,32,0.8) !important;
        border: 1px solid rgba(201,169,110,0.2) !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 0.75rem 1rem !important;
        color: #e8dde5 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }

    [data-testid="stChatMessageContent"] p {
        color: inherit !important;
        font-family: 'Raleway', sans-serif !important;
        font-size: 0.92rem !important;
        line-height: 1.65 !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        background: transparent !important;
    }
    [data-testid="stChatInput"] textarea {
        background: var(--velvet-card) !important;
        border: 1px solid var(--velvet-line) !important;
        border-radius: 14px !important;
        color: var(--cream) !important;
        font-family: 'Raleway', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 0.8rem 1rem !important;
        box-shadow: 0 0 0 0px var(--rose) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--rose-dark) !important;
        box-shadow: 0 0 0 2px rgba(201,116,138,0.15) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--muted) !important;
        font-style: italic !important;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        margin-top: 5rem;
        animation: fadeIn 1s ease;
    }
    .empty-rose { font-size: 3rem; margin-bottom: 1rem; }
    .empty-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.8rem;
        font-style: italic;
        font-weight: 300;
        color: var(--cream);
        margin-bottom: 0.5rem;
    }
    .empty-sub {
        color: var(--muted);
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] p { color: var(--rose-light) !important; font-style: italic; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: var(--velvet-mid); }
    ::-webkit-scrollbar-thumb { background: var(--velvet-line); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--rose-dark); }

    /* ── Animations ── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def initialize_session() -> None:
    if "active_thread" not in st.session_state:
        threads = load_all_threads()
        if threads:
            st.session_state.active_thread = threads[0]["thread_id"]
        else:
            _create_and_activate_thread()
    if "thread_messages" not in st.session_state:
        st.session_state.thread_messages = load_thread_messages(
            st.session_state.active_thread
        )


def _create_and_activate_thread() -> str:
    tid = str(uuid.uuid4())
    now = datetime.now()
    save_thread_meta(tid, "New conversation", now)
    st.session_state.active_thread = tid
    st.session_state.thread_messages = []
    return tid


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.active_thread}}


def auto_title(text: str) -> str:
    words = text.strip().split()
    title = " ".join(words[:6])
    return (title[:38] + "…") if len(title) > 38 else title


def switch_thread(tid: str) -> None:
    st.session_state.active_thread = tid
    st.session_state.thread_messages = load_thread_messages(tid)
    st.rerun()


def new_chat() -> None:
    _create_and_activate_thread()
    st.rerun()


def remove_thread(tid: str) -> None:
    delete_thread_meta(tid)
    threads = load_all_threads()
    if threads:
        switch_thread(threads[0]["thread_id"])
    else:
        _create_and_activate_thread()
        st.rerun()


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
        placeholder.markdown(streamed + "✦")
        time.sleep(0.025)
    placeholder.markdown(streamed.strip())

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🌹 Lumina</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-tagline">Your AI Companion</div>', unsafe_allow_html=True)

        # New Chat
        st.markdown('<div class="new-chat-wrap">', unsafe_allow_html=True)
        if st.button("✦  Begin New Conversation", key="new_chat", use_container_width=True):
            new_chat()
        st.markdown("</div>", unsafe_allow_html=True)

        # Thread list
        st.markdown('<div class="section-label">Past Conversations</div>', unsafe_allow_html=True)

        threads = load_all_threads()
        if not threads:
            st.markdown(
                "<p style='color:#5a3a45; font-size:0.78rem; padding:0.5rem; font-style:italic;'>"
                "No conversations yet…</p>",
                unsafe_allow_html=True,
            )
        else:
            for t in threads:
                is_active = t["thread_id"] == st.session_state.active_thread
                btn_cls = "thread-item-active" if is_active else "thread-item"
                label = ("🌸 " if is_active else "🌺 ") + t["title"]

                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f'<div class="{btn_cls}">', unsafe_allow_html=True)
                    if st.button(label, key=f"t_{t['thread_id']}", use_container_width=True):
                        if not is_active:
                            switch_thread(t["thread_id"])
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="del-btn">', unsafe_allow_html=True)
                    if st.button("✕", key=f"d_{t['thread_id']}", help="Delete"):
                        remove_thread(t["thread_id"])
                    st.markdown("</div>", unsafe_allow_html=True)

        # Footer
        st.markdown(
            "<div style='position:absolute; bottom:1.5rem; left:0; right:0; text-align:center; "
            "color:#3a2535; font-size:0.68rem; letter-spacing:0.08em;'>"
            "✦ &nbsp; LUMINA &nbsp; ✦<br>"
            "<span style='color:#2a1a25;'>Powered by Gemini + LangGraph</span>"
            "</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Chat area
# ─────────────────────────────────────────────────────────────────────────────

def display_chat_history() -> None:
    if not st.session_state.thread_messages:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-rose">🌹</div>
                <div class="empty-title">Begin your story…</div>
                <div class="empty-sub">Every great conversation starts with a single word.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for msg in st.session_state.thread_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def handle_user_input(user_input: str) -> None:
    tid = st.session_state.active_thread

    # Auto-title on first message
    if not st.session_state.thread_messages:
        title = auto_title(user_input)
        update_thread_title(tid, title)

    # User bubble
    st.session_state.thread_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant bubble
    with st.chat_message("assistant"):
        with st.spinner("Weaving a response…"):
            try:
                response = get_bot_response(user_input)
            except Exception as e:
                response = f"✦ Something went awry: {e}"
        stream_response(response)

    st.session_state.thread_messages.append({"role": "assistant", "content": response})

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    initialize_session()
    render_sidebar()

    # Current thread title as header
    threads = {t["thread_id"]: t for t in load_all_threads()}
    active = threads.get(st.session_state.active_thread, {})
    title = active.get("title", "New Conversation")

    st.markdown(f'<div class="chat-header">{title}</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-divider"></div>', unsafe_allow_html=True)

    display_chat_history()

    user_input = st.chat_input("Whisper your thoughts…")
    if user_input:
        handle_user_input(user_input)


if __name__ == "__main__":
    main()