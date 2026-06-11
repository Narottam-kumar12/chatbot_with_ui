# 🌹 Lumina — AI Chatbot with Tools & Persistent Memory

> A production-grade conversational AI built with **LangGraph**, **Gemini 2.5 Flash**, and **Streamlit** — featuring real-time tool use, SQLite-backed persistent memory, per-thread LangSmith tracing, and a romantic dark UI.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tools](#tools)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [LangSmith Tracing](#langsmith-tracing)
- [Graph Flow](#graph-flow)
- [Tech Stack](#tech-stack)
- [Known Issues & Fixes](#known-issues--fixes)

---

## Overview

Lumina is a multi-turn AI chatbot that goes beyond simple Q&A. It uses a **ReAct (Reason + Act)** agent pattern via LangGraph — the LLM decides when to call tools, executes them, then synthesizes a final response. All conversations are persisted to **SQLite** so they survive server restarts. Each thread is independently traced on **LangSmith** for full observability.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                         │
│         (app.py — romantic dark theme, sidebar threads)     │
└────────────────────────┬────────────────────────────────────┘
                         │ HumanMessage
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   LangGraph Agent Graph                     │
│                                                             │
│   START → [ model node ] ──── has tool_calls? ──► [ tools ]│
│                │                                      │     │
│                │ No                                   │     │
│                ▼                                      ▼     │
│               END  ◄──────────────────────── back to model │
└─────────────────────────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   SqliteSaver     LangSmith       Gemini 2.0
  (persistence)    (tracing)        (LLM)
```

---

## Features

- **Multi-turn memory** — Full conversation history via LangGraph checkpoints stored in SQLite
- **Thread management** — Create, switch, resume, and delete conversation threads from the sidebar
- **Auto-titling** — Each thread is automatically titled from the first user message
- **ReAct agent** — LLM decides when and which tool to call; supports multi-step tool chaining
- **Real-time tools** — Calculator, web search, news, stock prices, weather (all free, no extra API keys)
- **LangSmith tracing** — Every thread traced with `run_name`, `tags`, and `metadata` for isolation
- **Streaming responses** — Word-by-word animated output with typing cursor
- **Robust content handling** — Handles both `str` and `list` content from Gemini after tool calls

---

## Project Structure

```
lumina-chatbot/
│
├── app.py                  # Streamlit frontend — UI, sidebar, chat flow
├── langgraph_backend.py    # LangGraph agent graph, SQLite memory, thread helpers
│
├── tool_calculator.py      # 🧮 Safe math expression evaluator
├── tool_search.py          # 🔍 DuckDuckGo web & news search
├── tool_stock.py           # 📈 Yahoo Finance stock prices & history
├── tool_weather.py         # 🌤️ Open-Meteo weather & forecasts
│
├── requirements.txt        # All dependencies
├── .env.example            # Environment variable template
├── chat_history.db         # SQLite DB (auto-created on first run)
└── README.md
```

---

## Tools

| Tool | Function | Data Source | API Key |
|------|----------|-------------|---------|
| 🧮 Calculator | Arithmetic, trig, sqrt, log, factorial | Python `math` module | ❌ None |
| 🔍 Web Search | General knowledge, real-time web results | DuckDuckGo | ❌ None |
| 📰 News Search | Latest news on any topic | DuckDuckGo News | ❌ None |
| 📈 Stock Price | Current price, P/E, market cap, 52-week range | Yahoo Finance (`yfinance`) | ❌ None |
| 📊 Stock History | Historical OHLCV data, % performance | Yahoo Finance (`yfinance`) | ❌ None |
| 📉 Compare Stocks | Side-by-side multi-stock comparison | Yahoo Finance (`yfinance`) | ❌ None |
| 🌤️ Current Weather | Temperature, humidity, wind, UV index | Open-Meteo + Nominatim | ❌ None |
| 📅 Weather Forecast | 1–7 day daily forecast | Open-Meteo | ❌ None |
| 🌍 Compare Weather | Multi-city weather comparison table | Open-Meteo | ❌ None |

> **Only 2 API keys needed** for the entire project: Google Gemini and LangSmith (optional).

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/lumina-chatbot.git
cd lumina-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv myenv
source myenv/bin/activate        # macOS/Linux
myenv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys (see [Configuration](#configuration)).

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Configuration

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=lumina-chatbot
```

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | Gemini API key from [Google AI Studio](https://aistudio.google.com) |
| `LANGCHAIN_API_KEY` | ⚠️ Optional | LangSmith key from [smith.langchain.com](https://smith.langchain.com) |
| `LANGCHAIN_TRACING_V2` | ⚠️ Optional | Set `false` to disable tracing |
| `LANGCHAIN_PROJECT` | ⚠️ Optional | LangSmith project name (default: `lumina-chatbot`) |

---

## LangSmith Tracing

Every chatbot invocation is traced with thread-level metadata so you can isolate and debug individual conversations:

```python
{
    "run_name": "[Thread Title]",
    "tags": [
        "thread:uuid-xxxx",        # filter by exact thread
        "title:Python questions",  # filter by thread name
        "lumina-chatbot",          # filter all project runs
    ],
    "metadata": {
        "thread_id":    "uuid-xxxx",
        "thread_title": "Python questions"
    }
}
```

**On the LangSmith dashboard you can:**
- Filter runs by thread ID or title using tags
- See full tool call chains (model → tool → model)
- Inspect token usage and latency per step
- Debug errors with full stack traces

---

## Graph Flow

```
User Message
     │
     ▼
┌─────────────┐     tool_calls present?     ┌──────────────┐
│  model node │ ──────────── YES ─────────► │  tools node  │
│  (Gemini)   │                             │  (ToolNode)  │
└─────────────┘                             └──────┬───────┘
     │                                             │
     │ NO (plain text response)                    │ ToolMessage
     ▼                                             ▼
    END                                      back to model
                                          (synthesize answer)
```

The agent supports **multi-step tool chaining** — for example:

```
User: "Search for AAPL news and also tell me current price"
  → news_search("AAPL")     [tool call 1]
  → get_stock_price("AAPL") [tool call 2]
  → Final answer combining both results
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini 2.0 Flash |
| Agent Framework | LangGraph |
| Memory / Persistence | LangGraph `SqliteSaver` |
| Observability | LangSmith |
| Frontend | Streamlit |
| Web Search | DuckDuckGo (`duckduckgo-search`) |
| Stock Data | Yahoo Finance (`yfinance`) |
| Weather Data | Open-Meteo API + Nominatim Geocoding |
| Math Engine | Python `math` module (sandboxed `eval`) |
| Environment | `python-dotenv` |

---

## Known Issues & Fixes

### `AttributeError: 'list' object has no attribute 'split'`

**Cause:** Gemini returns `AIMessage.content` as a `list` of content blocks (instead of a plain string) after tool calls.

**Fix applied in `app.py`:**
```python
content = response["messages"][-1].content
if isinstance(content, list):
    text = " ".join(
        part["text"] for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ).strip()
    return text if text else "I couldn't generate a response."
return str(content) if content else "I couldn't generate a response."
```

---

## Example Queries

```
🧮  "2^32 + sqrt(144) kya hoga?"
📈  "Tesla stock price kya hai aaj?"
📊  "AAPL aur MSFT compare karo"
🌤️  "Mumbai mein aaj mausam kaisa hai?"
📅  "Delhi ka 7 din ka forecast batao"
🔍  "Latest developments in AI agents"
📰  "Today's top tech news"
🌍  "Compare weather in Dubai, London, and Tokyo"
```
