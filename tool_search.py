"""
tool_search.py
──────────────
Web search tool for the Lumina chatbot.
Uses DuckDuckGo (free, no API key needed) via the duckduckgo-search library.
"""

from langchain_core.tools import tool
from duckduckgo_search import DDGS


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return the top results.

    Use this tool when the user asks about:
    - Current events or recent news
    - General knowledge questions
    - People, places, companies, products
    - How-to guides or tutorials
    - Anything that needs up-to-date information

    Args:
        query: The search query string.
        max_results: Number of results to return (default 5, max 10).

    Returns:
        Formatted list of search results with title, URL, and snippet.

    Examples:
        "latest AI news 2025"
        "how to make pasta carbonara"
        "who is Elon Musk"
        "Python tutorial for beginners"
    """
    max_results = min(max(1, max_results), 10)  # clamp between 1–10

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"🔍 No results found for: **{query}**"

        lines = [f"🔍 **Search Results for:** `{query}`\n"]
        for i, r in enumerate(results, 1):
            title   = r.get("title", "No title")
            href    = r.get("href", "")
            snippet = r.get("body", "No description available.")
            lines.append(
                f"**{i}. {title}**\n"
                f"🔗 {href}\n"
                f"{snippet}\n"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"❌ **Search Error:** {e}"


@tool
def news_search(query: str, max_results: int = 5) -> str:
    """
    Search for the latest news articles on a topic using DuckDuckGo News.

    Use this when the user asks about:
    - Breaking news or recent events
    - News about a specific topic, company, or person
    - Latest updates on an ongoing situation

    Args:
        query: News topic to search for.
        max_results: Number of news articles to return (default 5).

    Returns:
        Formatted list of recent news articles with title, source, date, and URL.

    Examples:
        "AI regulation news"
        "stock market today"
        "climate change 2025"
    """
    max_results = min(max(1, max_results), 10)

    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))

        if not results:
            return f"📰 No news found for: **{query}**"

        lines = [f"📰 **Latest News for:** `{query}`\n"]
        for i, r in enumerate(results, 1):
            title  = r.get("title", "No title")
            source = r.get("source", "Unknown source")
            date   = r.get("date", "")
            url    = r.get("url", "")
            body   = r.get("body", "")

            lines.append(
                f"**{i}. {title}**\n"
                f"📌 {source}  |  🕐 {date}\n"
                f"🔗 {url}\n"
                f"{body}\n"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"❌ **News Search Error:** {e}"