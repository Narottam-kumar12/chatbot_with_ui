"""
tool_stock.py
─────────────
Stock price & financial data tool for the Lumina chatbot.
Uses yfinance (Yahoo Finance) — free, no API key required.
"""

from datetime import datetime, timedelta

import yfinance as yf
from langchain_core.tools import tool


def _fmt(value, prefix="", suffix="", decimals=2) -> str:
    """Format a number nicely, return 'N/A' if None."""
    if value is None:
        return "N/A"
    try:
        return f"{prefix}{float(value):,.{decimals}f}{suffix}"
    except Exception:
        return str(value)


@tool
def get_stock_price(ticker: str) -> str:
    """
    Get the current (latest) stock price and key stats for a given ticker symbol.

    Use this when the user asks:
    - "What is the price of Apple stock?"
    - "How is Tesla doing today?"
    - "Show me MSFT stock info"
    - "Current price of RELIANCE.NS" (Indian stocks use .NS suffix)

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, TSLA, MSFT, GOOGL, RELIANCE.NS, TCS.NS).

    Returns:
        Current price, daily change, volume, market cap, PE ratio, 52-week range.

    Examples:
        "AAPL"       → Apple Inc.
        "TSLA"       → Tesla Inc.
        "RELIANCE.NS"→ Reliance Industries (NSE India)
        "TCS.NS"     → Tata Consultancy Services (NSE India)
    """
    ticker = ticker.upper().strip()
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        name         = info.get("longName") or info.get("shortName") or ticker
        currency     = info.get("currency", "USD")
        price        = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close   = info.get("previousClose") or info.get("regularMarketPreviousClose")
        open_price   = info.get("open") or info.get("regularMarketOpen")
        day_high     = info.get("dayHigh") or info.get("regularMarketDayHigh")
        day_low      = info.get("dayLow")  or info.get("regularMarketDayLow")
        volume       = info.get("volume")  or info.get("regularMarketVolume")
        market_cap   = info.get("marketCap")
        pe_ratio     = info.get("trailingPE")
        week52_high  = info.get("fiftyTwoWeekHigh")
        week52_low   = info.get("fiftyTwoWeekLow")
        exchange     = info.get("exchange", "")

        # Daily change
        if price and prev_close:
            change     = price - prev_close
            change_pct = (change / prev_close) * 100
            arrow      = "🟢 ▲" if change >= 0 else "🔴 ▼"
            change_str = f"{arrow} {_fmt(abs(change))} ({abs(change_pct):.2f}%)"
        else:
            change_str = "N/A"

        # Market cap formatting
        if market_cap:
            if market_cap >= 1e12:
                mc_str = f"{market_cap/1e12:.2f}T {currency}"
            elif market_cap >= 1e9:
                mc_str = f"{market_cap/1e9:.2f}B {currency}"
            elif market_cap >= 1e6:
                mc_str = f"{market_cap/1e6:.2f}M {currency}"
            else:
                mc_str = _fmt(market_cap, suffix=f" {currency}", decimals=0)
        else:
            mc_str = "N/A"

        return (
            f"📈 **{name}** (`{ticker}` · {exchange})\n\n"
            f"💰 **Current Price:** {_fmt(price, suffix=f' {currency}')}\n"
            f"📊 **Change (Day):** {change_str}\n"
            f"🔓 **Open:** {_fmt(open_price, suffix=f' {currency}')}\n"
            f"📉 **Day Low:** {_fmt(day_low, suffix=f' {currency}')}\n"
            f"📈 **Day High:** {_fmt(day_high, suffix=f' {currency}')}\n"
            f"📦 **Volume:** {int(volume):,}\n"
            f"🏦 **Market Cap:** {mc_str}\n"
            f"📐 **P/E Ratio:** {_fmt(pe_ratio)}\n"
            f"📅 **52-Week Range:** {_fmt(week52_low)} – {_fmt(week52_high)} {currency}\n"
            f"⏰ **Prev Close:** {_fmt(prev_close, suffix=f' {currency}')}"
        )

    except Exception as e:
        return (
            f"❌ **Stock Error for `{ticker}`:** {e}\n\n"
            f"💡 Make sure the ticker symbol is correct. "
            f"For Indian stocks use `.NS` (NSE) or `.BO` (BSE) suffix, e.g. `RELIANCE.NS`"
        )


@tool
def get_stock_history(ticker: str, period: str = "1mo") -> str:
    """
    Get historical stock price data and performance summary for a ticker.

    Use this when the user asks:
    - "How has Apple performed over the last month?"
    - "Show me Tesla stock history for 3 months"
    - "NIFTY performance this year"

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, TSLA, ^NSEI for NIFTY).
        period: Time period — one of:
                "1d"  = 1 day
                "5d"  = 5 days
                "1mo" = 1 month  (default)
                "3mo" = 3 months
                "6mo" = 6 months
                "1y"  = 1 year
                "2y"  = 2 years
                "5y"  = 5 years

    Returns:
        Performance summary: start/end price, % change, high/low, avg volume.
    """
    ticker = ticker.upper().strip()
    valid_periods = ["1d","5d","1mo","3mo","6mo","1y","2y","5y","ytd","max"]
    if period not in valid_periods:
        period = "1mo"

    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period=period)

        if hist.empty:
            return f"❌ No historical data found for `{ticker}` with period `{period}`."

        info     = stock.info
        name     = info.get("longName") or info.get("shortName") or ticker
        currency = info.get("currency", "USD")

        start_price = hist["Close"].iloc[0]
        end_price   = hist["Close"].iloc[-1]
        high_price  = hist["High"].max()
        low_price   = hist["Low"].min()
        avg_volume  = hist["Volume"].mean()
        change_pct  = ((end_price - start_price) / start_price) * 100
        arrow       = "🟢 ▲" if change_pct >= 0 else "🔴 ▼"

        start_date = hist.index[0].strftime("%d %b %Y")
        end_date   = hist.index[-1].strftime("%d %b %Y")

        # Last 5 closing prices
        last5 = hist["Close"].tail(5)
        last5_str = "\n".join(
            f"  • {d.strftime('%d %b')}: {p:.2f} {currency}"
            for d, p in last5.items()
        )

        return (
            f"📊 **{name}** (`{ticker}`) — Period: `{period}`\n\n"
            f"📅 **From:** {start_date}  →  **To:** {end_date}\n\n"
            f"🚀 **Start Price:** {start_price:.2f} {currency}\n"
            f"🏁 **End Price:** {end_price:.2f} {currency}\n"
            f"📈 **Performance:** {arrow} {abs(change_pct):.2f}%\n"
            f"🔝 **Period High:** {high_price:.2f} {currency}\n"
            f"🔻 **Period Low:** {low_price:.2f} {currency}\n"
            f"📦 **Avg Daily Volume:** {int(avg_volume):,}\n\n"
            f"**Last 5 Closes:**\n{last5_str}"
        )

    except Exception as e:
        return f"❌ **History Error for `{ticker}`:** {e}"


@tool
def compare_stocks(tickers: str) -> str:
    """
    Compare current prices and performance of multiple stocks side-by-side.

    Use this when the user asks:
    - "Compare Apple and Microsoft"
    - "AAPL vs GOOGL vs MSFT"
    - "Which is better: TCS or Infosys?"

    Args:
        tickers: Comma-separated ticker symbols (e.g. "AAPL,MSFT,GOOGL").

    Returns:
        Side-by-side comparison table of key metrics.
    """
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(symbols) < 2:
        return "❌ Please provide at least 2 ticker symbols separated by commas. E.g. `AAPL,MSFT`"
    if len(symbols) > 5:
        symbols = symbols[:5]

    rows = []
    for sym in symbols:
        try:
            info     = yf.Ticker(sym).info
            name     = (info.get("longName") or info.get("shortName") or sym)[:25]
            price    = info.get("currentPrice") or info.get("regularMarketPrice")
            prev     = info.get("previousClose")
            currency = info.get("currency", "USD")
            chg_pct  = ((price - prev) / prev * 100) if price and prev else None
            pe       = info.get("trailingPE")
            mc       = info.get("marketCap")
            mc_b     = f"{mc/1e9:.1f}B" if mc else "N/A"
            chg_str  = (f"{'▲' if chg_pct>=0 else '▼'} {abs(chg_pct):.2f}%") if chg_pct else "N/A"

            rows.append({
                "sym": sym, "name": name,
                "price": f"{price:.2f} {currency}" if price else "N/A",
                "change": chg_str,
                "pe": f"{pe:.1f}" if pe else "N/A",
                "mc": mc_b,
            })
        except Exception:
            rows.append({"sym": sym, "name": sym, "price": "Error", "change": "-", "pe": "-", "mc": "-"})

    header = "📊 **Stock Comparison**\n\n"
    table  = f"{'Ticker':<12} {'Price':<18} {'Day Change':<14} {'P/E':<8} {'Mkt Cap'}\n"
    table += "─" * 65 + "\n"
    for r in rows:
        table += f"{r['sym']:<12} {r['price']:<18} {r['change']:<14} {r['pe']:<8} {r['mc']}\n"

    names = "\n".join(f"• `{r['sym']}` → {r['name']}" for r in rows)
    return header + f"```\n{table}```\n\n**Companies:**\n{names}"