"""
tool_calculator.py
──────────────────
Mathematical calculator tool for the Lumina chatbot.
Supports: basic arithmetic, power, sqrt, log, trig functions, factorial, etc.
"""

import math
import re
from langchain_core.tools import tool


def _safe_eval(expression: str) -> float:
    """
    Safely evaluate a mathematical expression.
    Only allows math functions and operators — no arbitrary code execution.
    """
    # Allowed names: math module functions + constants
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed.update({
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
    })

    # Clean input: remove spaces, replace ^ with **
    expr = expression.strip()
    expr = expr.replace("^", "**")
    expr = expr.replace("×", "*")
    expr = expr.replace("÷", "/")
    expr = expr.replace("√", "sqrt")
    expr = re.sub(r'[^\d\s\+\-\*\/\.\(\)\%\,\_a-zA-Z]', '', expr)

    try:
        result = eval(expr, {"__builtins__": {}}, allowed)  # noqa: S307
        return result
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed.")
    except Exception as e:
        raise ValueError(f"Could not evaluate expression: {e}")


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.

    Supports:
    - Basic arithmetic: +, -, *, /, % (modulo)
    - Power: ** or ^  (e.g. 2**10 or 2^10)
    - Square root: sqrt(x)
    - Logarithm: log(x), log10(x), log2(x)
    - Trigonometry: sin(x), cos(x), tan(x)  [x in radians]
    - Constants: pi, e, tau, inf
    - Factorial: factorial(n)
    - Floor/ceil: floor(x), ceil(x)
    - Absolute value: abs(x)
    - Rounding: round(x, n)

    Examples:
        "2 + 3 * 4"           → 14
        "sqrt(144)"           → 12.0
        "2^10"                → 1024
        "sin(pi/2)"           → 1.0
        "log(e)"              → 1.0
        "factorial(6)"        → 720
        "(3.5 + 1.5) * 2"     → 10.0

    Args:
        expression: Mathematical expression as a string.

    Returns:
        The computed result as a string.
    """
    try:
        result = _safe_eval(expression)

        # Format nicely: int if whole number, else float
        if isinstance(result, float) and result.is_integer():
            formatted = str(int(result))
        else:
            formatted = f"{result:.10g}"  # up to 10 significant digits

        return (
            f"🧮 **Calculation Result**\n\n"
            f"**Expression:** `{expression}`\n"
            f"**Result:** `{formatted}`"
        )
    except ValueError as e:
        return f"❌ **Calculator Error:** {e}"
    except Exception as e:
        return f"❌ **Unexpected Error:** {e}"