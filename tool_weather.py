"""
tool_weather.py
───────────────
Weather tool for the Lumina chatbot.
Uses Open-Meteo API (free, no API key) + Nominatim geocoding (OpenStreetMap).
"""

import requests
from langchain_core.tools import tool


# ── Geocoding ─────────────────────────────────────────────────────────────────

def _geocode(city: str) -> tuple[float, float, str]:
    """Convert city name → (latitude, longitude, display_name)."""
    url    = "https://nominatim.openstreetmap.org/search"
    params = {"q": city, "format": "json", "limit": 1}
    headers = {"User-Agent": "LuminaChatbot/1.0"}
    resp   = requests.get(url, params=params, headers=headers, timeout=8)
    resp.raise_for_status()
    data   = resp.json()
    if not data:
        raise ValueError(f"City not found: '{city}'. Try a more specific name.")
    return float(data[0]["lat"]), float(data[0]["lon"]), data[0]["display_name"]


# ── WMO Weather code descriptions ─────────────────────────────────────────────

WMO_CODES = {
    0: ("☀️", "Clear sky"),
    1: ("🌤️", "Mainly clear"), 2: ("⛅", "Partly cloudy"), 3: ("☁️", "Overcast"),
    45: ("🌫️", "Foggy"), 48: ("🌫️", "Icy fog"),
    51: ("🌦️", "Light drizzle"), 53: ("🌦️", "Moderate drizzle"), 55: ("🌧️", "Dense drizzle"),
    61: ("🌧️", "Slight rain"), 63: ("🌧️", "Moderate rain"), 65: ("🌧️", "Heavy rain"),
    71: ("🌨️", "Slight snow"), 73: ("🌨️", "Moderate snow"), 75: ("❄️", "Heavy snow"),
    77: ("🌨️", "Snow grains"),
    80: ("🌦️", "Slight showers"), 81: ("🌧️", "Moderate showers"), 82: ("⛈️", "Violent showers"),
    85: ("🌨️", "Snow showers"), 86: ("❄️", "Heavy snow showers"),
    95: ("⛈️", "Thunderstorm"), 96: ("⛈️", "Thunderstorm w/ hail"), 99: ("⛈️", "Severe thunderstorm"),
}

def _wmo(code: int) -> tuple[str, str]:
    return WMO_CODES.get(code, ("🌡️", f"Code {code}"))


def _wind_dir(degrees: float) -> str:
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    return dirs[round(degrees / 45) % 8]


def _uv_label(uv: float) -> str:
    if uv <= 2:   return "Low 🟢"
    if uv <= 5:   return "Moderate 🟡"
    if uv <= 7:   return "High 🟠"
    if uv <= 10:  return "Very High 🔴"
    return "Extreme 🟣"


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def get_current_weather(city: str) -> str:
    """
    Get the current weather conditions for any city in the world.

    Use this when the user asks:
    - "What's the weather in Mumbai?"
    - "Is it raining in London?"
    - "Temperature in New York right now"
    - "Weather in Delhi today"
    - "Kaisa mausam hai Dubai mein?"

    Args:
        city: Name of the city (e.g. "Mumbai", "London", "New York", "Paris").

    Returns:
        Current temperature, feels-like, humidity, wind, UV index, conditions.
    """
    try:
        lat, lon, display = _geocode(city)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "current": [
                "temperature_2m", "relative_humidity_2m", "apparent_temperature",
                "precipitation", "weather_code", "wind_speed_10m",
                "wind_direction_10m", "surface_pressure", "uv_index",
                "visibility", "is_day",
            ],
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        c = resp.json()["current"]

        icon, desc   = _wmo(c["weather_code"])
        temp         = c["temperature_2m"]
        feels        = c["apparent_temperature"]
        humidity     = c["relative_humidity_2m"]
        precip       = c["precipitation"]
        wind_spd     = c["wind_speed_10m"]
        wind_dir     = _wind_dir(c["wind_direction_10m"])
        pressure     = c["surface_pressure"]
        uv           = c["uv_index"]
        visibility   = c.get("visibility", 0) / 1000  # m → km
        time_str     = c["time"].replace("T", " ")

        return (
            f"{icon} **Current Weather — {display.split(',')[0]}**\n\n"
            f"🌡️ **Temperature:** {temp}°C  (feels like {feels}°C)\n"
            f"🌥️ **Condition:** {desc}\n"
            f"💧 **Humidity:** {humidity}%\n"
            f"🌧️ **Precipitation:** {precip} mm\n"
            f"💨 **Wind:** {wind_spd} km/h {wind_dir}\n"
            f"🔵 **Pressure:** {pressure} hPa\n"
            f"🌞 **UV Index:** {uv} — {_uv_label(uv)}\n"
            f"👁️ **Visibility:** {visibility:.1f} km\n"
            f"⏰ **Updated:** {time_str}"
        )

    except ValueError as e:
        return f"❌ {e}"
    except Exception as e:
        return f"❌ **Weather Error:** {e}"


@tool
def get_weather_forecast(city: str, days: int = 5) -> str:
    """
    Get a multi-day weather forecast for any city.

    Use this when the user asks:
    - "Weather forecast for Mumbai this week"
    - "Will it rain in Delhi tomorrow?"
    - "Next 7 days weather in London"
    - "Weekend weather in Paris"

    Args:
        city: Name of the city.
        days: Number of forecast days (1–7, default 5).

    Returns:
        Day-by-day forecast with temperature range, conditions, rain chance, wind.
    """
    days = min(max(1, days), 7)

    try:
        lat, lon, display = _geocode(city)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "daily": [
                "weather_code", "temperature_2m_max", "temperature_2m_min",
                "precipitation_sum", "precipitation_probability_max",
                "wind_speed_10m_max", "uv_index_max", "sunrise", "sunset",
            ],
            "forecast_days": days,
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        d = resp.json()["daily"]

        city_name = display.split(",")[0]
        lines = [f"📅 **{days}-Day Forecast — {city_name}**\n"]

        for i in range(days):
            date      = d["time"][i]
            icon, desc = _wmo(d["weather_code"][i])
            t_max     = d["temperature_2m_max"][i]
            t_min     = d["temperature_2m_min"][i]
            rain_sum  = d["precipitation_sum"][i]
            rain_prob = d["precipitation_probability_max"][i]
            wind      = d["wind_speed_10m_max"][i]
            uv        = d["uv_index_max"][i]
            sunrise   = d["sunrise"][i].split("T")[1] if "T" in d["sunrise"][i] else d["sunrise"][i]
            sunset    = d["sunset"][i].split("T")[1]  if "T" in d["sunset"][i]  else d["sunset"][i]

            lines.append(
                f"**📆 {date}** {icon} {desc}\n"
                f"  🌡️ {t_min}°C – {t_max}°C  |  "
                f"🌧️ {rain_sum}mm ({rain_prob}% chance)  |  "
                f"💨 {wind} km/h  |  ☀️ UV {uv}\n"
                f"  🌅 Sunrise {sunrise}  🌇 Sunset {sunset}\n"
            )

        return "\n".join(lines)

    except ValueError as e:
        return f"❌ {e}"
    except Exception as e:
        return f"❌ **Forecast Error:** {e}"


@tool
def compare_weather(cities: str) -> str:
    """
    Compare current weather between multiple cities.

    Use this when the user asks:
    - "Compare weather in Mumbai and Delhi"
    - "Is it hotter in Dubai or London?"
    - "Weather comparison: Paris, Tokyo, New York"

    Args:
        cities: Comma-separated city names (e.g. "Mumbai,Delhi,Bangalore").

    Returns:
        Side-by-side weather comparison for all cities.
    """
    city_list = [c.strip() for c in cities.split(",") if c.strip()]
    if len(city_list) < 2:
        return "❌ Please provide at least 2 cities separated by commas."
    if len(city_list) > 5:
        city_list = city_list[:5]

    results = []
    for city in city_list:
        try:
            lat, lon, display = _geocode(city)
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat, "longitude": lon,
                "current": [
                    "temperature_2m", "relative_humidity_2m",
                    "apparent_temperature", "weather_code", "wind_speed_10m",
                ],
                "wind_speed_unit": "kmh",
                "timezone": "auto",
            }
            resp = requests.get(url, params=params, timeout=8)
            resp.raise_for_status()
            c    = resp.json()["current"]
            icon, desc = _wmo(c["weather_code"])
            results.append({
                "city":     display.split(",")[0][:18],
                "temp":     c["temperature_2m"],
                "feels":    c["apparent_temperature"],
                "humidity": c["relative_humidity_2m"],
                "wind":     c["wind_speed_10m"],
                "desc":     f"{icon} {desc}",
            })
        except Exception as e:
            results.append({
                "city": city, "temp": "–", "feels": "–",
                "humidity": "–", "wind": "–", "desc": f"❌ {e}",
            })

    lines = ["🌍 **Weather Comparison**\n"]
    lines.append(f"{'City':<20} {'Temp':>6} {'Feels':>7} {'Humidity':>10} {'Wind':>10}  Condition")
    lines.append("─" * 80)
    for r in results:
        temp  = f"{r['temp']}°C"  if r['temp'] != "–" else "–"
        feels = f"{r['feels']}°C" if r['feels'] != "–" else "–"
        hum   = f"{r['humidity']}%"    if r['humidity'] != "–" else "–"
        wind  = f"{r['wind']} km/h"    if r['wind'] != "–" else "–"
        lines.append(f"{r['city']:<20} {temp:>6} {feels:>7} {hum:>10} {wind:>10}  {r['desc']}")

    return "```\n" + "\n".join(lines) + "\n```"