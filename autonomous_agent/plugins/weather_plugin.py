"""
Example plugin: Weather Tool
Place any file in the plugins/ directory and it will be auto-loaded.
Each plugin must expose a register(tool_manager) function.
"""

import json
import urllib.request
import urllib.parse

from tools.tool_manager import BaseTool


class WeatherTool(BaseTool):
    name = "weather"
    description = (
        "Get current weather for a city using the Open-Meteo API (no key required). "
        "Returns temperature, conditions, and wind speed."
    )
    parameters = {
        "city": {
            "type": "string",
            "description": "City name, e.g. 'London' or 'Cairo'"
        }
    }

    def execute(self, city: str) -> str:
        # Step 1: geocode city → lat/lon via nominatim
        geo_url = (
            "https://nominatim.openstreetmap.org/search?"
            + urllib.parse.urlencode({"q": city, "format": "json", "limit": 1})
        )
        headers = {"User-Agent": "AutonomousAgent/1.0"}
        req = urllib.request.Request(geo_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                geo_data = json.loads(resp.read())
        except Exception as e:
            return f"Geocoding failed for '{city}': {e}"

        if not geo_data:
            return f"Could not find location: {city}"

        lat = float(geo_data[0]["lat"])
        lon = float(geo_data[0]["lon"])

        # Step 2: fetch weather from open-meteo
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current_weather=true"
            f"&temperature_unit=celsius"
        )
        try:
            with urllib.request.urlopen(weather_url, timeout=10) as resp:
                weather_data = json.loads(resp.read())
        except Exception as e:
            return f"Weather fetch failed: {e}"

        cw = weather_data.get("current_weather", {})
        temp = cw.get("temperature", "N/A")
        wind = cw.get("windspeed", "N/A")
        code = cw.get("weathercode", 0)

        # Simple WMO weather code mapping
        conditions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 51: "Light drizzle", 61: "Light rain", 63: "Moderate rain",
            71: "Light snow", 80: "Rain showers", 95: "Thunderstorm",
        }
        condition = conditions.get(code, f"Code {code}")

        return (
            f"Weather in {city} (lat={lat:.2f}, lon={lon:.2f}):\n"
            f"  Temperature : {temp}°C\n"
            f"  Condition   : {condition}\n"
            f"  Wind speed  : {wind} km/h"
        )


def register(tool_manager):
    """Called automatically by ToolManager.load_plugins()"""
    tool_manager.register(WeatherTool())
