"""
Simple Weather MCP Server for testing the generic MCP tool.

This server provides mock weather data for testing purposes.
Run with: uv run python server.py [port]
Default port is 8765, endpoint will be at http://localhost:8765/mcp
"""

import random
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

# Create the MCP server
mcp = FastMCP("Weather API Server")


# Mock weather data for various cities
CITY_WEATHER = {
    "new york": {"temp_f": 45, "temp_c": 7, "condition": "Cloudy", "humidity": 65, "wind_mph": 12},
    "los angeles": {"temp_f": 72, "temp_c": 22, "condition": "Sunny", "humidity": 40, "wind_mph": 8},
    "chicago": {"temp_f": 38, "temp_c": 3, "condition": "Windy", "humidity": 55, "wind_mph": 25},
    "seattle": {"temp_f": 52, "temp_c": 11, "condition": "Rainy", "humidity": 85, "wind_mph": 10},
    "miami": {"temp_f": 82, "temp_c": 28, "condition": "Partly Cloudy", "humidity": 75, "wind_mph": 15},
    "london": {"temp_f": 50, "temp_c": 10, "condition": "Foggy", "humidity": 80, "wind_mph": 5},
    "tokyo": {"temp_f": 55, "temp_c": 13, "condition": "Clear", "humidity": 50, "wind_mph": 7},
    "paris": {"temp_f": 48, "temp_c": 9, "condition": "Overcast", "humidity": 70, "wind_mph": 6},
}

CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Clear", "Windy", "Foggy", "Stormy"]


def _get_city_weather(city: str) -> dict:
    """Get weather data for a city, using mock data or generating random data."""
    city_lower = city.lower()
    if city_lower in CITY_WEATHER:
        return CITY_WEATHER[city_lower]
    # Generate random weather for unknown cities
    temp_c = random.randint(-10, 35)
    return {
        "temp_f": round(temp_c * 9 / 5 + 32),
        "temp_c": temp_c,
        "condition": random.choice(CONDITIONS),
        "humidity": random.randint(30, 95),
        "wind_mph": random.randint(0, 30),
    }


@mcp.tool
def get_current_weather(
    city: Annotated[str, Field(description="The name of the city to get weather for (e.g., 'New York', 'London')")],
) -> str:
    """Get the current weather for a city."""
    weather = _get_city_weather(city)
    return (
        f"Current weather in {city.title()}:\n"
        f"Temperature: {weather['temp_f']}°F ({weather['temp_c']}°C)\n"
        f"Condition: {weather['condition']}\n"
        f"Humidity: {weather['humidity']}%\n"
        f"Wind: {weather['wind_mph']} mph"
    )


@mcp.tool
def get_weather_forecast(
    city: Annotated[str, Field(description="The name of the city to get the forecast for")],
    days: Annotated[int, Field(description="Number of days for the forecast (1-7)")] = 3,
) -> str:
    """Get the weather forecast for a city."""
    days = max(1, min(7, days))  # Clamp to 1-7 days
    base_weather = _get_city_weather(city)

    forecast_lines = [f"Weather forecast for {city.title()} ({days} days):"]
    forecast_lines.append("-" * 40)

    for day in range(1, days + 1):
        # Add some variation for forecast days
        temp_variation = random.randint(-5, 5)
        temp_c = base_weather["temp_c"] + temp_variation
        temp_f = round(temp_c * 9 / 5 + 32)
        condition = random.choice(CONDITIONS)
        humidity = max(0, min(100, base_weather["humidity"] + random.randint(-15, 15)))

        forecast_lines.append(f"Day {day}:")
        forecast_lines.append(f"  Temperature: {temp_f}°F ({temp_c}°C)")
        forecast_lines.append(f"  Condition: {condition}")
        forecast_lines.append(f"  Humidity: {humidity}%")
        forecast_lines.append("")

    return "\n".join(forecast_lines)


@mcp.tool
def compare_weather(
    city1: Annotated[str, Field(description="The first city to compare")],
    city2: Annotated[str, Field(description="The second city to compare")],
) -> str:
    """Compare the current weather between two cities."""
    weather1 = _get_city_weather(city1)
    weather2 = _get_city_weather(city2)

    temp_diff = abs(weather1["temp_f"] - weather2["temp_f"])
    warmer = city1.title() if weather1["temp_f"] > weather2["temp_f"] else city2.title()

    return (
        f"Weather comparison: {city1.title()} vs {city2.title()}\n"
        f"{'=' * 50}\n\n"
        f"{city1.title()}:\n"
        f"  Temperature: {weather1['temp_f']}°F ({weather1['temp_c']}°C)\n"
        f"  Condition: {weather1['condition']}\n"
        f"  Humidity: {weather1['humidity']}%\n"
        f"  Wind: {weather1['wind_mph']} mph\n\n"
        f"{city2.title()}:\n"
        f"  Temperature: {weather2['temp_f']}°F ({weather2['temp_c']}°C)\n"
        f"  Condition: {weather2['condition']}\n"
        f"  Humidity: {weather2['humidity']}%\n"
        f"  Wind: {weather2['wind_mph']} mph\n\n"
        f"Summary: {'Both cities have the same temperature.' if temp_diff == 0 else f'{warmer} is {temp_diff}°F warmer.'}"
    )


if __name__ == "__main__":
    import sys

    port = 8765
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid port '{sys.argv[1]}'. Using default port {port}.")
    mcp.run(transport="http", host="0.0.0.0", port=port)
