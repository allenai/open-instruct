# Weather MCP Server

A simple MCP (Model Context Protocol) server that provides mock weather data for testing the `generic_mcp` tool integration.

## Tools Provided

The server exposes three weather tools:

1. **get_current_weather(city)**: Get current weather for a city
2. **get_weather_forecast(city, days)**: Get a multi-day weather forecast
3. **compare_weather(city1, city2)**: Compare weather between two cities

## Usage

### Running the Server

```bash
# Using uvicorn directly
cd open_instruct/tools/servers/weather_mcp_server
uv run uvicorn server:app --host 0.0.0.0 --port 8765

# Or run the script directly
cd open_instruct/tools/servers/weather_mcp_server
uv run python server.py
```

The server will start at `http://localhost:8765` with the MCP endpoint at `http://localhost:8765/mcp`.

### Using with generic_mcp Tool

```bash
# In your training command
--tools generic_mcp \
--tool_configs '{"server_url": "http://localhost:8765/mcp"}'
```

### Testing the Server

You can test the server using curl:

```bash
# List available tools
curl -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/list", "params": {}}'

# Call a tool
curl -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/call", "params": {"name": "get_current_weather", "arguments": {"city": "New York"}}}'
```

## Mock Data

The server includes predefined weather data for popular cities:
- New York, Los Angeles, Chicago, Seattle, Miami
- London, Tokyo, Paris

For unknown cities, random weather data is generated.

## Test Script

A complete test script is available at:
```bash
./scripts/train/debug/generic_mcp_weather_test.sh
```

This script:
1. Starts the weather server automatically
2. Runs GRPO training with the `generic_mcp` tool
3. Cleans up the server on exit
