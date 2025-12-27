#!/bin/bash
# Start the Python code execution server for local tool use
#
# Usage:
#   ./scripts/train/debug/start_code_server.sh          # Start on default port 1212
#   ./scripts/train/debug/start_code_server.sh 8080     # Start on port 8080
#   PORT=8080 ./scripts/train/debug/start_code_server.sh

PORT=${1:-${PORT:-1212}}

echo "Starting code execution server on port $PORT..."
echo "API endpoint: http://0.0.0.0:$PORT/execute"
echo ""
echo "Test with:"
echo "  curl -X POST http://localhost:$PORT/execute -H 'Content-Type: application/json' -d '{\"code\": \"print(1+1)\"}'"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd open_instruct/tools/code_server
uv run uvicorn server:app --host 0.0.0.0 --port "$PORT"

