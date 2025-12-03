#!/bin/bash
set -e

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping local server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
    fi
}

trap cleanup EXIT

if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  echo "Error: Uncommitted changes detected. Please commit or stash before running."
  echo "------- git status (short) -------"
  git status --short
  exit 1
fi

echo "Building Docker image for ghcr.io..."
docker build -t ghcr.io/allenai/open-instruct/python-code-executor -f open_instruct/tool_utils/Dockerfile .

echo "Starting server locally on port 1212..."
docker run -p 1212:8080 -e OPEN_INSTRUCT_TOOL_API_KEY="$OPEN_INSTRUCT_TOOL_API_KEY" tool-server &
SERVER_PID=$!

echo ""
echo "========================================="
echo "Server started! (PID: $SERVER_PID)"
echo "========================================="
echo ""
echo "USAGE INSTRUCTIONS:"
echo "Test the server with the following commands to verify:"
echo "1) The timeout works correctly"
echo "2) The timeout in the first curl does not block the second curl"
echo ""
echo "Test 1 - This should timeout after 3 seconds:"
echo "curl -X POST http://localhost:1212/execute \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -H \"X-API-Key: \$OPEN_INSTRUCT_TOOL_API_KEY\" \\"
echo "     -d '{\"code\": \"import time;time.sleep(4)\", \"timeout\": 3}' \\"
echo "     -w '\\nTotal time: %{time_total}s\\n'"
echo ""
echo "Test 2 - This should complete quickly:"
echo "curl -X POST http://localhost:1212/execute \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -H \"X-API-Key: \$OPEN_INSTRUCT_TOOL_API_KEY\" \\"
echo "     -d '{\"code\": \"print(1)\", \"timeout\": 3}' \\"
echo "     -w '\\nTotal time: %{time_total}s\\n'"
echo ""
echo "========================================="
echo ""

read -p "Do you want to deploy to Google Cloud Run? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing Docker image to ghcr.io..."
    docker push ghcr.io/allenai/open-instruct/python-code-executor

    echo "Deploying to Google Cloud Run..."
    gcloud run deploy open-instruct-tool-server --project ai2-allennlp --region us-central1 --source .
fi

if [ -n "$BEAKER_TOKEN" ]; then
    echo ""
    read -p "BEAKER_TOKEN detected. Do you want to deploy to Beaker? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deploying to Beaker..."
        beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')
        beaker image delete $beaker_user/tool-server || true
        beaker image create tool-server -n tool-server -w ai2/$beaker_user
        uv run python mason.py \
            --cluster ai2/phobos-cirrascale \
            --workspace ai2/scaling-rl \
            --image $beaker_user/tool-server --pure_docker_mode \
            --priority high \
            --budget ai2/oe-adapt \
            --gpus 0 -- python tool_server.py
    fi
fi

echo ""
echo "Local server will be stopped automatically when script exits."
