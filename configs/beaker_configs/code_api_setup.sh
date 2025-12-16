#!/bin/bash

set -e

# Configuration
TOTAL_CPUS=$(nproc)
CODE_SERVER_CPUS=128
STARTING_CPU=0
NGINX_PORT=8070
API_BASE_PORT=1234

# Get leader replica IP
BEAKER_LEADER_REPLICA_IP=$(getent hosts ${BEAKER_LEADER_REPLICA_HOSTNAME} | head -n 1 | awk '{print $1}')

# Set up environment
export PYTHONPATH=$REPO_PATH
export PATH="/root/.local/bin:$PATH"

# Increase the number of worker connections to 100000
sudo sed -i 's/worker_connections [0-9]*;/worker_connections 100000;/' /etc/nginx/nginx.conf

# Function to create the Nginx configuration file on head node
setup_nginx_head() {
    local config_file="/etc/nginx/conf.d/api_loadbalancer.conf"

    # Create upstream entries for all nodes
    upstream_entries=""

    # Add local servers
    for ((i=0; i<CODE_SERVER_CPUS; i++)); do
        PORT=$((API_BASE_PORT + i))
        upstream_entries+="    server 127.0.0.1:$PORT;\n"
    done

    # Create a proper configuration file
    cat > /tmp/api_loadbalancer.conf << EOF
upstream api_servers {
    least_conn;
$(echo -e "$upstream_entries")
}

server {
    listen $NGINX_PORT;
    client_max_body_size 0;
    location / {
        proxy_pass http://api_servers;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_connect_timeout 5s;
        proxy_send_timeout 5s;
        proxy_read_timeout 5s;
    }
}
EOF

    # Move the config file and restart Nginx
    sudo mv /tmp/api_loadbalancer.conf $config_file
    if [ -f /run/nginx.pid ] && [ -s /run/nginx.pid ]; then
        # Nginx is running, reload configuration
        sudo nginx -t && sudo nginx -s reload
    else
        # Nginx is not running or PID file is invalid, start it
        sudo nginx -t && sudo nginx
    fi
    echo "Nginx load balancer configured and started on port $NGINX_PORT"
}

# Function to start uvicorn instances on any node
start_uvicorn_servers() {
    echo "Starting API servers on $(hostname)"

    # Create logs directory
    mkdir -p api_logs

    # Start multiple uvicorn instances on separate cores
    for ((i=0; i<CODE_SERVER_CPUS; i++)); do
        CPU_ID=$((STARTING_CPU + i))
        PORT=$((API_BASE_PORT + i))

        echo "Starting API server on core $CPU_ID, port $PORT"

        # Use absolute path and better command structure
        cd "$REPO_PATH"

        # Start uvicorn with better error handling
        taskset -c $CPU_ID nohup uvicorn open_instruct.code_utils.api:app --host 0.0.0.0 --port $PORT > api_logs/api_server_$PORT.log 2>&1 &
        SERVER_PID=$!
        echo "API server started on port $PORT with PID $SERVER_PID"

        # Give it a moment to start
        if [ $i -eq 0 ]; then
            echo "Waiting for first server to start..."
            sleep 2
            # Test if the first server is responding
            if curl -s http://localhost:$PORT/health > /dev/null; then
                echo "✓ First API server is responding on port $PORT"
            else
                echo "⚠ First API server not responding on port $PORT, checking logs..."
                tail -5 api_logs/api_server_$PORT.log
            fi
        fi
    done

    echo "Waiting for all servers to start up..."
    sleep 5
}

# Export the API URL
export CODE_API_URL="http://${BEAKER_LEADER_REPLICA_IP}:$NGINX_PORT"

# Main execution
echo "Setting up Code API infrastructure"
echo "Working directory: $(pwd)"
echo "REPO_PATH: $REPO_PATH"
echo "PYTHONPATH: $PYTHONPATH"

start_uvicorn_servers
setup_nginx_head

# Test the API endpoint
echo "Testing API endpoint..."
echo "Testing localhost nginx: http://localhost:$NGINX_PORT/health"
if curl -s http://localhost:$NGINX_PORT/health > /dev/null; then
    echo "✓ Local nginx is responding"
else
    echo "⚠ Local nginx not responding"
fi

echo "Testing actual CODE_API_URL: $CODE_API_URL/health"
if curl -s $CODE_API_URL/health > /dev/null; then
    echo "✓ CODE_API_URL is responding correctly"
else
    echo "⚠ CODE_API_URL not responding, checking setup..."
    echo "BEAKER_LEADER_REPLICA_IP: $BEAKER_LEADER_REPLICA_IP"
    echo "NGINX_PORT: $NGINX_PORT"
    sudo nginx -t
    echo "Nginx processes:"
    ps aux | grep nginx
    echo "Sample API server logs:"
    ls api_logs/ | head -3 | xargs -I {} tail -3 api_logs/{}
fi

echo "Code API setup complete!"
echo "API URL: $CODE_API_URL"
