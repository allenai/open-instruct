#!/bin/bash

set -e

# Configuration
TOTAL_CPUS=$(nproc)
BALLSIM_SERVER_CPUS=128
STARTING_CPU=0
NGINX_PORT=8072
API_BASE_PORT=3456

# Get leader replica IP
BEAKER_LEADER_REPLICA_IP=$(getent hosts ${BEAKER_LEADER_REPLICA_HOSTNAME} | head -n 1 | awk '{print $1}')

# Set up environment
export PYTHONPATH=$REPO_PATH
export PATH="/root/.local/bin:$PATH"

# Increase the number of worker connections to 100000
sudo sed -i 's/worker_connections [0-9]*;/worker_connections 100000;/' /etc/nginx/nginx.conf

setup_nginx_head() {
    local config_file="/etc/nginx/conf.d/ballsim_loadbalancer.conf"

    upstream_entries=""
    for ((i=0; i<BALLSIM_SERVER_CPUS; i++)); do
        PORT=$((API_BASE_PORT + i))
        upstream_entries+="    server 127.0.0.1:$PORT;\n"
    done

    cat > /tmp/ballsim_loadbalancer.conf << EOF
upstream ballsim_servers {
    least_conn;
$(echo -e "$upstream_entries")
}

server {
    listen $NGINX_PORT;
    client_max_body_size 0;
    location / {
        proxy_pass http://ballsim_servers;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_connect_timeout 5s;
        proxy_send_timeout 5s;
        proxy_read_timeout 5s;
    }
}
EOF

    sudo mv /tmp/ballsim_loadbalancer.conf $config_file
    if [ -f /run/nginx.pid ] && [ -s /run/nginx.pid ]; then
        sudo nginx -t && sudo nginx -s reload
    else
        sudo nginx -t && sudo nginx
    fi
    echo "Nginx load balancer configured and started on port $NGINX_PORT"
}

start_uvicorn_servers() {
    echo "Starting BounceSim API servers on $(hostname)"
    mkdir -p api_logs

    for ((i=0; i<BALLSIM_SERVER_CPUS; i++)); do
        CPU_ID=$((STARTING_CPU + i))
        PORT=$((API_BASE_PORT + i))

        echo "Starting BounceSim API server on core $CPU_ID, port $PORT"
        cd "$REPO_PATH"
        taskset -c $CPU_ID nohup uvicorn open_instruct.code_utils.ballsim_api:app --host 0.0.0.0 --port $PORT > api_logs/ballsim_server_$PORT.log 2>&1 &
        SERVER_PID=$!
        echo "BounceSim API server started on port $PORT with PID $SERVER_PID"

        if [ $i -eq 0 ]; then
            echo "Waiting for first server to start..."
            sleep 2
            if curl -s http://localhost:$PORT/health > /dev/null; then
                echo "First BounceSim API server is responding on port $PORT"
            else
                echo "First BounceSim API server not responding on port $PORT, checking logs..."
                tail -5 api_logs/ballsim_server_$PORT.log
            fi
        fi
    done

    echo "Waiting for all servers to start up..."
    sleep 5
}

export BALLSIM_API_URL="http://${BEAKER_LEADER_REPLICA_IP}:$NGINX_PORT"

echo "Setting up BounceSim API infrastructure"
echo "Working directory: $(pwd)"
echo "REPO_PATH: $REPO_PATH"
echo "PYTHONPATH: $PYTHONPATH"

start_uvicorn_servers
setup_nginx_head

echo "Testing API endpoint..."
echo "Testing localhost nginx: http://localhost:$NGINX_PORT/health"
if curl -s http://localhost:$NGINX_PORT/health > /dev/null; then
    echo "Local nginx is responding"
else
    echo "Local nginx not responding"
fi

echo "Testing actual BALLSIM_API_URL: $BALLSIM_API_URL/health"
if curl -s $BALLSIM_API_URL/health > /dev/null; then
    echo "BALLSIM_API_URL is responding correctly"
else
    echo "BALLSIM_API_URL not responding, checking setup..."
    echo "BEAKER_LEADER_REPLICA_IP: $BEAKER_LEADER_REPLICA_IP"
    echo "NGINX_PORT: $NGINX_PORT"
    sudo nginx -t
    echo "Nginx processes:"
    ps aux | grep nginx
    echo "Sample BounceSim API server logs:"
    ls api_logs/ | head -3 | xargs -I {} tail -3 api_logs/{}
fi

echo "BounceSim API setup complete!"
echo "API URL: $BALLSIM_API_URL"
