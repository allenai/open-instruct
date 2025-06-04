export CURRENT_DATETIME=$(python -c "import datetime; import pytz; print(datetime.datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%m%d%y_%H%M%S'))")
export PYTHONPATH=$REPO_PATH
export PATH="/root/.local/bin:$PATH"
export NCCL_CUMEM_ENABLE=0


echo CURRENT_DATETIME=$CURRENT_DATETIME
echo PYTHONPATH=$PYTHONPATH
echo PATH=$PATH

# python3 -c "import os, ray; print(os.path.dirname(ray.__file__))"

BEAKER_LEADER_REPLICA_IP=$(getent hosts ${BEAKER_LEADER_REPLICA_HOSTNAME} | awk '{print $1}')

RAY_NODE_PORT=8888

# Configuration for code server functionality
TOTAL_CPUS=$(nproc)
CODE_SERVER_CPUS=128
STARTING_CPU=0
NGINX_PORT=8070
API_BASE_PORT=1234

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
    location / {
        proxy_pass http://api_servers;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
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
    
    # Start multiple uvicorn instances on separate cores
    for ((i=0; i<CODE_SERVER_CPUS; i++)); do
        CPU_ID=$((STARTING_CPU + i))
        PORT=$((API_BASE_PORT + i))
        
        echo "Starting API server on core $CPU_ID, port $PORT"
        cd open_instruct/code
        uv run taskset -c $CPU_ID nohup uvicorn api:app --host 0.0.0.0 --port $PORT > api_server_$PORT.log 2>&1 &
        echo "API server started on port $PORT with PID $!"
        cd ../..
    done
}

export CODE_API_URL="http://${BEAKER_LEADER_REPLICA_IP}:$NGINX_PORT"

ray stop --force

if [ "$BEAKER_REPLICA_RANK" == "0" ]; then
    echo "Configuring head node"
    
    # Start uvicorn servers and nginx
    start_uvicorn_servers
    setup_nginx_head
    
    # Start Ray on the remaining cores
    RAY_CPUS=""
    for ((i=CODE_SERVER_CPUS; i<TOTAL_CPUS; i++)); do
        if [ -z "$RAY_CPUS" ]; then
            RAY_CPUS="$i"
        else
            RAY_CPUS="$RAY_CPUS,$i"
        fi
    done
    
    echo "Starting Ray head node on CPUs $RAY_CPUS"
    taskset -c $RAY_CPUS ray start --head --port=$RAY_NODE_PORT
else
    echo "Starting Ray worker node $BEAKER_REPLICA_RANK"
    ray start --address="${BEAKER_LEADER_REPLICA_IP}:${RAY_NODE_PORT}" --block
fi