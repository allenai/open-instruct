# Set up MCP Tools 

## Steps 

1. Download and install the mcp backend, install the dependencies, and run the following command to start the mcp server 
    ```bash
    fastmcp run rag_mcp/main.py:mcp --transport streamable-http --port 8000
    ```
2. Set the environment variables 
    ```bash 
    MCP_TRANSPORT="StreamableHttpTransport" 
    MCP_TRANSPORT_PORT=8000
    ```
3. To test the `tool_mcp.py`, simply run 
    ```bash 
    python open_instruct/tool_utils/tool_mcp.py
    # MCP_TRANSPORT="StreamableHttpTransport" MCP_TRANSPORT_PORT=8000 python open_instruct/tool_utils/tool_mcp.py
    ```