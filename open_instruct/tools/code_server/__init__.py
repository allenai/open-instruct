"""
Code execution server for the Python tool.

This is a standalone FastAPI server that executes Python code snippets.
It can be deployed as a Docker container or cloud service.

Usage:
    cd open_instruct/tools/code_server
    uv run uvicorn server:app --host 0.0.0.0 --port 1212

See server.py for full documentation and deployment instructions.
"""
