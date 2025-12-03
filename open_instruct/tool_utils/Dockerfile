# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.8.8 /uv /bin/uv

# Copy requirements first to leverage Docker cache
COPY open_instruct/tool_utils/requirements.txt requirements.txt

# Install dependencies using uv
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the tool server files
COPY open_instruct/__init__.py open_instruct/__init__.py
COPY open_instruct/logger_utils.py open_instruct/logger_utils.py
COPY open_instruct/tool_utils/tool_server.py tool_server.py

# Create cache directory for code execution
RUN mkdir -p cache && chmod 777 cache

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["uv", "run", "--no-project", "tool_server.py"]