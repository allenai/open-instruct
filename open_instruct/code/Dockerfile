# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY open_instruct/code/requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire package structure
COPY open_instruct/code /app/code

# Create cache directory for code execution
RUN mkdir -p cache && chmod 777 cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV PYTHONPATH=/app
ENV PYTHONFAULTHANDLER=1
ENV LOG_LEVEL=INFO
ENV PYTHONHASHSEED=random

# Expose the port the app runs on
EXPOSE ${PORT}

# Command to run the application with better process management
CMD uvicorn code.api:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --log-level info \
    --timeout-keep-alive 75 \
    --limit-concurrency 100 \
    --backlog 2048 \
    --limit-max-requests 10000